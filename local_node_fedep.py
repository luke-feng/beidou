import lightning.pytorch as pl
import torch
import os, sys
import math
from mnistmodel import MNISTModelMLP
from subset import ChangeableSubset
from torch.utils.data import DataLoader, random_split
from fed_avg import fed_avg
from krum import krum
from trimmedmean import trimmedMean
from median import median
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Subset
import random
import numpy as np
from poisoning_attack import modelpoison
import copy
from util import cosine_metric, cosine_metric2, manhattan_metric, chebyshev_metric, pearson_correlation_metric, euclidean_metric
from sklearn.cluster import DBSCAN
from lightning.pytorch.strategies import DDPStrategy
from fmnistmodel import FashionMNISTModelMLP
from cifar10model import SimpleMobileNet
from syscallmodel import SYSCALLModelMLP
from data_util import DynamicDataLoader, DynamicDataset, dynamic_transformer
from itertools import product
from local_node import local_node


class local_node_fedep(local_node):

    epsilon_prime = 0.1

    def __init__(self,
        node_id: int,
        config: dict={},
        data_train: DataLoader=None , 
        data_val: DataLoader=None ,
        test_dataset: DataLoader=None,
        backdoor_valid_loader: DataLoader=None,
    ):
        local_node.__init__(self, node_id, config, data_train, data_val, test_dataset, backdoor_valid_loader)

        # FedEP specific parameters
        self._labels = np.array([label for _, label in self.train_dataloader.dataset])
        self.data_size = len(self.train_dataloader.dataset)

        self._distribution_fitting_max_components_fraction = 0.5
        self._EM_algorithm_max_iterations = 1500
        self._EM_algorithm_epsilon = 1e-6
        self._gaussian_epsilon = 1e-2

        self.theta = None
        self._thetas_with_samples_num = {}
        self._prediction_precision = 1e-3
        self._prediction_epsilon = 1e-2
        self._gaussian_epsilon = 1e-1
        self._theta_global = None
        self._prob_global = None
        self._clients_probs = None
        self._KL_divergence = None
        self._labels_unique = None
        self.alpha_k = None

        print(f"[NodeFedEP] Initializing FedEP node")

    def get_theta(self):
        return self.theta

    def get_data_size(self):
        return self.data_size

    def get_alpha_k(self):
        return self.alpha_k

    def aggregation(self, testing:bool=True):
        current_round_nei_models = self.nei_model_record[self.curren_round]
        nei_models_list = {}
        
        if self.curren_round > 0:
            # calculate the reputation score
            nei_reputation_score = self.cal_reputation(euclidean_metric)
            self.logger.log_metrics({"nei_reputation_score":nei_reputation_score})
            
            # get the reputation threshold
            rep_threshold, trigger = self.get_rep_threshold_trigger(nei_reputation_score)
            
            #MTD        
            if self.is_proactive:
                # proactive
                if self.dynamic_agg:
                    self.dynamic_aggregation()            
                if self.dynamic_topo:            
                    self.dynamic_topology(nei_reputation_score, rep_threshold)
            else:
                # reactive
                if trigger:
                    if self.dynamic_agg:
                        self.dynamic_aggregation()            
                    if self.dynamic_topo:            
                        self.dynamic_topology(nei_reputation_score, rep_threshold)
                    
        self.logger.log_metrics({"neiList":self.neiList})
        self.logger.log_metrics({"aggregation":self.curr_aggregation})
        for nei in current_round_nei_models:
            if nei in self.neiList:
                nei_models_list[nei] = current_round_nei_models[nei]            
            
        print(f"Node {self.node_id} aggregate {len(nei_models_list)} models with {self.neiList}")
        
        aggregated_model_para = self.aggregator(self.curr_aggregation, self.alpha_k , nei_models_list)
          
        self.aggregated_model.load_state_dict(aggregated_model_para)
        self.model.load_state_dict(aggregated_model_para)

        trainer = pl.Trainer(logger=self.logger,
                             max_epochs=self.maxEpoch, 
                             devices=1,
                             accelerator="cuda",
                             enable_progress_bar=False, 
                             enable_checkpointing=False,
                             )
        print(f"Performance of Node {self.node_id} after aggregation at round {self.curren_round}")
        trainer.test(self.model, self.test_dataloader)
        self.cal_backdoor_acc()
    


    #########################################################################################################################################################
    #    FedEP                                                                                                                                              #
    #  1. Fitting the data distribution of each client's dataset to a Gaussian Mixture Model (GMM) with an Expectation-Maximization (EM) method.
    #  2. Sharing the parameters of the GMM as statistical characteristics of each neighbers.
    #  3. Using the shared statistical characteristics to calculate the global distribution.
    #  4. Calculating the cross entropy (KL divergence) between the global distribution and each client's local distribution
    #  5. Using these cross entropy to calculate the parameters for each node.    
    #  6. Using these parameters in model aggregation.
    ##########################################################################################################################################################

    def fit_distribution(self):
        '''
        FedEP specific round of fitting the local distribution of the node with a GMMs model
        
        Args: Y: complete sorted of lable that is not unique, for example ([0,0,0,....,8,8,8,9,9,9])

        return:
            theta_hs: the parameters of the GMMs is a 3 x M matrix,  [π, μ, σ^2], where M is the number of mixture components.
                The mixture coefficient vector π = [π1, π2, . . . , πM ], with each element as the coefficient of the m-th Gaussian distribution.
                The vector μ = [μ1, μ2, . . . , μM ], with each element as the mean of the m-th Gaussian distribution. 
                The vector σ^2 = [σ^2_1 , σ^2_2 , . . . , σ^2_M ],with each element as the variance of the m-th Gaussian distribution. 
            likelihood: the likelihood of the data given theta_hs
        '''
        Y = np.array(np.sort(self._labels))
        print("Y",Y)
        # deciede the maximum number of mixture components
        Ms = math.ceil(len(set(Y.tolist())) * self._distribution_fitting_max_components_fraction) 
        theta_hs = np.empty(Ms, dtype=object)
        likelihood_hs = np.zeros(Ms)
        BICs = np.zeros(Ms)
        # AICs = np.zeros(Ms)
        for M in range(0,Ms):
            theta_hs[M], likelihood_hs[M] = self._expectation_maximum_algorithm(M+1 , Y)
            BICs[M] = -2*likelihood_hs[M] + M * np.log(len(Y))
            # AICs[M] = -2*likelihood_hs[M] + 2 * M
        min_BIC_index = np.argmin(BICs)
        print(f"[FedEP] local theta: {theta_hs[min_BIC_index]}")
        self.theta = theta_hs[min_BIC_index]
        print("theta",self.theta)
    
    def add_distribution(self, node_id, theta, data_size):
        """
        add distributions from other node to the collection of this node.

        Args:
            theta: local distribution of this client, represented by theta (a 3xM array).
            node_id: get the distribution from node node_id.
            datasize: Number of samples used to get the model.

        self._thetas_with_samples_num:
        {'10.10.10.10': ([[0.2, 9.0, 0.1], [0.8, 1.18181818, 1.33]], 100),
         '9.9.9.9': ([[0.3, 4.0, 0.1], [0.7, 5.118, 1.33]], 200)}
        """
        self._thetas_with_samples_num[node_id] = (theta, data_size)
        print(f"received (theta,data_size) from node {node_id}")
    
    def _expectation_maximum_algorithm(self, M, Y):
        '''
        derived theta by EM algorithm

        Args: M: the number of mixture components
              Y: complete list of lable that is ununique, for example ([0,0,0,....,8,8,8,9,9,9,])
        return:
            theta: the parameters of the GMMs given M and Y 
        '''
        theta = self._parameter_initialization(M,Y)
        likelihood_prev = 0
        theta_prev = theta
        iteration = 0
        while iteration <  self._EM_algorithm_max_iterations:
            gamma_lm, n_m = self._E_step(theta,Y)
            theta, likelihood = self._M_step(gamma_lm, n_m, Y)
            iteration += 1
            if likelihood == np.NINF or math.isnan(likelihood):
                return theta_prev, likelihood_prev
            if abs(likelihood - likelihood_prev) < self._EM_algorithm_epsilon:
                break
            likelihood_prev = likelihood
            theta_prev = theta
        return theta, likelihood

    def _parameter_initialization(self, M,Y):
        '''
        Initialized theta

        Args: M: the number of mixture components
              Y: complete list of lable that is ununique, for example ([0,0,0,....,8,8,8,9,9,9,])
        '''
        L=len(Y)
        # π (mixture weights)
        pi = np.random.rand(M)
        pi /= np.sum(pi)
        # μ (means)
        mu = Y[np.random.choice(L, M, replace=False)]
        # ϵ^2 (variances)
        sigma_squared = [np.var(Y.tolist())] * M
        # theta_0
        return np.column_stack((pi, mu, sigma_squared))
    
    def _gaussian(self, Y, mu, sigma_squared):
        '''
        probability density function of Gaussian distribution
        '''
        return np.exp(-np.square(Y-mu+self._gaussian_epsilon)/(2*(sigma_squared+self._gaussian_epsilon)))/(np.sqrt(2 * np.pi * (sigma_squared + self._gaussian_epsilon)))

    def _E_step(self, theta,Y):
        '''
        given theta, calculate the latent variable gamma_lm and the number of samples n_m for each mixture component m
        '''
        M = theta.shape[0]
        gamma_lm = np.zeros((len(Y),M))
        n_m = np.zeros(M)
        sum_gaussians = torch.zeros([len(Y)])
        for m in range(M):
            sum_gaussians += theta[m,0] * self._gaussian(Y, theta[m,1], theta[m,2])
        for m in range(M):
            gamma_lm[:,m] = theta[m,0] * self._gaussian(Y, theta[m,1], theta[m,2])/ sum_gaussians
            n_m[m] = np.sum(gamma_lm[:,m])
        return gamma_lm, n_m
    
    def _M_step(self, gamma_lm, n_m, Y):
        '''
        given gamma and n_m, calculate the new theta
        '''
        pi_h = n_m / len(Y)
        mu_h = np.array([gamma_lm[:,m] @ Y / n_m[m] for m in range(len(n_m))])
        sigma_squared_h = np.array([gamma_lm[:,m] @ ((Y-mu_h[m])**2 + self._gaussian_epsilon) / n_m[m] for m in range(len(n_m))])
        theta_h = np.column_stack((pi_h, mu_h, sigma_squared_h))

        likelihood = np.sum([n_m[m] * np.log(pi_h[m])+ gamma_lm[:,m] @ np.log(self._gaussian(Y, mu_h[m], sigma_squared_h[m])+self._gaussian_epsilon) for m in range(len(n_m))])
        return theta_h, likelihood

    
    def _predict_likelihood(self, theta, precision=4):
        '''
        given theta and labels, calculate the likelihood of the labels
        '''
        prob = np.zeros(len(self._labels_unique))
        for i in range(len(prob)): 
            prob[i] = np.round(np.sum(theta[:,0] * self._gaussian(self._labels_unique[i], theta[:,1], theta[:,2])), precision)
        return prob
    
    def pooling(self, global_unique_targets):
        '''
        Examples:

        self._theta_global:
        {'10.10.10.10': [[0.2*(1/3), 9.0, 0.1], [0.8*(1/3), 1.18181818, 1.33]],
         '9.9.9.9': [[0.3*(2/3), 4.0, 0.1], [0.7*(2/3), 5.118, 1.33]]}

        self._prob_global:
        [0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.7] (len(self._prob_global)=10 && sum(self._prob_global)=1)

        self._clients_probs:
        {'10.10.10.10': [0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.7],
         '9.9.9.9':[ 0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0.7]}

        self._KL_divergence:
        {'10.10.10.10': 3, '9.9.9.9': 2}

        self.alpha_k:
        {'10.10.10.10': 3/(2+3), '9.9.9.9': 2/(2+3)}

        '''
        # Total Samples
        # get the unique labels from the LOCAL labelset self._labels
        print("----------------------------------------------------")
        print("self._thetas_with_samples_num.", self._thetas_with_samples_num)
        print("----------------------------------------------------")

        self._labels_unique = global_unique_targets
        print("label_unique: ",global_unique_targets)

        total_samples = sum([datasize for _, datasize in self._thetas_with_samples_num.values()])
        print(f"[FedEP]total_samples: {total_samples}")

        q_k = {[k][0]: v[1]/total_samples for k, v in self._thetas_with_samples_num.items()}
        
        self._theta_global = {
            node_id: np.array([
                [round(param[0] * (datasize / total_samples), 5), param[1], param[2]]
                for param in theta
            ])
            for node_id, (theta, datasize) in self._thetas_with_samples_num.items()
        }
        print(f"[FedEP]theta_global: {self._theta_global}")

        # Calculate global probability of labels
        '''
        self._prob_global = [] with a length equals number of labels space and summing up to 1
        '''
        prob_global = []
        for label in self._labels_unique:
            prob_label = 0
            for theta in self._theta_global.values():
                for g in theta:
                    prob_label +=  g[0] * self._gaussian(label, g[1], g[2]) 
            prob_global.append(prob_label)
        self._prob_global = prob_global

        # Calculate client probabilities
        self._clients_probs = { 
            node_id : self._predict_likelihood(theta)
            for node_id,theta in self._theta_global.items()
        }
        print(f"[FedEP]clients_probs: {self._clients_probs}")
        
        # Calculate KL divergences
        self._KL_divergence = {
            node_id: np.sum([self._prob_global[i] * np.log2((self._prob_global[i] + self._prediction_epsilon) / (probs[i] + self._prediction_epsilon)) for i in range(len(self._prob_global))])
            for node_id,probs in self._clients_probs.items()
        }
        print("----------------------------------------------------")
        print(f"[FedEP]KL_divergence: {self._KL_divergence}")
        print("----------------------------------------------------")

        # Calculate alpha_k
        kl_sum = np.sum([kl_div for kl_div in self._KL_divergence.values()])
        self.alpha_k = {
            node_id: kl_div / kl_sum
            for node_id, kl_div in self._KL_divergence.items()
        }
        print("----------------------------------------------------")
        print(f"[FedEP]alpha_k: {self.alpha_k}")
        print("----------------------------------------------------")
        

  
    
    



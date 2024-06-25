import lightning.pytorch as pl
import torch
import os, sys
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

class local_node():
    def __init__(
            self,
            node_id: int,
            config: dict={},
            data_train: DataLoader=None , 
            data_val: DataLoader=None ,
            test_dataset: DataLoader=None,
            backdoor_valid_loader: DataLoader=None,

    ):
        # basic info
        self.node_id = node_id
        self.config = config
        
        # personal info
        self.indices = self.config['indices']
        self.experimentsName = self.config['experimentsName']  
        self.experimentsName_path = self.config['experimentsName_path']          
        self.dataset_name = self.config['dataset_name']
        self.neiList = self.config['neiList']
        if self.node_id not in self.neiList:
            self.add_nei_to_neiList(self.node_id)
        
        self.maxRound = self.config['maxRound']
        self.maxEpoch = self.config['maxEpoch']
        self.num_peers = self.config['num_peers']
        self.curr_aggregation = self.config['aggregation']
        
        # attack info
        self.attack_type = self.config['attack_type']
        self.targeted = self.config['targeted']
        self.noise_injected_ratio = self.config['noise_injected_ratio']
        self.poisoned_sample_ratio = self.config['poisoned_sample_ratio']
        self.model_poison = False
        
        if self.attack_type.lower() == "model poisoning":
            self.model_poison = True
        
        # mtd
        self.dynamic_topo = self.config['dynamic_topo']
        self.dynamic_agg = self.config['dynamic_agg']
        self.dynamic_data = self.config['dynamic_data']
        self.is_proactive = self.config['is_proactive']
        
        
        self.candidate_threshold = int(self.num_peers/2)
        self.reputation_threshold = 0.5
        
        # logger
        self.logger = CSVLogger(save_dir=self.experimentsName_path, name=f"node_{node_id}")
        self.logger.log_metrics(config)

        # dataset and model
        if self.dataset_name == "MNIST":            
            self.model = MNISTModelMLP()
            self.aggregated_model = MNISTModelMLP()
        if self.dataset_name == "FashionMNIST":            
            self.model = FashionMNISTModelMLP()
            self.aggregated_model = FashionMNISTModelMLP()
        if self.dataset_name == "Cifar10":            
            self.model = SimpleMobileNet()
            self.aggregated_model = SimpleMobileNet()
        if self.dataset_name == "Syscall":            
            self.model = SYSCALLModelMLP()
            self.aggregated_model = SYSCALLModelMLP()
        
        self.nei_models = {}
        self.train_dataloader = data_train
        self.val_dataloader = data_val
        self.test_dataloader = test_dataset
        self.backdoor_valid_dataloader = backdoor_valid_loader
        self.target_label = backdoor_valid_loader.dataset.target_label

        self.train_dataset = data_train.dataset
        self.val_dataset = data_val.dataset
        self.test_dataset = test_dataset.dataset
        self.backdoor_valid_dataset = backdoor_valid_loader.dataset

        
        # run time record        
        self.curren_round = 0      

        self.local_model_record = {}
        self.local_model_record[0] = self.model
        self.aggregated_model_record = {}
        self.aggregated_model_record[0] = self.aggregated_model
        self.nei_model_record = {}
        self.nei_reputation_score_record = {} 

    def get_model(self):
        model_info = self.model        
        return model_info
    
    def get_model_param(self):
        model_param = copy.deepcopy(self.model.state_dict())
        return model_param
    
    def set_aggregation_fun(self, aggregation_fun:str):
        self.curr_aggregation = aggregation_fun
    
    def next_round(self):
        self.curren_round += 1

    def get_current_round(self):
        return self.curren_round

    def set_model(self, round, model):
        self.model_record[round] = model
    
    def set_current_model(self, model):
        self.model = model

    def set_current_aggregated_model(self, model):
        self.aggregated_model = model
    
    def replace_local_aggregated_model(self):
        self.model = self.aggregated_model

    def get_neiList(self):
        return self.neiList

    def set_neiList(self, new_neiList):
        self.neiList = new_neiList
    
    def add_nei_to_neiList(self, nei_id):
        if nei_id not in self.neiList:
            self.neiList.append(nei_id)
    
    def remove_nei_from_neiList(self, nei_id):
        if nei_id in self.neiList:
            self.neiList.remove(nei_id)
    
    def add_nei_model(self, round, nei_id, nei_model):
        # print(f"I am node {self.node_id}, I am adding model from node {nei_id} to my list")
        if round in self.nei_model_record:
            self.nei_model_record[round][nei_id]=nei_model
        else:
            self.nei_model_record[round] = {}
            self.nei_model_record[round][nei_id]=nei_model
       
    def local_training(self,with_checkpoints:bool=False):
        # trainer = pl.Trainer(max_epochs=self.maxEpoch, accelerator='cuda', devices=-1) 
        # ddp = DDPStrategy(process_group_backend="gloo")
        trainer = pl.Trainer(logger=self.logger,
                             max_epochs=self.maxEpoch, 
                             devices=1,
                             accelerator="cuda",
                             enable_progress_bar=False, 
                             enable_checkpointing=False,
                            #  strategy=ddp
                             )
        
        if self.dynamic_data:
            applier = dynamic_transformer(self.train_dataset.dataset[0][0].shape[-1], self.train_dataset.dataset[0][0].shape[-1])
            print(f"current applier{applier}")
            data_train_dynamic = DynamicDataset(self.train_dataset, applier)
            data_val_dynamic = DynamicDataset(self.val_dataset, applier)
            # test_dataset_dynamic = DynamicDataset(self.test_dataset, applier)
            # backdoor_valid_dataset_dynamic = DynamicDataset(self.backdoor_valid_dataset, applier)

            self.train_dataloader = DataLoader(data_train_dynamic, batch_size=64, shuffle=True )
            self.val_dataloader = DataLoader(data_val_dynamic, batch_size=64,shuffle=False)
            # self.test_dataloader = DataLoader(test_dataset_dynamic, batch_size=64,shuffle=False)
            # self.backdoor_valid_dataloader = DataLoader(backdoor_valid_dataset_dynamic, batch_size=64,shuffle=False)


        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)
        

        
        # if model poisoning attack, change the model before send to others
        if self.model_poison and self.curren_round >= 1:
            cur_model_para = self.get_model_param()
            poisoned_model_dict = modelpoison(cur_model_para, self.noise_injected_ratio)
            self.model.load_state_dict(poisoned_model_dict)
        
        print(f"Performance of Node {self.node_id} before aggregation at round {self.curren_round}")
        trainer.test(self.model, self.test_dataloader)
        self.cal_backdoor_acc()

        if with_checkpoints:
            trainer.save_checkpoint(f"{self.experimentsName_path}/checkpoint_{self.experimentsName}_node_{self.node_id}_round_{self.curren_round}.ckpt")
    
    def cal_backdoor_acc(self):
        data_loader = self.backdoor_valid_dataloader

        all_targets, all_predictions = [], []
        with torch.no_grad():
            for i, (features, targets) in enumerate(data_loader):
                logits = self.model(features)
                _, predicted_labels = torch.max(logits, 1)
                all_targets.extend(targets.detach().cpu())
                all_predictions.extend(predicted_labels.detach().cpu())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
        if class_labels.shape[0] == 1:
            if class_labels[0] != 0:
                class_labels = np.array([0, class_labels[0]])
            else:
                class_labels = np.array([class_labels[0], 1])
        n_labels = class_labels.shape[0]
        lst = []
        z = list(zip(all_targets, all_predictions))
        for combi in product(class_labels, repeat=2):
            lst.append(z.count(combi))
        confmat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)

        target_label = self.target_label
        num_predicted_target = confmat.sum(axis=0)[target_label] - confmat.item((target_label, target_label))
        num_samples = len(data_loader.dataset) - confmat.item((target_label, target_label))
        attacker_success = num_predicted_target / num_samples
        self.logger.log_metrics({"Test/ASR-backdoor": attacker_success})
        print("Computed ASR Backdoor: {}".format(attacker_success))
        return confmat
        

    def aggregator(self, func, *args):
        return func(*args)

    def dynamic_aggregation(self):
        aggregation_list = [krum, trimmedMean, median]
        random_index = random.randint(0, len(aggregation_list)-1)
        self.curr_aggregation = aggregation_list[random_index]
    
    def get_rep_threshold_trigger(self, nei_reputation_score, sensitive=0.1):
        trigger = False
        rep_threshold = 0
        
        repList = list(nei_reputation_score.values())       
        max_value = int(max(repList))
        if max_value==1:
            repList.remove(max_value)
        
        X = np.array(list(repList)).reshape(-1, 1)
        nan_mask = np.isnan(X)
        X = X[~nan_mask]
        db = DBSCAN(eps=sensitive, min_samples=1)
        clusters = db.fit_predict(X.reshape(-1,1))
        unique_labels = set(clusters)
        
        if len(unique_labels) > 1:
            print(f"[MTD] malicious detected!")
            trigger = True            
        
            lowerbound = []
            upperbound = []
            
            for label in unique_labels:
                cluster_points = X[clusters == label]
                lowerbound.append(min(cluster_points))
                upperbound.append(max(cluster_points))
            
            # the threshold is the average of lowerbound of the first class and upperbound of the last class
            lowerbound.sort()
            upperbound.sort()
            rep_threshold = np.mean([lowerbound[-1], upperbound[0]])
            
        return rep_threshold, trigger
        
        
    def cal_reputation(self, reputation_func=euclidean_metric):
        nei_reputation_score = {}
        model_param = self.get_model_param()
        current_round_nei_models = self.nei_model_record[self.curren_round]
        for nei in current_round_nei_models:
            nei_model_param = current_round_nei_models[nei]
            nei_reputation_score[nei] = reputation_func(model_param, nei_model_param)
        print(f"[Reputation] in {self.node_id}: {nei_reputation_score}")
        return nei_reputation_score
    
    def dynamic_topology(self, nei_reputation_score, rep_threshold):
        if self.attack_type.lower() == 'no_attack':
            connected_node = len(self.neiList)
            self.reputation_threshold = rep_threshold
            print(f"[Reputation_threshold] in {self.node_id}: {self.reputation_threshold}")
            node_list = list(nei_reputation_score.keys())
            random.shuffle(node_list)
            for node_id in node_list:
                if node_id in self.neiList:
                    if nei_reputation_score[node_id] < self.reputation_threshold:
                        self.remove_nei_from_neiList(node_id)
                        print(f"[dynamic_topology] in {self.node_id}: remove {node_id} from nei list")
                        connected_node -= 1
                else:
                    if connected_node <= self.candidate_threshold and nei_reputation_score[node_id] >= self.reputation_threshold:
                        connected_node += 1
                        self.add_nei_to_neiList(node_id)
                        print(f"[dynamic_topology] in {self.node_id}: add {node_id} from nei list")
    
    def aggregation(self, testing:bool=True):
        current_round_nei_models = self.nei_model_record[self.curren_round]
        nei_models_list = []
        
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
                nei_models_list.append(current_round_nei_models[nei])                     
            
        print(f"Node {self.node_id} aggregate {len(nei_models_list)} models with {self.neiList}")
        
        aggregated_model_para = self.aggregator(self.curr_aggregation, nei_models_list)
          
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


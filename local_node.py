import lightning.pytorch as pl
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

from util import cosine_metric, cosine_metric2, manhattan_metric, chebyshev_metric, pearson_correlation_metric, euclidean_metric


class local_node():
    def __init__(
            self,
            node_id: int,
            config: dict={},
            data_train: DataLoader=None , 
            data_val: DataLoader=None ,
            test_dataset: DataLoader=None 

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
            self.neiList.append(self.node_id)
        
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
        self.is_proactive = self.config['is_proactive']
        
        self.candidate_threshold = int(self.num_peers/2)
        self.reputation_threshold = 0.5
        
        # logger
        self.logger = CSVLogger(save_dir=self.experimentsName_path, name=f"node_{node_id}")  

        # dataset and model
        if self.dataset_name == "MNIST":            
            self.model = MNISTModelMLP()
            self.aggregated_model = MNISTModelMLP()
        
        self.nei_models = {}
        self.data_train = data_train
        self.data_val = data_val
        self.test_dataset = test_dataset

        
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
    
    def add_nei_model(self, round, nei_id, nei_model):
        if round in self.nei_model_record:
            self.nei_model_record[round][nei_id]=nei_model
        else:
            self.nei_model_record[round] = {}
            self.nei_model_record[round][nei_id]=nei_model
       
    def local_training(self):
        # trainer = pl.Trainer(max_epochs=self.maxEpoch, accelerator='cuda', devices=-1) 
        trainer = pl.Trainer(logger=self.logger,
                             max_epochs=self.maxEpoch, 
                             devices=1,
                             accelerator="cuda",
                             enable_progress_bar=False)
        
        trainer.fit(self.model, train_dataloaders=self.data_train, val_dataloaders=self.data_val)

        
        # if model poisoning attack, change the model before send to others
        if self.model_poison:
            poisoned_model_dict = modelpoison(self.model.state_dict(), self.noise_injected_ratio)
            self.model.load_state_dict(poisoned_model_dict)
        
        print(f"Performance of Node {self.node_id} before aggregation at round {self.curren_round}")
        trainer.test(self.model, self.test_dataset)

        trainer.save_checkpoint(f"{self.experimentsName_path}/checkpoint_{self.experimentsName}_node_{self.node_id}_round_{self.curren_round}.ckpt")
        
    def aggregator(self, func, *args):
        return func(*args)

    def dynamic_aggregation(self):
        aggregation_list = [krum, trimmedMean, median]
        random_index = random.randint(0, len(aggregation_list)-1)
        self.curr_aggregation = aggregation_list[random_index]
        
    def cal_reputation(self, reputation_func=cosine_metric):
        nei_reputation_score = {}
        model_param = self.model.state_dict()
        current_round_nei_models = self.nei_model_record[self.curren_round]
        for nei in current_round_nei_models:
            nei_model_param = current_round_nei_models[nei].state_dict()
            nei_reputation_score[nei] = reputation_func(model_param, nei_model_param)
        return nei_reputation_score
    
    def dynamic_topology(self, nei_reputation_score):
        if self.attack_type.lower() == 'no attack':
            connected_node = len(self.neiList)
            for node_id in nei_reputation_score:
                if node_id in self.neiList and nei_reputation_score[node_id] < self.reputation_threshold:
                    self.neiList.remove(node_id)
                    connected_node -= 1
                if node_id not in self.neiList and connected_node < self.candidate_threshold and nei_reputation_score[node_id] >= self.reputation_threshold:
                    connected_node += 1
                    self.neiList.append(node_id)
                
        
    
    def aggregation(self):
        current_round_nei_models = self.nei_model_record[self.curren_round]
        nei_models_list = []
        
        #MTD
        if self.dynamic_agg:
            self.dynamic_aggregation()
        
        if self.dynamic_topo:
            nei_reputation_score = self.cal_reputation()
            self.dynamic_topology(nei_reputation_score)
        
        for nei in current_round_nei_models:
            if nei in self.neiList:
                nei_models_list.append(current_round_nei_models[nei].state_dict())                     
            
        print(f"Node {self.node_id} aggregate {len(nei_models_list)} models with {self.neiList}")
        
        aggregated_model_para = self.aggregator(self.curr_aggregation, nei_models_list)
            
        # if self.curr_aggregation == "fed_avg":        
        #     aggregated_model_para = self.aggregator(fed_avg, nei_models_list)
        # elif self.curr_aggregation == "krum":        
        #     aggregated_model_para = self.aggregator(krum, nei_models_list)    
        # elif self.curr_aggregation == "trimmedmean":        
        #     aggregated_model_para = self.aggregator(trimmedMean, nei_models_list)
        # elif self.curr_aggregation == "median":        
        #     aggregated_model_para = self.aggregator(median, nei_models_list)
          
        self.aggregated_model.load_state_dict(aggregated_model_para)
        self.model.load_state_dict(aggregated_model_para)

        trainer = pl.Trainer(logger=self.logger,
                             max_epochs=self.maxEpoch, 
                             devices=1,
                             accelerator="cuda",
                             enable_progress_bar=False) 
        print(f"Performance of Node {self.node_id} after aggregation at round {self.curren_round}")
        trainer.test(self.model, self.test_dataset)


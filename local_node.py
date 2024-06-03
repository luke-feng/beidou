import lightning.pytorch as pl
import os, sys
from mnistmodel import MNISTModelMLP
from subset import ChangeableSubset
from torch.utils.data import DataLoader, random_split
from fed_avg import fed_avg
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Subset
import random
import numpy as np

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
        self.maxRound = self.config['maxRound']
        self.maxEpoch = self.config['maxEpoch']
        self.num_peers = self.config['num_peers']

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


    def get_model(self):
        model_info = self.model        
        return model_info
    
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
                             enable_progress_bar=False,
                            )
        
        trainer.fit(self.model, train_dataloaders=self.data_train, val_dataloaders=self.data_val)

        print(f"Performance of Node {self.node_id} before aggregation at round {self.curren_round}")
        trainer.test(self.model, self.test_dataset)

        trainer.save_checkpoint(f"{self.experimentsName_path}/checkpoint_{self.experimentsName}_node_{self.node_id}_round_{self.curren_round}.ckpt")


    
    def aggregation(self):
        current_rount_nei_models = self.nei_model_record[self.curren_round]
        nei_models_list = []
        
        for nei in current_rount_nei_models:
            if nei in self.neiList:
                nei_models_list.append(current_rount_nei_models[nei].state_dict())        
        if self.node_id not in self.neiList:
            nei_models_list.append(self.model.state_dict())
            self.neiList.append(self.node_id)
            
        print(f"Node {self.node_id} aggregate model with {self.neiList}")
        aggregated_model_para = fed_avg(nei_models_list)     
        self.aggregated_model.load_state_dict(aggregated_model_para)
        self.replace_local_aggregated_model()

        trainer = pl.Trainer(logger=self.logger,
                             devices=1,
                             accelerator="cuda",) 
        print(f"Performance of Node {self.node_id} after aggregation at round {self.curren_round}")
        trainer.test(self.model, self.test_dataset)

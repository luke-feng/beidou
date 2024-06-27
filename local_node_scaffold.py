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
import torch.nn.functional as F



def create_scaffold_model_class(base_class):
    class ScaffoldModel(base_class):
        def __init__(self):
            super(ScaffoldModel, self).__init__()
            self.automatic_optimization = False 

            self.client_control_variate = {}
            self.global_control_variate = {}
        
        def set_client_control_variate(self, client_control_variate):
            self.client_control_variate = client_control_variate
        
        def set_global_control_variate(self, global_control_variate):
            self.global_control_variate = global_control_variate
        
        # def step(self, batch, phase):
        #     images, labels = batch
        #     images = images.to(self.device)
        #     labels = labels.to(self.device)
        #     y_pred = self.forward(images)
        #     loss = self.criterion(y_pred, labels)

        #     # Get metrics for each batch and log them
        #     self.log(f"{phase}/Loss", loss, prog_bar=False, sync_dist=True)
        #     self.process_metrics(phase, y_pred, labels, loss)

        #     return loss
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            x = x.to(super().device)
            y = y.to(super().device)
            
            y_pred = super().forward(x)
            # print(f"y_hat: {y_hat}")

            loss = self.criterion(y_pred, y)
            # print(f"loss: {loss}")
            # Get metrics for each batch and log them
            self.log(f"{"Train"}/Loss", loss, prog_bar=False, sync_dist=True)
            super().process_metrics("Train", y_pred, y, loss)

            optimizer = self.optimizers()  # Retrieve the optimizer
            optimizer.zero_grad()  # Zero out the gradients

            # Perform the forward pass
            super().manual_backward(loss)  # Perform the backward pass

            # Initialize control variates if empty with zeros of the same size as gradients
            if self.client_control_variate=={}:
                self.client_control_variate = {name: torch.zeros_like(param) for name, param in self.named_parameters() if param.grad is not None}
            if self.global_control_variate=={}:
                self.global_control_variate = {name: torch.zeros_like(param) for name, param in self.named_parameters() if param.grad is not None}

            # Iterate over all model parameters to update them according to the SCAFFOLD algorithm
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # Get the current gradient
                    grad = param.grad

                    # Get the client and global control variates
                    device = param.data.device 
                    client_cv = self.client_control_variate.get(name).to(device)                  
                    global_cv = self.global_control_variate.get(name).to(device)

                    # Update the parameter according to the SCAFFOLD algorithm
                    eta = optimizer.param_groups[0]['lr']  # Get the learning rate
                    # print(f"eta: {eta}")
                    # print(f"Gradient for {name}: {param.grad}")
                    # print(f"global_cv: {global_cv}")
                    # print(f"client_cv: {client_cv}")
                    param.data -= eta * (grad + global_cv - client_cv)


            # Return the loss to be logged
            return loss
        
        # def configure_optimizers(self):
        #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #     return optimizer
        
        def update_client_control_variate(self, x_params, y_params, local_step_number):
            """
            Update client_control_variate after training.

            Args:
            - x_params (dict): Dictionary of parameter tensors before training.
            - y_params (dict): Dictionary of parameter tensors after training.
            - local_step_number (int): Local step number of this training.

            """
            # if not self.client_control_variate:
            #     self.client_control_variate = {name: torch.zeros_like(param) for name, param in x_params.items()}

            with torch.no_grad():
                for name, param in self.named_parameters():
                    device = param.data.device 
                    if name in self.client_control_variate:
                        x = x_params[name].to(device)
                        yi = y_params[name].to(device)
                        self.client_control_variate[name] -= (1 / local_step_number) * (x - yi)
                    

    return ScaffoldModel

class local_node_scaffold(local_node):

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

        # Scaffold specific parameters
        self.nei_control_variates = {}
        self.client_control_variate = {}
        self.global_control_variate = {}

        # Scaffold specific dataset and model
        if self.dataset_name == "MNIST":            
            self.model = create_scaffold_model_class(MNISTModelMLP)()
            self.aggregated_model = create_scaffold_model_class(MNISTModelMLP)()
        if self.dataset_name == "FashionMNIST":            
            self.model = create_scaffold_model_class(FashionMNISTModelMLP)()
            self.aggregated_model = create_scaffold_model_class(FashionMNISTModelMLP)()
        if self.dataset_name == "Cifar10":            
            self.model = create_scaffold_model_class(SimpleMobileNet)()
            self.aggregated_model = create_scaffold_model_class(SimpleMobileNet)()
        if self.dataset_name == "Syscall":            
            self.model = create_scaffold_model_class(SYSCALLModelMLP)()
            self.aggregated_model = create_scaffold_model_class(SYSCALLModelMLP)()


    def add_nei_control_variates(self, round, nei_id, nei_control_variates):
        # print(f"I am node {self.node_id}, I am adding control_variates from node {nei_id} to my list")
        if round in self.nei_control_variates:
            self.nei_control_variates[round][nei_id]=nei_control_variates
        else:
            self.nei_control_variates[round] = {}
            self.nei_control_variates[round][nei_id]=nei_control_variates

    def aggregation(self, testing:bool=True):
        current_round_nei_models = self.nei_model_record[self.curren_round]
        current_round_nei_control_variates = self.nei_control_variates.get(self.curren_round, {})
        nei_models_list = []
        nei_control_variates = {}
        
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

        for nei,v in current_round_nei_control_variates.items():
            if nei in self.neiList:
                nei_control_variates[nei] = v

            
        print(f"Node {self.node_id} aggregate {len(nei_models_list)} models with {self.neiList}")
        

        print(f"current_round_nei_control_variates: {current_round_nei_control_variates.keys()}")
        print(f"nei_control_variates: {nei_control_variates.keys()}")


        aggregated_model_para, aggregated_control_variates = self.aggregator(self.curr_aggregation, nei_models_list, nei_control_variates)
        
        # assign the return control variates to the global control variates
        self.global_control_variate = aggregated_control_variates
        self.aggregated_model.set_global_control_variate(aggregated_control_variates)
        self.model.set_global_control_variate(aggregated_control_variates)

        # assign the aggregated model to the model
        self.aggregated_model.load_state_dict(aggregated_model_para)
        self.model.load_state_dict(aggregated_model_para)

        trainer = pl.Trainer(logger=self.logger,
                             max_epochs=self.maxEpoch, 
                             devices=1,
                             accelerator="cpu",
                             enable_progress_bar=False, 
                             enable_checkpointing=False,
                             )
        print(f"Performance of Node {self.node_id} after aggregation at round {self.curren_round}")
        trainer.test(self.model, self.test_dataloader)
        self.cal_backdoor_acc()

      
    def local_training(self,with_checkpoints:bool=False):
        # trainer = pl.Trainer(max_epochs=self.maxEpoch, accelerator='cuda', devices=-1) 
        # ddp = DDPStrategy(process_group_backend="gloo")
        trainer = pl.Trainer(logger=self.logger,
                             max_epochs=self.maxEpoch, 
                             devices=1,
                             accelerator="cpu",
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

        print(f"round {self.curren_round}, node {self.node_id}")
        
        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        # after fitting, update the client control variates
        self.model.update_client_control_variate(self.model.state_dict(), self.aggregated_model.state_dict(), self.maxEpoch)
        self.client_control_variate = self.model.client_control_variate
        
        # if model poisoning attack, change the model before send to others
        if self.model_poison and self.curren_round >= 1:
            cur_model_para = self.get_model_param()
            poisoned_model_dict = modelpoison(cur_model_para, self.noise_injected_ratio)
            self.model.load_state_dict(poisoned_model_dict)
        
        print(f"Performance of Node {self.node_id} before aggregation at round {self.curren_round}")
        print("testing")
        trainer.test(self.model, self.test_dataloader)
        self.cal_backdoor_acc()

        if with_checkpoints:
            trainer.save_checkpoint(f"{self.experimentsName_path}/checkpoint_{self.experimentsName}_node_{self.node_id}_round_{self.curren_round}.ckpt")
    


        
import ray
import lightning.pytorch as pl
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import wandb
from mnistmodel import MNISTModelMLP
from subset import ChangeableSubset
from torch.utils.data import DataLoader, random_split
from fed_avg import fed_avg

@ray.remote
class local_node():
    def __init__(
            self,
            node_id: int,
            experimentsName=None,
            maxRound: int = 10,
            maxEpoch: int = 3,
            train_dataset=None,
            test_dataset=None,
            indices=[],
            neiList=[],
            experimentsName_path = None,
            logger:wandb=None
    ):
        self.node_id = node_id
        self.indices = indices
        self.logger=logger
        self.model = MNISTModelMLP()
        self.neiList = neiList
        self.maxRound = maxRound
        self.maxEpoch = maxEpoch
        self.experimentsName = experimentsName
        self.nei_models = {}
        self.dataset = train_dataset
        self.test_dataset = test_dataset
        
        tr_subset = ChangeableSubset(
            self.dataset, indices)
        self.data_train, self.data_val = random_split(
                    tr_subset,
                    [
                        int(len(tr_subset) * 0.8),
                        len(tr_subset) - int(len(tr_subset) * 0.8),
                    ],
                )

        
        self.curren_round = 0
        self.aggregated_model = MNISTModelMLP()

        self.local_model_record = {}
        self.local_model_record[0] = self.model

        self.aggregated_model_record = {}
        self.aggregated_model_record[0] = self.aggregated_model

        self.nei_model_record = {}  
        self.experimentsName_path = experimentsName_path      


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
       
    def train_func_per_worker(self):
        # trainer = pl.Trainer(max_epochs=self.maxEpoch, accelerator='cuda', devices=-1) 
        ckpt_report_callback = RayTrainReportCallback()
        ray_lightning_environment = RayLightningEnvironment()
        ray_DDPStrategy = RayDDPStrategy()
        trainer = pl.Trainer(max_epochs=self.maxEpoch, 
                             devices="auto",
                             accelerator="auto",
                             strategy=ray_DDPStrategy,
                             callbacks=[ckpt_report_callback],
                             plugins=[ray_lightning_environment],
                             enable_progress_bar=False,
                            )
        trainer = prepare_trainer(trainer) 
        trainer.fit(self.model, train_dataloaders=DataLoader(self.data_train, batch_size=64, shuffle=True))

        print(f"Performance of Node {self.node_id} before aggregation at round {self.curren_round}")
        trainer.test(self.model, DataLoader(self.test_dataset, batch_size=64,shuffle=False))

        trainer.save_checkpoint(f"{self.experimentsName_path}/checkpoint_{self.experimentsName}_node_{self.node_id}_round_{self.curren_round}.ckpt")

    def local_training(self):
        scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
        run_config = RunConfig(
            name=f"ptl-mnist-example_node_{self.node_id}",
            storage_path="/tmp/ray_results",
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_accuracy",
                checkpoint_score_order="max",
            ),
        )

        trainer = TorchTrainer(
            self.train_func_per_worker,
            scaling_config=scaling_config,
            run_config=run_config,
        )
        trainer.fit()
    
    def aggregation(self):
        current_rount_nei_models = self.nei_model_record[self.curren_round]
        nei_models_list = []
        print(f"Node {self.node_id} aggregate model with {self.neiList}")
        for nei in current_rount_nei_models:
            if nei in self.neiList:
                nei_models_list.append(current_rount_nei_models[nei].state_dict())
        
        if self.node_id not in self.neiList:
            nei_models_list.append(self.model.state_dict())
        aggregated_model_para = fed_avg(nei_models_list)     
        self.aggregated_model.load_state_dict(aggregated_model_para)
        self.replace_local_aggregated_model()

        trainer = pl.Trainer(devices="auto",
                             accelerator="auto") 
        print(f"Performance of Node {self.node_id} after aggregation at round {self.curren_round}")
        trainer.test(self.model, DataLoader(self.test_dataset, batch_size=64,shuffle=False))

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics import MetricCollection
import lightning.pytorch as pl
import torch.nn.functional as F

from mnistmodel import MNISTModelMLP
from fmnistmodel import FashionMNISTModelMLP
from cifar10model import SimpleMobileNet
from syscallmodel import SYSCALLModelMLP


class GeneralModel(pl.LightningModule):
    def __init__(self, dataset_name, current_control_variates):
        super(GeneralModel, self).__init__()
        self.dataset_name = dataset_name
        self.current_control_variates = current_control_variates
        
        # 动态选择模型
        if self.dataset_name == "MNIST":
            self.model = MNISTModelMLP()
        elif self.dataset_name == "FashionMNIST":
            self.model = FashionMNISTModelMLP()
        elif self.dataset_name == "Cifar10":
            self.model = SimpleMobileNet()
        elif self.dataset_name == "Syscall":
            self.model = SYSCALLModelMLP()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)

        # 假设 current_control_variates 是一个与模型参数维度匹配的字典
        if self.current_control_variates:
            for name, param in self.model.named_parameters():
                if name in self.current_control_variates:
                    control_variate = self.current_control_variates[name]
                    loss += F.mse_loss(param, control_variate)  # 将 control variates 的差异加入到 loss 中

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
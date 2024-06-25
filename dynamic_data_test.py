from util import adjacency_matrix_to_nei_list, dirichlet_sampling_balanced, \
    get_adjacency_matrix, generate_node_configs, generate_attack_matrix, load_dataset
import numpy as np
from subset import ChangeableSubset
from torch.utils.data import DataLoader, random_split
from itertools import product
from torch.utils.data import Subset
import torch
from data_util import DynamicDataLoader, DynamicDataset, dynamic_transformer
from mnistmodel import MNISTModelMLP
import lightning.pytorch as pl
import time, random

def cal_backdoor_acc(backdoor_valid_loader, model, target_label):
    data_loader = backdoor_valid_loader

    all_targets, all_predictions = [], []
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            logits = model(features)
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

    num_predicted_target = confmat.sum(axis=0)[target_label] - confmat.item((target_label, target_label))
    num_samples = len(data_loader.dataset) - confmat.item((target_label, target_label))
    attacker_success = num_predicted_target / num_samples
    print("Computed ASR Backdoor: {}".format(attacker_success))
    return confmat


dataset_name = "MNIST"
train_dataset, test_dataset = load_dataset(dataset_name)
targets = train_dataset.targets
alpha = 100
num_peers = 10
client_indices = dirichlet_sampling_balanced(targets, alpha, num_peers)
client = 0
indices = client_indices[client]


tr_subset = ChangeableSubset(train_dataset, indices, data_poisoning=True, 
                            poisoned_sample_ratio=100, targeted=True,
                            target_label=0)
    
data_train, data_val = random_split(
            tr_subset,
            [
                int(len(tr_subset) * 0.8),
                len(tr_subset) - int(len(tr_subset) * 0.8),
            ],
        )

number_test = int(len(test_dataset)/num_peers)
test_indices = random.sample(range(len(test_dataset)), number_test)   


number_backdoor_valid = int(number_test*0.2)
test_backdoor_valid_indices = random.sample(test_indices, number_backdoor_valid)
test_indices = list(set(test_indices) - set(test_backdoor_valid_indices))

test_backdoor_valid = ChangeableSubset(test_dataset, test_backdoor_valid_indices, data_poisoning=True, 
                            poisoned_sample_ratio=100, targeted=True,
                            target_label=0, backdoor_validation=True)

test_dataset = ChangeableSubset(test_dataset, test_indices) 

# start = time.time()
# data_train_dynamic = DynamicDataset(data_train, applier)
# data_val_dynamic = DynamicDataset(data_val, applier)
# test_dataset_dynamic = DynamicDataset(test_dataset, applier)
# end = time.time()

# print(end-start)

data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True )
data_val_loader = DataLoader(data_val, batch_size=64,shuffle=False)
test_dataset_loader = DataLoader(test_dataset, batch_size=64,shuffle=False)
backdoor_valid_loader = DataLoader(test_backdoor_valid, batch_size=64,shuffle=False)


model = MNISTModelMLP()
for i in range(10):
    trainer = pl.Trainer(max_epochs=3, 
                        devices=1,
                        accelerator="cuda",
                        enable_progress_bar=False, 
                        enable_checkpointing=False
                        )
    
    applier = dynamic_transformer(data_train.dataset[0][0].shape[-1], data_train.dataset[0][0].shape[-1])
    print(f"current applier{applier}")

    # data_train_dynamic = DynamicDataset(data_train, applier)
    # data_val_dynamic = DynamicDataset(data_val, applier)
    # test_dataset_dynamic = DynamicDataset(test_dataset, applier)
    # test_backdoor_valid_dynamic = DynamicDataset(test_backdoor_valid, applier)

    # data_train_loader = DataLoader(data_train_dynamic, batch_size=64, shuffle=True )
    # data_val_loader = DataLoader(data_val_dynamic, batch_size=64,shuffle=False)
    # backdoor_valid_loader = DataLoader(test_backdoor_valid_dynamic, batch_size=64,shuffle=False)
    # test_dataset_loader = DataLoader(test_dataset_dynamic, batch_size=64,shuffle=False)


            

    trainer.fit(model, train_dataloaders=data_train_loader, val_dataloaders=data_val_loader)
    trainer.test(model, dataloaders=test_dataset_loader)
    cal_backdoor_acc(backdoor_valid_loader, model,0)
    
    # data_train_loader_dy = DynamicDataLoader(data_train_loader,applier)
    # data_val_loader_dy = DynamicDataLoader(data_val_loader, applier)
    # test_dataset_loader_dy = DynamicDataLoader(test_dataset_loader, applier)
    # trainer.fit(model, train_dataloaders=data_train_loader_dy, val_dataloaders=data_val_loader_dy)
    # trainer.test(model, test_dataset_loader_dy)
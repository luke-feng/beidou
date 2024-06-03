import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from subset import ChangeableSubset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset

# Define function for Dirichlet sampling and balanced data distribution
def dirichlet_sampling_balanced(targets, alpha, num_clients):
    num_classes = len(np.unique(targets))
    data_per_client = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        for i in range(num_clients):
            data_per_client[i].extend(splits[i])
    
    # Ensure each client has the same number of samples
    min_samples = min(len(data) for data in data_per_client)
    balanced_data_per_client = [data[:min_samples] for data in data_per_client]
    
    return balanced_data_per_client


# L0 norm, number of non zero items
def l0_norm(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return torch.sum(x != 0).item()

# L1 norm, abs value of all items
def l1_norm(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return torch.sum(torch.abs(x)).item()

# L2 norm, the square root of the sum of the squares of the items
def l2_norm(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return torch.sqrt(torch.sum(x**2)).item()

# L∞ norm, the maximum absolute value among the items
def l_inf_norm(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return torch.max(torch.abs(x)).item()


def plot_class_distribution(targets, client_indices, num_clients):
    num_classes = len(np.unique(targets))
    fig, axs = plt.subplots(1, num_clients, figsize=(8, 4))
    axs = axs.flatten()
    
    for i in range(num_clients):
        unique, counts = np.unique(targets, return_counts=True)
        axs[i].bar(unique, counts, tick_label=unique)
        axs[i].set_title(f'Client {i+1}')
        axs[i].set_xlabel('Class')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def generate_logger_config(project, group, dataset, dist_alpha, node_id, epoch, round):
    config = {'project':project,
              'group': group,
              'name':f"node_{node_id}",
              'config':{
                  'dataset': dataset,
                  'dist_alpha': dist_alpha,
                  'round': round,
                  'epoch': epoch
              }}
    return config

def adjacency_matrix_to_nei_list(adjacency_matrix):
    nei_list = {}
    for i, adj in enumerate(adjacency_matrix):
        nei_list[i] = []
        for nei, j in enumerate(adj):
            if j == 1:
                nei_list[i].append(nei)
    return nei_list

def generate_node_configs(node_id, indices, experimentsName, experimentsName_path, dataset_name, neiList, num_peers, maxRound, maxEpoch, train_dataset, test_dataset):
    basic_config = {}
    node_config = {}   

    basic_config = {
        "node_id": node_id,
        "indices": indices,
        "experimentsName": experimentsName,
        "experimentsName_path": experimentsName_path,
        "dataset_name": dataset_name,
        "neiList": neiList,
        "maxRound": maxRound,
        "maxEpoch": maxEpoch,
        "num_peers": num_peers,     
       }
    
    number_test = int(len(test_dataset)/num_peers)
    test_indices = random.sample(range(len(test_dataset)), number_test)
    test_dataset = Subset(test_dataset, test_indices)
      
    tr_subset = ChangeableSubset(
        train_dataset, indices)
    data_train, data_val = random_split(
                tr_subset,
                [
                    int(len(tr_subset) * 0.8),
                    len(tr_subset) - int(len(tr_subset) * 0.8),
                ],
            )

    data_train_loader = DataLoader(data_train, batch_size=64,shuffle=True)
    data_val_loader = DataLoader(data_val, batch_size=64,shuffle=False)
    test_dataset_loader = DataLoader(test_dataset, batch_size=64,shuffle=False)

    node_config = {
        "basic_config":basic_config,
        "data_train_loader": data_train_loader,
        "data_val_loader": data_val_loader,
        "test_dataset_loader": test_dataset_loader

    }
    return node_config



import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from subset import ChangeableSubset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from typing import Union
import networkx as nx
from typing import OrderedDict, List, Optional
import logging

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import FashionMNIST
from syscalldataset import SYSCALL

# Define function for Dirichlet sampling and balanced data distribution
def dirichlet_sampling_balanced(targets, alpha, num_clients):
    targets = np.array(targets)
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

def generate_node_configs(node_id:int, indices:list, experimentsName:str, experimentsName_path:str, 
                          dataset_name:str, neiList:list, num_peers:int, maxRound:int, maxEpoch:int, 
                          train_dataset, test_dataset, attack_type:str, targeted:Union[bool, str], 
                          noise_injected_ratio:int, poisoned_sample_ratio:int, aggregation,
                          dynamic_topo:bool, dynamic_agg:bool,dynamic_data:bool, is_proactive:bool):
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
        "attack_type": attack_type,
        "targeted": targeted,
        "noise_injected_ratio": noise_injected_ratio,
        "poisoned_sample_ratio": poisoned_sample_ratio,
        "aggregation": aggregation,
        "dynamic_topo": dynamic_topo,
        "dynamic_agg": dynamic_agg,
        "dynamic_data": dynamic_data,
        "is_proactive": is_proactive
       }
       
    label_flipping = False
    data_poisoning = False
    attack_targeted = False
    if attack_type.lower() not in ["label flipping", "no attack", "sample poisoning", "model poisoning"]:
        print(f"{attack_type} attack type is not supported")
    if attack_type.lower() == "label flipping":
        label_flipping = True
    elif attack_type.lower() == "sample poisoning":
        data_poisoning = True
    
    if str(targeted).lower() == 'true':
        attack_targeted = True
    
    target_label=3
    target_changed_label=7
    noise_type="salt"
    
      
    tr_subset = ChangeableSubset(train_dataset, indices, label_flipping, data_poisoning, 
                                 poisoned_sample_ratio, noise_injected_ratio, attack_targeted,
                                 target_label, target_changed_label, noise_type)
    
    data_train, data_val = random_split(
                tr_subset,
                [
                    int(len(tr_subset) * 0.8),
                    len(tr_subset) - int(len(tr_subset) * 0.8),
                ],
            )
    data_train_loader = DataLoader(data_train, batch_size=64,shuffle=True)
    data_val_loader = DataLoader(data_val, batch_size=64,shuffle=False)

    number_test = int(len(test_dataset)/num_peers)
    test_indices = random.sample(range(len(test_dataset)), number_test)   

    
    number_backdoor_valid = int(number_test*0.2)
    test_backdoor_valid_indices = random.sample(test_indices, number_backdoor_valid)
    test_indices = list(set(test_indices) - set(test_backdoor_valid_indices))
    test_backdoor_valid = ChangeableSubset(test_dataset, test_backdoor_valid_indices, label_flipping=False, 
                                           data_poisoning=True, poisoned_sample_ratio=100, noise_injected_ratio=100, 
                                           targeted=True, target_label=3, target_changed_label=7, noise_type="salt", 
                                           backdoor_validation=True)
    
    backdoor_valid_loader = DataLoader(test_backdoor_valid, batch_size=64,shuffle=False)
    
    test_dataset = Subset(test_dataset, test_indices)            
    test_dataset_loader = DataLoader(test_dataset, batch_size=64,shuffle=False)

    node_config = {
        "basic_config":basic_config,
        "data_train_loader": data_train_loader,
        "data_val_loader": data_val_loader,
        "test_dataset_loader": test_dataset_loader,
        "backdoor_valid_loader": backdoor_valid_loader
    }
    return node_config

def generate_attack_matrix(node_list:list, attack_type:str, targeted:Union[bool, str], \
    poisoned_node_ratio: Union[int, list], noise_injected_ratio:int, poisoned_sample_ratio:int):
    """_summary_

    Args:
        node_list (list): list of node ids
        attack_type (str): the type of attacks, label flipping, sample poisoning, model poisoning
        targeted (Union[bool, str]): targeted or not
        poisoned_node_ratio (Union[int, list]): how many nodes are poisoned, could be the list of the poisoned node id, or the percent of poisoned node
        noise_injected_ration (int): how much noise attacker want to inject to the attack sample/model
        poisoned_sample_ratio (int): how many samples/labels the attacker want to change
    """
    attack_matrix = {}
    poisoned_node_num = 0
    attacked_node_list = []
    if type(poisoned_node_ratio)==list:
        poisoned_node_num = len(poisoned_node_ratio)
        attacked_node_list = poisoned_node_ratio
    elif type(poisoned_node_ratio)==int:
        poisoned_node_num = int(poisoned_node_ratio/100*len(node_list))
        attacked_node_list = random.sample(node_list, poisoned_node_num)
    
    print(f"attacked_node_list_{attacked_node_list}")
    for node_id in node_list:
        if node_id in attacked_node_list:
            attack_matrix[node_id] = {
                "attack_type": attack_type,
                "targeted": targeted,
                "noise_injected_ratio": noise_injected_ratio,
                "poisoned_sample_ratio": poisoned_sample_ratio
            }
        elif node_id not in attacked_node_list:
            attack_matrix[node_id] = {
                "attack_type": "no attack",
                "targeted": None,
                "noise_injected_ratio": None,
                "poisoned_sample_ratio": None
            }
    return attack_matrix

def get_adjacency_matrix(topology:str, num_peers:int, avg_degree:int=2, prob_changed:float=0.5):
    # creating adjecency matrix
    adj_matrix = []
    if topology == "fully":
        G = nx.complete_graph(num_peers)
    elif topology == "star":
        G = nx.star_graph(num_peers-1)
    elif topology == "ring":
        G = nx.cycle_graph(num_peers)
    elif topology == "bus":
        G = nx.path_graph(num_peers)
    elif topology == "rondom":
        G = nx.watts_strogatz_graph(num_peers, avg_degree, prob_changed, seed=None)

    adj_matrix = nx.adjacency_matrix(G).todense()
    return adj_matrix


def cosine_metric2(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = True) -> Optional[float]:
    if model1 is None or model2 is None:
        logging.info("Cosine similarity cannot be computed due to missing model")
        return None

    cos_similarities = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten().to(torch.float32)
            l2 = model2[layer].flatten().to(torch.float32)
            if l1.shape != l2.shape:
                # Adjust the shape of the smaller layer to match the larger layer
                min_len = min(l1.shape[0], l2.shape[0])
                l1, l2 = l1[:min_len], l2[:min_len]

            cos_sim = torch.nn.functional.cosine_similarity(l1.unsqueeze(0), l2.unsqueeze(0), dim=1)
            cos_similarities.append(cos_sim.item())

    if cos_similarities:
        avg_cos_sim = torch.mean(torch.tensor(cos_similarities))
        # result = torch.clamp(avg_cos_sim, min=0).item()
        # return result
        return avg_cos_sim.item() if similarity else (1 - avg_cos_sim.item())
    else:
        return None
    
def cosine_metric(model1: OrderedDict, model2: OrderedDict, similarity: bool = True) -> Optional[float]:
    if model1 is None or model2 is None:
        logging.info("Cosine similarity cannot be computed due to missing model")
        return None

    cos_similarities: List = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].to('cpu')
            l2 = model2[layer].to('cpu')
            if l1.shape != l2.shape:
                # Adjust the shape of the smaller layer to match the larger layer
                min_len = min(l1.shape[0], l2.shape[0])
                l1, l2 = l1[:min_len], l2[:min_len]
            cos = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
            cos_mean = torch.mean(cos(l1.float(), l2.float())).mean()
            cos_similarities.append(cos_mean)
        else:
            logging.info("Layer {} not found in model 2".format(layer))

    if cos_similarities:    
        cos = torch.Tensor(cos_similarities)
        avg_cos = torch.mean(cos)
        relu_cos = torch.nn.functional.relu(avg_cos)  # relu to avoid negative values
        return relu_cos.item() if similarity else (1 - relu_cos.item())
    else:
        return None
        

def euclidean_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], standardized: bool = False, similarity: bool = True) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    distances = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten().to(torch.float32)
            l2 = model2[layer].flatten().to(torch.float32)
            if standardized:
                l1 = (l1 - l1.mean()) / l1.std()
                l2 = (l2 - l2.mean()) / l2.std()
            
            distance = torch.norm(l1 - l2, p=2)
            if similarity:
                norm_sum = torch.norm(l1, p=2) + torch.norm(l2, p=2)
                similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                distances.append(similarity_score.item())
            else:
                distances.append(distance.item())

    if distances:
        avg_distance = torch.mean(torch.tensor(distances))
        return avg_distance.item()
    else:
        return None
    

def minkowski_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], p: int, similarity: bool = True) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    distances = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten().to(torch.float32)
            l2 = model2[layer].flatten().to(torch.float32)

            distance = torch.norm(l1 - l2, p=p)
            if similarity:
                norm_sum = torch.norm(l1, p=p) + torch.norm(l2, p=p)
                similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                distances.append(similarity_score.item())
            else:
                distances.append(distance.item())

    if distances:
        avg_distance = torch.mean(torch.tensor(distances))
        return avg_distance.item()
    else:
        return None

def chebyshev_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = True) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    distances = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten().to(torch.float32)
            l2 = model2[layer].flatten().to(torch.float32)

            distance = torch.norm(l1 - l2, p=float('inf'))
            if similarity:
                norm_sum = torch.norm(l1, p=float('inf')) + torch.norm(l2, p=float('inf'))
                similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                distances.append(similarity_score.item())
            else:
                distances.append(distance.item())

    if distances:
        avg_distance = torch.mean(torch.tensor(distances))
        return avg_distance.item()
    else:
        return None


def manhattan_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = True) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    distances = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten().to(torch.float32)
            l2 = model2[layer].flatten().to(torch.float32)

            distance = torch.norm(l1 - l2, p=1)
            if similarity:
                norm_sum = torch.norm(l1, p=1) + torch.norm(l2, p=1)
                similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                distances.append(similarity_score.item())
            else:
                distances.append(distance.item())

    if distances:
        avg_distance = torch.mean(torch.tensor(distances))
        return avg_distance.item()
    else:
        return None


def pearson_correlation_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = True) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    correlations = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten()
            l2 = model2[layer].flatten()

            if l1.shape != l2.shape:
                min_len = min(l1.shape[0], l2.shape[0])
                l1, l2 = l1[:min_len], l2[:min_len]

            correlation = torch.corrcoef(torch.stack((l1, l2)))[0, 1]
            if similarity:
                adjusted_similarity = (correlation + 1) / 2
                correlations.append(adjusted_similarity.item())
            else:
                correlations.append(1 - (correlation + 1) / 2)

    if correlations:
        avg_correlation = torch.mean(torch.tensor(correlations))
        return avg_correlation.item()
    else:
        return None


def load_dataset(dataset_name:str='MNIST'):
    if dataset_name == "MNIST":
        train_dataset = MNIST(
            f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
        )
        test_dataset = MNIST(
            f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
        )
    if dataset_name == "FashionMNIST":
        fashionmnist_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        train_dataset = FashionMNIST(
            f"{sys.path[0]}/data", train=True, download=True, transform=fashionmnist_transforms
        )
        test_dataset = FashionMNIST(
            f"{sys.path[0]}/data", train=False, download=True, transform=fashionmnist_transforms
        )
    if dataset_name == "Cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        cifar10_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = CIFAR10(
            f"{sys.path[0]}/data", train=True, download=True, transform=cifar10_transforms
        )
        test_dataset = CIFAR10(
            f"{sys.path[0]}/data", train=False, download=True, transform=cifar10_transforms
        )
    if dataset_name == "Syscall":
        syscall_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        train_dataset = SYSCALL(
            f"{sys.path[0]}/data", train=True, download=True, transform=syscall_transforms
        )
        test_dataset = SYSCALL(
            f"{sys.path[0]}/data", train=False, download=True, transform=syscall_transforms
        )
    return train_dataset, test_dataset

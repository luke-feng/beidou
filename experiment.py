import os, sys
from datetime import datetime
import time
from util import adjacency_matrix_to_nei_list, dirichlet_sampling_balanced, \
    get_adjacency_matrix, generate_node_configs, generate_attack_matrix
from torchvision.datasets import MNIST
from torchvision import transforms
from local_node import local_node
import pickle

from fed_avg import fed_avg
from krum import krum
from trimmedmean import trimmedMean
from median import median
import copy 

import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
# configure logging on module level, redirect to file
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))
log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)


start_time = time.time()
num_peers = 3
alpha = 100
node_list = {}
maxRound = 1
maxEpoch = 1
dataset_name = "MNIST"

# topology should be one of ["fully", "star", "ring", "bus", "rondom"]
topology = "ring"
# init_aggregation should be one of [fed_avg, krum, trimmedmean, median]
init_aggregation = fed_avg

adj_matrix = get_adjacency_matrix(topology, num_peers)
nei_list = adjacency_matrix_to_nei_list(adj_matrix)
train_dataset = None
test_dataset = None

# define the attack
# attack should be one of ['sample poisoning', 'model poisoning', 'label flipping', 'no attack']
attack_type = 'model poisoning'
targeted = False
poisoned_node_ratio = [0]
noise_injected_ratio = 70
poisoned_sample_ratio = 100
attack_matrix = generate_attack_matrix(list(range(num_peers)), attack_type, targeted, poisoned_node_ratio, noise_injected_ratio, poisoned_sample_ratio)

# dataset
if dataset_name == "MNIST":
    train_dataset = MNIST(
        f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = MNIST(
        f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
    )

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
experimentsName = f'{num_peers}_clients_alpha_{alpha}_{dataset_name}_{topology}_'+dt_string
targets = train_dataset.targets
client_indices = dirichlet_sampling_balanced(targets, alpha, num_peers)
cwd = os.getcwd()

experimentsName_path = cwd+'/experiments/'+experimentsName
os.mkdir(experimentsName_path)

# mtd
dynamic_topo = True
dynamic_agg = False 
is_proactive  = True 

# initial the nodes
for client in range(num_peers):
    indices = client_indices[client]
    node_id = client      
    neiList = nei_list[client]
    
    attack_info = attack_matrix[node_id]
    attack_type = attack_info['attack_type']
    targeted = attack_info['targeted']
    noise_injected_ratio = attack_info['noise_injected_ratio']
    poisoned_sample_ratio = attack_info['poisoned_sample_ratio']

    node_config = generate_node_configs(node_id, indices, experimentsName, experimentsName_path, 
                                        dataset_name, neiList, num_peers, maxRound, maxEpoch, 
                                        train_dataset, test_dataset, attack_type, targeted, 
                                        noise_injected_ratio, poisoned_sample_ratio, init_aggregation,
                                        dynamic_topo, dynamic_agg, is_proactive)
    
    basic_config = node_config['basic_config']
    data_train_loader = node_config['data_train_loader']
    data_val_loader = node_config['data_val_loader']
    test_dataset_loader = node_config['test_dataset_loader']
    
    node = local_node(node_id,basic_config, data_train_loader, data_val_loader, test_dataset_loader)
    node_list[node_id] = node
    
    with open(experimentsName_path+f"/{node_id}_config.pk", "wb") as f:
        pickle.dump(node_config, f)
        f.close()
    
# initial aggregation, without training
for node_id in node_list:
    node = node_list[node_id]
    node.curren_round = 0
    for nei in node_list:
        node_list[nei].add_nei_model(0, node_id, copy.deepcopy(node.model.state_dict()))
for node_id in node_list:
    node = node_list[node_id]
    node.aggregation()

# federated learning
for round in range(1, maxRound+1):  
    for node_id in node_list:
        node = node_list[node_id]
        node.curren_round = round
        node.local_training()
        for nei in node_list:
            # model will be send to all nodes, but only aggregate within neiList
            # print(f"I am node {node_id}, I am sending my model to node {nei}")
            node_list[nei].add_nei_model(round, node_id, copy.deepcopy(node.model.state_dict()))
    
    for node_id in node_list:
        node = node_list[node_id]
        node.aggregation()
        for nei_id in node_list:
            if nei_id in  node.get_neiList():
                #build the dual link
                node_list[nei_id].add_nei_to_neiList(node_id)
            else:
                node_list[nei_id].remove_nei_from_neiList(node_id)

end_time = time.time()
print(f"finished in {end_time-start_time} seconds")

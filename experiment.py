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

start_time = time.time()
num_peers = 5
alpha = 100
node_list = {}
maxRound = 2
maxEpoch = 1
dataset_name = "MNIST"

# topology should be one of ["fully", "star", "ring", "bus", "rondom"]
topology = "fully"
# init_aggregation should be one of [fed_avg, krum, trimmedmean, median]
init_aggregation = median

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
dynamic_topo = False
dynamic_agg = False 
is_proactive  = True 

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
    

for round in range(maxRound):
    for node_id in node_list:
        node = node_list[node_id]
        node.curren_round = round+1
        node.local_training()
        for nei in node_list:
            node_list[nei].add_nei_model(round+1, node_id, node.model)
    
    for node_id in node_list:
        node = node_list[node_id]
        node.aggregation()

end_time = time.time()
print(f"finished in {end_time-start_time} seconds")

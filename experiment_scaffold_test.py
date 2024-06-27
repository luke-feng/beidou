import os, sys
from datetime import datetime
import time
from util import adjacency_matrix_to_nei_list, dirichlet_sampling_balanced, \
    get_adjacency_matrix, generate_node_configs, generate_attack_matrix, load_dataset, dirichlet_sampling_balanced_mixed

from evaluation_util import read_experiment_csvs
from local_node import local_node
from local_node_scaffold import local_node_scaffold
import pickle

from fed_avg import fed_avg
from krum import krum
from trimmedmean import trimmedMean
from median import median
from fed_scaffold import scaffold
import copy 

import time
from datetime import timedelta

import logging
# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
# configure logging on module level, redirect to file
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.FileHandler("core.log"))
log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)



# num_peers = 10
# alpha_list = [100, 1, 0.1]
# topology_list = ["fully", "star", "ring"]
# maxRound = 30
# maxEpoch = 1
# init_aggregation_dict = {'fed_avg':fed_avg, 'krum':krum, 'trimmedMean':trimmedMean, 'median':median}
# attack_type_list = ['sample poisoning', 'model poisoning', 'label flipping', 'no_attack']
# poisoned_node_ratio_list = [10, 30, 50, 70, 90]
# noise_injected_ratio = 70
# poisoned_sample_ratio = 100
# # dataset_name_list = ["MNIST", "FashionMNIST", "Cifar10", "Syscall"]
# dataset_name = "MNIST"


# 6月22日 fedavg 
num_peers = 10
# alpha_list = [[50,5],[50,1],[50,0.5]]
alpha_list = [[50,5]]
topology_list = ["fully"]
maxRound = 30
maxEpoch = 2
init_aggregation_dict = {'scaffold':scaffold}
attack_type_list = ['no_attack']
poisoned_node_ratio_list = [0]
noise_injected_ratio = 70
poisoned_sample_ratio = 100
# dataset_name_list = ["MNIST", "FashionMNIST", "Cifar10"]
# dataset_name = "MNIST"
dataset_name_list = ["MNIST"]

for dataset_name in dataset_name_list:
    for alpha in alpha_list:
        for topology in topology_list:
            for init_aggregation_name in init_aggregation_dict:
                init_aggregation = init_aggregation_dict[init_aggregation_name]
                for attack_type in attack_type_list:
                    for poisoned_node_ratio in poisoned_node_ratio_list:
                        start_time = time.time()
                        node_list = {}                   

                        # topology should be one of ["fully", "star", "ring", "bus", "rondom"]
                        #topology = "ring"
                        # init_aggregation should be one of [fed_avg, krum, trimmedMean, median]
                        # init_aggregation = fed_avg

                        adj_matrix = get_adjacency_matrix(topology, num_peers)
                        nei_list = adjacency_matrix_to_nei_list(adj_matrix)
                        train_dataset = None
                        test_dataset = None

                        # define the attack
                        # attack should be one of ['sample poisoning', 'model poisoning', 'label flipping', 'no_attack']
                        #attack_type = 'model poisoning'
                        targeted = False
                        #poisoned_node_ratio = [0]
                        

                        attack_matrix = generate_attack_matrix(list(range(num_peers)), attack_type, targeted, poisoned_node_ratio, noise_injected_ratio, poisoned_sample_ratio)

                        # dataset
                        train_dataset, test_dataset = load_dataset(dataset_name)
                        
                        # mtd
                        dynamic_topo = False
                        dynamic_agg = False 
                        dynamic_data = False 
                        is_proactive  = False 

                        # datetime object containing current date and time
                        now = datetime.now()
                        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                        experimentsName = f'Mixed2_{init_aggregation_name}_alpha_{alpha[0]}_{alpha[1]}_clients_{num_peers}_{dataset_name}_{topology}_{attack_type}_{poisoned_node_ratio}_dynamic_topo_{dynamic_topo}_dynamic_agg_{dynamic_agg}_dynamic_data_{dynamic_data}_is_proactive_{is_proactive}'+dt_string
                        targets = train_dataset.targets
                        # client_indices = dirichlet_sampling_balanced(targets, alpha, num_peers)
                        client_indices, client_distribution = dirichlet_sampling_balanced_mixed(targets, alpha, num_peers)
                        # log distribution:
                        log_file_path = ".clients_distribution.log"
                        with open(log_file_path, 'w') as f:
                            f.write(f"{experimentsName}:\n")
                            for client_id, distribution in client_distribution.items():
                                f.write(f"Client {client_id}:\n")
                                for class_id, count in distribution.items():
                                    f.write(f"\tClass {class_id}: {count}\n")

                        cwd = os.getcwd()

                        experimentsName_path = cwd+'/experiments/'+experimentsName
                        if os.path.exists(experimentsName_path) and os.path.isdir(experimentsName_path):
                            print(experimentsName_path)
                        else:
                            os.mkdir(experimentsName_path)  

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
                                            dynamic_topo, dynamic_agg, dynamic_data, is_proactive)
                            
                            basic_config = node_config['basic_config']
                            data_train_loader = node_config['data_train_loader']
                            data_val_loader = node_config['data_val_loader']
                            test_dataset_loader = node_config['test_dataset_loader']
                            backdoor_valid_loader = node_config['backdoor_valid_loader']
        
                            node = local_node_scaffold(node_id,basic_config, data_train_loader, data_val_loader, test_dataset_loader, backdoor_valid_loader)
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
                                    node_list[nei].add_nei_control_variates(round, node_id, copy.deepcopy(node.model.client_control_variate))
                            
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
                        read_experiment_csvs(experimentsName=experimentsName)                    
                        time.sleep(10)



import os, sys
from datetime import datetime
import time
from util import adjacency_matrix_to_nei_list, dirichlet_sampling_balanced, \
    get_adjacency_matrix, generate_node_configs, generate_attack_matrix, load_dataset,dirichlet_sampling_balanced_mixed

from evaluation_util import read_experiment_csvs
from local_node import local_node
import pickle

from fed_avg import fed_avg
from krum import krum
from trimmedmean import trimmedMean
from median import median
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


num_peers = 10
alpha_list = [[50,1]]
topology_list = ["fully"]
maxRound = 30
maxEpoch = 3
init_aggregation_dict = {'fed_avg':fed_avg}
attack_type_list = ['no_attack']
poisoned_node_ratio_list = [0]
noise_injected_ratio = 70
poisoned_sample_ratio = 100
dataset_name_list = ["MNIST", "FashionMNIST", "Cifar10"]
dataset_name = "MNIST"



for alpha in alpha_list:
    print(f"alpha: {alpha}")
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

   
                    targeted = False
                    

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
                    experimentsName = f'{num_peers}_clients_alpha_{alpha}_{dataset_name}_{topology}_{init_aggregation_name}_{attack_type}_{poisoned_node_ratio}_dynamic_topo_{dynamic_topo}_dynamic_agg_{dynamic_agg}_dynamic_data_{dynamic_data}_is_proactive_{is_proactive}'+dt_string
                    targets = train_dataset.targets
                    client_indices = dirichlet_sampling_balanced_mixed(targets, alpha, num_peers)
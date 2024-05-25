import os, sys
from datetime import datetime
import networkx as nx
from util import adjacency_matrix_to_nei_list, dirichlet_sampling_balanced, generate_logger_config
import ray
from torchvision.datasets import MNIST
from torch.utils.data import Subset
from torchvision import transforms
import numpy as np
from node import local_node
from multiprocessing import Manager, Process


G = nx.complete_graph(10)
adj_matrix = nx.adjacency_matrix(G).todense()
nei_list = adjacency_matrix_to_nei_list(adj_matrix)

mnist_train = MNIST(
    f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
)
mnist_val = MNIST(
    f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
)

mnist_val = Subset(mnist_val, range(2000))
targets = np.array(mnist_train.targets)


alpha = 100
num_clients = 10
node_list = {}
maxRound = 5
maxEpoch = 3
train_dataset = mnist_train
test_dataset = mnist_val
experimentsName = f'{num_clients}_clients_alpha_{alpha}_MNIST_fully'
client_indices = dirichlet_sampling_balanced(targets, alpha, num_clients)
cwd = os.getcwd()
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

experimentsName_path = cwd+'\\'+experimentsName+dt_string
os.mkdir(experimentsName_path)

ray.init()
try:
    for client in range(num_clients):
        indices = client_indices[client]
        node_id = client  
        
        neiList = nei_list[client]
        logger_config = generate_logger_config('local_test', experimentsName, 'MNIST', alpha, node_id, maxEpoch, maxRound)
        logger = None

        # logger = wandb.init(
        #     project=logger_config['project'],
        #     group=logger_config['group'],
        #     name=logger_config['name'],
        #     config=logger_config['config'],

        # )
        node = local_node.remote(node_id, experimentsName, maxRound, maxEpoch, train_dataset, test_dataset, indices, neiList, experimentsName_path,logger)
        # node = local_node(node_id, experimentsName, maxRound, maxEpoch, train_dataset, test_dataset, indices, neiList, experimentsName_path,logger)
        node_list[client] = node
    
    for round in range(maxRound):
        ray.get([node_list[node_id].next_round.remote() for node_id in node_list])
        ray.get([node_list[node_id].local_training.remote() for node_id in node_list])

        for node_id in node_list:
            neiList = ray.get(node_list[node_id].get_neiList.remote())
            for nei in neiList:
                model = ray.get(node_list[nei].get_model.remote())
                ray.get(node_list[nei].add_nei_model.remote(round+1, node_id, model))
        
        ray.get([node_list[node_id].aggregation.remote() for node_id in node_list])

except Exception:
    print(Exception)
finally:
    ray.shutdown()

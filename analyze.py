import pandas as pd
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt

def read_as_pd(experiment_name):
    # read 
    with open(f'./experiments/{experiment_name}/res.json', 'r') as file:
        result_json = json.load(file)
    # to dataframe
    dfs = {}
    for node, metrics in result_json.items():
        print(node)
        data={}
        for metric, values in metrics.items():
            print(metric)
            if metric != "attack_type":
                print(len(values))
                data[f"{metric}"] = values
        # df = pd.DataFrame(data)
        # dfs[f"{node}"]=df
    return dfs


def read_json_as_pd(experiment_name):
    # Read
    with open(f'./experiments/{experiment_name}/res.json', 'r') as file:
        result_json = json.load(file)
    
    # Initialize a 2D array
    nodes = list(result_json.keys())
    metrics = set()
    for metrics_dict in result_json.values():
        metrics.update(metrics_dict.keys())
    metrics = [m for m in metrics if m != "attack_type"]
    
    data = np.empty((len(nodes), len(metrics)), dtype=object)
    
    # Populate the 2D array
    for i, node in enumerate(nodes):
        for j, metric in enumerate(metrics):
            if metric in result_json[node]:
                data[i, j] = result_json[node][metric]
            else:
                data[i, j] = None
    
    # To DataFrame
    df = pd.DataFrame(data, index=nodes, columns=metrics)
    
    return df

def rename(experiment_name):
    if "fed_avg" in experiment_name:
        algo = "fedavg"
    if "fedep" in experiment_name:
        algo = "fedep"

    if "_MNIST" in experiment_name:
        data = "MNIST"
    elif "FashionMNIST" in experiment_name:
        data = "FashionMNIST"
    else:
        data = "Cifar10"

    pattern = r'(\d+)_clients_alpha_([\d.]+)_.*'

    match = re.search(pattern, experiment_name)
    if match:
        clients_num = match.group(1)
        alpha_value = match.group(2)
    return  f'{algo}_{data}_alpha-{alpha_value}_clients-{clients_num}'

def read_folder(folder="./experiments"):
    dfs={}
    for experiment_name in os.listdir(folder):
        new_name = rename(experiment_name)

        df = read_json_as_pd(experiment_name)
        dfs[new_name] = df
    return dfs

def plot_experiment(name, df):

    if not os.path.exists('assets'):
        os.makedirs('assets')

    rows, cols = 6, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 20))

    fig.suptitle(name, fontsize=16)

    # Extract the index (nodes)
    nodes = df.index

    for j in range(df.shape[1]):
    # locate current subgraph
        row, col = divmod(j, cols)
        ax = axes[row, col]

        data = df.iloc[:, j].values.tolist()  # Convert to list for easier plotting

        # Plot each line with its own legend
        for i, node in enumerate(nodes):
            ax.plot(data[i], label=node)

            # Add labels and title
            ax.set_xlabel('epochs')
            ax.set_ylabel('value')
            ax.set_title(df.columns[j])
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    plt.savefig(f'assets/{name}.png')
    plt.show()
        



dfs = read_folder("./experiments")


if __name__ == "__main__":
    dfs_test = read_as_pd("TrialRun_10_clients_alpha_5_MNIST_fully_fedep_no_attack_0_dynamic_topo_False_dynamic_agg_False_dynamic_data_False_is_proactive_False22_06_2024_14_25_57")

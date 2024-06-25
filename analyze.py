import pandas as pd
import json
import os

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

if __name__ == "__main__":
    dfs_test = read_as_pd("TrialRun_10_clients_alpha_5_MNIST_fully_fedep_no_attack_0_dynamic_topo_False_dynamic_agg_False_dynamic_data_False_is_proactive_False22_06_2024_14_25_57")

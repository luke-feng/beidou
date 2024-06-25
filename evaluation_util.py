import os, sys
import pandas as pd
import numpy as np
import json
import re


def extract_node_id(string):
    match = re.search(r'node_\d+', string)
    if match:
        return match.group(0)
    else:
        return None
    
def get_test_res(dataSeries):
    datalist = list(dataSeries.dropna())
    res_before_agg = [datalist[0]]
    res_after_agg = []
    for i in range(1, len(datalist)):
        if i%2 != 0:
            res_before_agg.append(datalist[i])
        else:
            res_after_agg.append(datalist[i])
    return res_before_agg, res_after_agg


def read_experiment_csvs(experimentsName:str):
    # experimentsName = '10_clients_alpha_100_Syscall_fully_fed_avg_model poisoning_0_dynamic_topo_False_dynamic_agg_False_is_proactive_False10_06_2024_13_01_11'
    cwd = os.getcwd()
    experimentsName_path = cwd+'/experiments/'+experimentsName
    results_csv_list = []
    for root, dirs, files in os.walk(experimentsName_path):
        for file in files:
                if 'metrics.csv' in file:
                    results_csv_list.append(os.path.join(root, file))

    train_metric_name = ['TrainEpoch/Accuracy', 'TrainEpoch/F1Score', 'TrainEpoch/Precision', 'TrainEpoch/Recall']
    val_metric_name = ['ValidationEpoch/Accuracy', 'ValidationEpoch/F1Score', 'ValidationEpoch/Precision', 'ValidationEpoch/Recall']
    test_metric_name = ['Test/Accuracy', 'Test/F1Score', 'Test/Loss', 'Test/Precision',  'Test/Recall'  ]


    res_dict = {}
    for csv_name in results_csv_list:
        # df = pd.read_csv(results_csv_list[0])
        df = pd.read_csv(csv_name)
        node_id = extract_node_id(csv_name)
        node_res = {}

        if 'attack_type' in  df.columns:
            attack_type = list(df['attack_type'].dropna())
            node_res['attack_type'] = attack_type[0]
            
        for train_metric in train_metric_name:
            if train_metric in df.columns:
                metric = list(df[train_metric].dropna())
                node_res[train_metric] = metric

        for val_metric in val_metric_name:
            if val_metric in df.columns:
                metric = list(df[val_metric].dropna())
                node_res[val_metric] = metric

        for test_metric in test_metric_name:
            if test_metric in df.columns:
                res_before_agg, res_after_agg = get_test_res(df[train_metric])
                node_res[f"{test_metric}_before_aggregation"] = res_before_agg
                node_res[f"{test_metric}_after_aggregation"] = res_after_agg
        
        res_dict[node_id] = node_res

    with open(experimentsName_path+'/res.json', 'w') as f:
        json.dump(res_dict, f)


if __name__ == "__main__":
    for experiment_name in os.listdir("./experiments"):
        read_experiment_csvs(experiment_name)

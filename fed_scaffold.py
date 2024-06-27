import torch
import numpy as np

def scaffold( models, control_variates):
    """
    Weighted average of the models.

    Args:
        models: list of model state_dict
    """
    if len(models) == 0:
        return None

    # Total Samples
    total_samples = len(models)

    # Create a Zero Model
    accum = {layer: torch.zeros_like(models[0][layer]) for layer in models[0]}
    # for layer in models[0]:
    #     print(models[0][layer].dtype, accum[layer].dtype)

    # Add weighted models
    for model in models:
        for layer in accum:
            accum[layer] += model[layer]

    # Normalize Accum
    for layer in accum:
        avg = accum[layer]/total_samples
        accum[layer] = avg.to(accum[layer].dtype)
        
    # Compute avg_control_variates
    avg_control_variates = {}

    # Accumulate control variates across all nodes
    if control_variates == {}:
        avg_control_variates = {}
    else:
        for node_id, node_control_variates in control_variates.items():
            for param, values in node_control_variates.items():
                if param not in avg_control_variates.keys():
                    avg_control_variates[param] = values
                else:
                    avg_control_variates[param] += values
            
    # Normalize avg_control_variates by dividing by the number of nodes
    num_nodes = len(control_variates)
    for param in avg_control_variates.keys():
        avg_control_variates[param] /= num_nodes
    
    print(f"avg_control_variates keys: {avg_control_variates.keys()}")

    return accum, avg_control_variates
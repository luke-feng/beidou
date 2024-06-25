from fed_avg import fed_avg
import torch


def fedep(alpha_k, models):
    """
    Ponderated average of the models.

    Args:
        models: Dictionary with the models (node: model,num_samples).
        model : {layer: tensor, ...}
    """
    # Check if there are models to aggregate
    if len(models) == 0:
        print("[FedEP] Trying to aggregate models when there is no models")
        return None
    
    print(f"[FedEP.aggregate] Aggregating models: num={len(models)}")

    # Create a shape of the weights use by all nodes
    accum = {layer: torch.zeros_like(param).float() for layer, param in list(models.values())[-1].items()}


    # Add weighted models
    
    for node_id, model in models.items():
        print(f"accumulating address: {node_id}")
        for layer in accum:
            accum[layer] += model[layer] * alpha_k[node_id]
        
    return accum
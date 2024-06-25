import torch
import logging
import numpy
from util import l2_norm


def krum(models):
    """
    Krum selects one of the m local models that is similar to other models
        as the global model, the euclidean distance between two local models is used.

    Args:
        models: list of model state_dict
    """
    # Check if there are models to aggregate
    if len(models) == 0:
        print(
            "[Krum] Trying to aggregate models when there is no models"
        )
        return None

    # Total models
    total_models = len(models)

    # Create a Zero Model
    accum = {layer: torch.zeros_like(param) for layer, param in models[0].items()}
    
    # Add weighteds models
    print("[Krum.aggregate] Aggregating models: num={}".format(total_models))
        
    # Create model distance list
    distance_list = [0 for i in range(0, total_models)]
    
    # Calculate the L2 Norm between xi and xj
    min_index = 0
    min_distance_sum = float('inf')
    
    
    for i in range(0, total_models):
        m1 = models[i]
        for j in range(0, total_models):
            m2 = models[j]
            distance = 0
            if i == j:
                distance = 0
            else:
                for layer in m1:
                    l1 = m1[layer]
                    # l1 = l1.view(len(l1), 1)

                    l2 = m2[layer]
                    # l2 = l2.view(len(l2), 1)
                    # distance += numpy.linalg.norm(l1 - l2)
                    distance += l2_norm(l1 - l2)
            distance_list[i] += distance

        if min_distance_sum > distance_list[i]:
            min_distance_sum = distance_list[i]
            min_index = i

    # Assign the model with min distance with others as the aggregated model
    m = models[min_index]
    print(f"[Krum.aggregate] {min_index} has beed selected as aggregated model")
    # for layer in m:
    #     accum[layer] = accum[layer] + m[layer]
    #     return accum
    return m
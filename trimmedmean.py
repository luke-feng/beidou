import logging

import torch
import numpy as np


import logging

import torch
import numpy as np


def get_trimmedmean(weights, beta=1):
    """
    For weight list [w1j,w2j,··· ,wmj], removes the largest and
    smallest β of them, and computes the mean of the remaining
    m-2β parameters

    Args:
        weights: weights list, 2D tensor
    """

    # check if the weight tensor has enough space
    weight_len = len(weights)
    if weight_len == 0:
        print(
            "[TrimmedMean] Trying to aggregate models when there is no models"
        )
        return None

    if weight_len <= 2 * beta:
        # logging.error(
        #     "[TrimmedMean] Number of model should larger than 2 * beta"
        # )
        remaining_wrights = weights
        res = torch.mean(remaining_wrights, 0)

    else:
        # remove the largest and smallest β items
        arr_weights = np.asarray(weights)
        nobs = arr_weights.shape[0]
        start = beta
        end = nobs - beta
        atmp = np.partition(arr_weights, (start, end - 1), 0)
        sl = [slice(None)] * atmp.ndim
        sl[0] = slice(start, end)
        # print(atmp[tuple(sl)])
        arr_median = np.mean(atmp[tuple(sl)], axis=0)
        res = torch.tensor(arr_median)

    # get the mean of the remaining weights

    return res


def trimmedMean(models):
    """
    Weighted average of the models.

    Args:
        models: list of model state_dict
    """
    # Check if there are models to aggregate
    if len(models) == 0:
        print(
            "[Median] Trying to aggregate models when there is no models"
        )
        return None

    # Total models
    total_models = len(models)

    # Create a Zero Model
    accum = {layer: torch.zeros_like(param) for layer, param in models[0].items()}
    
    # Add models
    print("[Median.aggregate] Aggregating models: num={}".format(total_models))
    
    
    # Calculate the trimmedmean for each parameter
    for layer in accum:
        weight_layer = accum[layer]
        # get the shape of layer tensor
        l_shape = list(weight_layer.shape)

        # get the number of elements of layer tensor
        number_layer_weights = torch.numel(weight_layer)
        # if its 0-d tensor
        if l_shape == []:
            weights = torch.tensor([models[j][layer] for j in range(0, total_models)])
            weights = weights.double()
            w = get_trimmedmean(weights,1)
            accum[layer] = w

        else:
            # flatten the tensor
            weight_layer_flatten = weight_layer.view(number_layer_weights)

            # flatten the tensor of each model
            models_layer_weight_flatten = torch.stack([models[j][layer].view(number_layer_weights) for j in range(0, total_models)], 0)

            # get the weight list [w1j,w2j,··· ,wmj], where wij is the jth parameter of the ith local model
            trimmedmean = get_trimmedmean(models_layer_weight_flatten, 1)
            accum[layer] = trimmedmean.view(l_shape)
    return accum
    
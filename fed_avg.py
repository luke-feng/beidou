import torch

def fed_avg(models):
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
        
    # self.print_model_size(accum)

    return accum
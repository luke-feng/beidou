import torch

def fed_avg(models):
    """
    Weighted average of the models.

    Args:
        models: Dictionary with the models (node: model, num_samples).
    """
    if len(models) == 0:
        return None

    # Total Samples
    total_samples = len(models)

    # Create a Zero Model
    accum = {layer: torch.zeros_like(param) for layer, param in models[0].items()}

    # Add weighted models
    for model in models:
        for layer in accum:
            accum[layer] += model[layer]

    # Normalize Accum
    for layer in accum:
        accum[layer] /= total_samples
        
    # self.print_model_size(accum)

    return accum
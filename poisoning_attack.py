import copy
import numpy as np
import random
import torch
from collections import OrderedDict
from skimage.util import random_noise

def labelFlipping(dataset, indices, poisoned_percent=0, targeted=False, target_label=4, target_changed_label=7):
    """
    select flipping_persent of labels, and change them to random values.
    Args:
        dataset: the dataset of training data, torch.util.data.dataset like.
        indices: Indices of subsets, list like.
        flipping_persent: The ratio of labels want to change, float like.
    """
    new_dataset = copy.deepcopy(dataset)
    targets = new_dataset.targets.detach().clone()
    num_indices = len(indices)
    # classes = new_dataset.classes
    # class_to_idx = new_dataset.class_to_idx
    # class_list = [class_to_idx[i] for i in classes]
    class_list = set(targets.tolist())
    if targeted == False:
        num_flipped = int(poisoned_percent * num_indices)
        if num_indices == 0:
            return new_dataset
        if num_flipped > num_indices:
            return new_dataset
        flipped_indice = random.sample(indices, num_flipped)

        for i in flipped_indice:
            t = targets[i]
            flipped = torch.tensor(random.sample(class_list, 1)[0])
            while t == flipped:
                flipped = torch.tensor(random.sample(class_list, 1)[0])
            targets[i] = flipped
    else:
        for i in indices:
            if int(targets[i]) == int(target_label):
                targets[i] = torch.tensor(target_changed_label)
    new_dataset.targets = targets
    return new_dataset


def modelpoison(model: OrderedDict, poisoned_ratio, noise_type="gaussian"):
    """
    Function to add random noise of various types to the model parameter.
    """
    poisoned_model = OrderedDict()

    for layer in model:
        bt = model[layer]
        t = bt.detach().clone()
        single_point = False
        if len(t.shape) == 0:
            t = t.view(-1)
            single_point = True
        # print(t)
        if noise_type == "salt":
            # Replaces random pixels with 1.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        elif noise_type == "gaussian":
            # Gaussian-distributed additive noise.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
        elif noise_type == "s&p":
            # Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        else:
            print("ERROR: @modelpoisoning: poison attack type not supported.")
            poisoned = t
        if single_point:
            poisoned = poisoned[0]
        poisoned_model[layer] = poisoned

    return poisoned_model


def datapoison(dataset, indices, poisoned_percent, poisoned_ratio, targeted=False, target_label=3, noise_type="salt", backdoor_validation=False):
    """
    Function to add random noise of various types to the dataset.
    """
    new_dataset = copy.deepcopy(dataset)
    train_data = new_dataset.data
    targets = new_dataset.targets
    num_indices = len(indices)

    if not targeted:
        num_poisoned = int(poisoned_percent * num_indices)
        if num_indices == 0:
            return new_dataset
        if num_poisoned > num_indices:
            return new_dataset
        poisoned_indice = random.sample(indices, num_poisoned)

        for i in poisoned_indice:
            t = train_data[i]
            if noise_type == "salt":
                # Replaces random pixels with 1.
                noise_img = random_noise(t, mode=noise_type, amount=poisoned_ratio)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                poisoned = torch.tensor(noise_img)               

            elif noise_type == "gaussian":
                # Gaussian-distributed additive noise.
                # poisoned = torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
                noise_img = random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                poisoned = torch.tensor(noise_img)
            elif noise_type == "s&p":
                # Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
                # poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
                noise_img = random_noise(t, mode=noise_type, amount=poisoned_ratio)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                poisoned = torch.tensor(noise_img)
            # elif noise_type == "nlp_rawdata":
            #     # for NLP data, change the word vector to 0 with p=poisoned_ratio
            #     poisoned = poison_to_nlp_rawdata(t, poisoned_ratio)
            else:
                print("ERROR: @datapoisoning: poison attack type not supported.")
                poisoned = t
            train_data[i] = poisoned
    else:
        if backdoor_validation:
            # mark all instances for testing
            print("Datapoisoning: generating watermarked samples for testing (all classes)")
            for i in indices:
                t = train_data[i]
                poisoned = add_x_to_image(t)
                train_data[i] = poisoned
        else:
            # only mark samples from specific target for training
            print("Datapoisoning: generating watermarked samples for training, target: " + str(target_label))
            for i in indices:
                if int(targets[i]) == int(target_label):
                    t = train_data[i]
                    poisoned = add_x_to_image(t)
                    train_data[i] = poisoned
    new_dataset.data = train_data
    return new_dataset


def add_x_to_image(img):
    """
    Add a 10*10 pixels X at the top-left of a image
    """
    size = 10
    for i in range(0, size):
        for j in range(0, size):
            img[i][j] = 255
        # img[i][size - i - 1] = 255
    return torch.tensor(img).clone().detach()


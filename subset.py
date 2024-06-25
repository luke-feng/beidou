import copy
from poisoning_attack import labelFlipping, datapoison
from torch.utils.data import Subset

class ChangeableSubset(Subset):
    """
    Could change the elements in Subset Class
    """

    def __init__(self,
                 dataset,
                 indices,
                 label_flipping:bool=False,
                 data_poisoning:bool=False,
                 poisoned_sample_ratio:int=0,
                 noise_injected_ratio:int=0,
                 targeted:bool=False,
                 target_label:int=0,
                 target_changed_label:int=0,
                 noise_type:str="salt",
                 backdoor_validation:bool=False):
        super().__init__(dataset, indices)
        new_dataset = copy.copy(dataset)
        self.dataset = new_dataset
        self.indices = indices
        self.label_flipping = label_flipping
        self.data_poisoning = data_poisoning
        self.poisoned_sample_ratio = poisoned_sample_ratio
        self.noise_injected_ratio = noise_injected_ratio
        self.targeted = targeted
        self.target_label = target_label
        self.target_changed_label = target_changed_label
        self.noise_type = noise_type
        self.backdoor_validation = backdoor_validation

        if self.label_flipping:
            self.dataset = labelFlipping(self.dataset, self.indices, self.poisoned_sample_ratio, self.targeted, self.target_label, self.target_changed_label)
        if self.data_poisoning:
            self.dataset = datapoison(self.dataset, self.indices, self.poisoned_sample_ratio, self.noise_injected_ratio, self.targeted, self.target_label, self.noise_type, self.backdoor_validation)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

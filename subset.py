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
                 label_flipping=False,
                 data_poisoning=False,
                 poisoned_sample_ratio=0,
                 noise_injected_ratio=0,
                 targeted=False,
                 target_label=0,
                 target_changed_label=0,
                 noise_type="salt"):
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

        if self.label_flipping:
            self.dataset = labelFlipping(self.dataset, self.indices, self.poisoned_sample_ratio, self.targeted, self.target_label, self.target_changed_label)
        if self.data_poisoning:
            self.dataset = datapoison(self.dataset, self.indices, self.poisoned_sample_ratio, self.noise_injected_ratio, self.targeted, self.target_label, self.noise_type)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import concurrent.futures
import random
import copy 


def plot_tensor(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

class DynamicDataset(torch.utils.data.Dataset):
    # def __init__(self, dataset, transform=None):
    #     self.dataset = copy.deepcopy(dataset)
    #     self.transform = transform

    # def __getitem__(self, index):
    #     image, label = self.dataset[index]
    #     if self.transform:
    #         image = self.transform(image)
    #     return image, label

    # def __len__(self):
    #     return len(self.dataset)
    def __init__(self, dataset, transform=None):
        self.dataset = copy.deepcopy(dataset)
        self.transform = transform
        
        # Preprocess and store the images using concurrent futures for parallel processing
        self.preprocessed_data = self.preprocess_data()

    def preprocess_data(self):
        preprocessed_data = []
        
        def process_sample(index):
            image, label = self.dataset[index]
            if self.transform:
                image = self.transform(image)
            return (image, label)
        
        # # Use ThreadPoolExecutor for parallel processing with multiple threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_sample, range(len(self.dataset))))
        
        preprocessed_data.extend(results)
        return preprocessed_data

    def __getitem__(self, index):
        return self.preprocessed_data[index]

    def __len__(self):
        return len(self.dataset)


class DynamicDataLoader():
    def __init__(self, dataloader, transform):
        self.dataloader = dataloader
        self.transform = transform

    def __iter__(self):
        for batch in self.dataloader:
            images, labels = batch
            transformed_images = torch.stack([self.transform(image) for image in images])
            yield transformed_images, labels

    def __len__(self):
        return len(self.dataloader)


def dynamic_transformer(initilSizeX:int=28, initilSizeY:int=28):
    perspective_transformer = v2.RandomPerspective(distortion_scale=0.6, p=1.0)
    rotater = v2.RandomRotation(degrees=(0, 180))
    inverter = v2.RandomInvert()
    affine_transfomer = v2.RandomAffine(degrees=(30, 70))
    elastic_transformer = v2.ElasticTransform()
    random_crop_x = random.randint(int(initilSizeX*2/3), initilSizeX)
    random_crop_y = random.randint(int(initilSizeY*2/3), initilSizeY)
    randomcrop_resize = v2.Compose([
    v2.RandomCrop(size=(random_crop_x, random_crop_y)),
    v2.Resize(size=(initilSizeX,initilSizeY))
    ])
    resize_cropper = v2.RandomResizedCrop(size=(initilSizeX, initilSizeY))
    augMix = v2.AugMix()
    num_aug = random.randint(1,2)
    aug_list = [randomcrop_resize, rotater, elastic_transformer, affine_transfomer, resize_cropper, augMix]
    rand_applier = random.sample(aug_list, num_aug)  
    applier = v2.RandomApply(transforms=rand_applier, p=1)
    return applier
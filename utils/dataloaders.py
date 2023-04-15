import os
import cv2
import numpy as np
import yaml
import torch
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


with open("data/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

COLORMAP = list(config["color"].values())


class CustomDataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        image_dir = os.path.join(dir, "images")
        self.images = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.images.sort()
        
        mask_dir = os.path.join(dir, "masks")
        self.masks = [os.path.join(mask_dir, x) for x in os.listdir(mask_dir)]
        self.masks.sort()
        
        self.transform = transform

    @staticmethod
    def convert_to_segmentation_mask(mask):
        height, width = mask.shape[: 2]
        segmentation_mask = np.zeros((height, width, len(COLORMAP)))
        
        for i, label in enumerate(COLORMAP):
            segmentation_mask[:, :, i] = np.all(mask == label, axis=-1)
        
        return segmentation_mask
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.masks[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert_to_segmentation_mask(mask)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]
            mask = transformed["mask"]
            
            return image.float(), mask.argmax(dim=2).long()
        
        else:
            return image.float(), mask.long()
        
        
def create_dataloaders(dir, image_size, batch_size, num_workers=os.cpu_count()):
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    dataset = CustomDataset(dir=dir, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return dataloader, dataset
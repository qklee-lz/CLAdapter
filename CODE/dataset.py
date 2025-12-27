import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np
import os
import pandas as pd


class CLAdapterDataset(Dataset):
    def __init__(self, mode, img_size, root):
        # example of a dataset
        self.mode = mode
        if self.mode == 'train':
            self.df = pd.read_csv("./data/Industry/defect_supervised/yoke-suspension/anno/train.csv")            
        elif self.mode == 'valid':
            self.df = pd.read_csv("./data/Industry/defect_supervised/yoke-suspension/anno/val.csv")            
        elif self.mode == 'test':
            self.df = pd.read_csv("./data/Industry/defect_supervised/yoke-suspension/anno/test.csv")           

        self.images = [os.path.join(root, path) for path in self.df['image_path'].tolist()]
        self.labels = self.df['label'].tolist()
        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) #Image_Net
        # self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32) #CLIP
        # self.std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) #CLIP
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.img_size = img_size
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'), dtype=np.float32)
        image = self.transforms(image=image)['image']
        image = self.norm(image)
        image = torch.from_numpy(image.transpose((2,0,1)))
        label = torch.as_tensor(self.labels[index])
        return image, label
    
    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
    
    def get_transforms(self,):
        if self.mode == 'train':
            transforms=(A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.3, rotate_limit=180, border_mode=0, p=0.7),
            ]))
        else:
            transforms=(A.Compose([A.Resize(self.img_size, self.img_size)]))
        return transforms

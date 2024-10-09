import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class Rellis3D(Dataset):
    def __init__(self, dataset_root = "../data/rellis", split_type = 'train', transform=None, target_transform=None):
        self.root_dir = dataset_root
        self.transform = transform 
        self.target_transform = target_transform     
        self.split_type = split_type  
        if split_type == 'train':
            self.images_dir =  os.path.join(self.root_dir, 'Rellis3D', 'images', 'training')
            self.annotations_dir = os.path.join(self.root_dir, 'Rellis3D', 'annotations', 'training')
            
        elif split_type == 'val':
            self.images_dir =  os.path.join(self.root_dir, 'Rellis3D', 'images', 'validation')
            self.annotations_dir = os.path.join(self.root_dir, 'Rellis3D', 'annotations', 'validation')
        
        if not os.path.exists(self.images_dir):
            raise ValueError("{} directory does not exist".format(self.images_dir))
        elif not os.path.exists(self.annotations_dir):
            raise ValueError("{} directory does not exist".format(self.annotations_dir))
        else:
            pass
        
        self.image_files = sorted(os.listdir(images_dir), key = lambda x: [len(x), x])        

    def __getitem__(self, index):
        img = np.asarray(
            Image.open(os.path.join(self.images_dir, self.image_files[index])))
        annotation = np.asarray(
            Image.open(os.path.join(self.annotations_dir, self.image_files[index].replace('.jpg', '.png'))))
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            annotation = self.target_transform(annotation)
        
        return img, annotation
    
    def __len__(self):
        return len(self.image_files)
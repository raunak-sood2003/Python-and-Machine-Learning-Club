import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Torch vision is a PyTorch library for computer vision
import numpy as np
from tqdm import tqdm # Progress Bar library
import matplotlib.pyplot as plt

class DOGVCAT_DATA(Dataset):
    def __init__(self, data_npy, transforms = None):
        self.data_npy = data_npy
        self.transforms = transforms
        
        self.X_train = torch.Tensor([i[0] for i in self.data_npy]).view(-1, 1, 50, 50) # (# of imgs, channels, img_size, img_size)
        self.X_train /= 255 # Normalizing
        self.y_train = torch.Tensor([i[1] for i in self.data_npy])
        
    def __len__(self): # Special method to define the length of the data (REQUIRED)
        return self.y_train.shape[0]
    def __getitem__(self, idx): # Special method to define how to access an item (REQUIRED)
        return self.X_train[idx], self.y_train[idx]
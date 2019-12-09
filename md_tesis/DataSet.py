from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
import numpy as np
from torchvision import transforms, utils

class CloudDataSet(Dataset):
    def __init__(self, cvs_file, root_dir, trnasform=None):
        super().__init__()
        self.set_train = pd.read_csv(csv_file)
        self.root_dir =  root_dir
        self.transform = transform

def __len__(self):
    return len(self.set_train)

def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    
    img_name = os.path.join(self.root_dir, self.set_train.iloc[idx, 0])
    image = io.imread(img_name)
    
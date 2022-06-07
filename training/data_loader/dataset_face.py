import numpy as np
import cv2
import os
import glob
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, path, resolution=512):
        self.resolution = resolution

        self.A_imgs = glob.glob(os.path.join(path, 'train_A', '*.*'))
        self.B_imgs = glob.glob(os.path.join(path, 'train_B', '*.*'))
        self.length = len(self.A_imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_A = cv2.imread(self.A_imgs[index], cv2.IMREAD_COLOR)
        img_A = cv2.resize(img_A, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        img_B = cv2.imread(self.B_imgs[index], cv2.IMREAD_COLOR)
        img_B = cv2.resize(img_B, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        
        img_A = img_A.astype(np.float32)/255.
        img_B = img_B.astype(np.float32)/255.

        img_A =  (torch.from_numpy(img_A) - 0.5) / 0.5
        img_B =  (torch.from_numpy(img_B) - 0.5) / 0.5
        
        img_A = img_A.permute(2, 0, 1).flip(0) # BGR->RGB
        img_B = img_B.permute(2, 0, 1).flip(0) # BGR->RGB

        return img_A, img_B


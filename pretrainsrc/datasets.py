import os,sys,re,glob,random
import pandas as pd
import librosa as lb
import IPython.display as ipd
import soundfile as sf
import numpy as np
import ast, joblib
from pathlib import Path

import librosa.display
from sklearn import preprocessing

#Deep learning from pytorch
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm
from torch.nn.parameter import Parameter
import copy, codecs
import sklearn.metrics
import audiomentations as AA

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self):
        lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
        return lam, 1 - lam

class WaveformDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 CFG,
                 prilabelp=1.0,
                 seclabelp=0.5,
                 mixup_prob = 0.15,
                 mixup_alpha = 2.0,
                 smooth=0.005,
                 train = True,
                 factor = 5
                 ):
      
        self.df = df.reset_index(drop=True)
        self.df["sort_index"] = self.df.index
        self.smooth = smooth
        self.prilabelp = prilabelp
        self.seclabelp = seclabelp
        self.train = train
        self.mixup_prob = mixup_prob
        self.cfg = CFG
        
        
        
    def __len__(self):
        return len(self.df)

    def load_img(self,row):
        image = torch.load(row.apath)[:,:3000]
        if not self.train:
            image = image[:,:501]
        else:
            image_mix = torch.zeros_like(image)
            perms = torch.randperm(self.factor)
            for i, perm in enumerate(perms):
                image_mix[:,i*self.frame:(i+1)*self.frame] = image[:,perm*self.frame:(perm+1)*self.frame]
                lam1, lam2 = self.mixup.get_lambda()
                image = lam1*image + lam2*image_mix
            
        labels = torch.zeros(self.cfg.CLASS_NUM, dtype=torch.float32) + self.smooth
        if row.sec_num != 0:
            labels[row.labels_id] = self.seclabelp
        if row.label_id != -1:
            labels[row.label_id] = self.prilabelp

        return image, labels
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, label = self.load_img(row) 
        if self.train:
            if (random.uniform(0,1) < self.mixup_prob):
                pair_idx = np.random.choice(len(self.df))
                row2 = self.df.iloc[pair_idx]
                img2, label2 = self.load_img(row)
                lam1, lam2 = self.mixup.get_lambda()
                img = lam1*img + lam2*img2
                label = lam1*label + lam2*label2
        weight = row.weight
            
        return img[None,:,:], label, weight
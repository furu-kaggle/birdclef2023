import os,sys,re,glob,random
import pandas as pd
import librosa as lb
import IPython.display as ipd
import soundfile as sf
import numpy as np
import cv2
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

from madgrad import MADGRAD

import audiomentations as AA
from audiomentations import (
    AddGaussianSNR,
)
import timm

class Model(nn.Module):
    def __init__(self,name,pretrained=True,path=None):
        super(Model, self).__init__()
        self.model = timm.create_model(name,pretrained=pretrained, drop_rate=0.4, drop_path_rate=0.2, in_chans=1)
        self.model.reset_classifier(num_classes=0)
        if path is not None:
          self.model.load_state_dict(torch.load(path))
        
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, CFG.CLASS_NUM)
        self.dropout = nn.Dropout(p=0.2)
        
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y=None, w=None):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        if (y is not None)&(w is not None):
            loss = self.loss_fn(x, y)
            #ラベル方向で平均化する
            loss = (loss.mean(dim=1) * w) / w.sum()
            loss = loss.sum()
            return x, loss
        else:
            return x
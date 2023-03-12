import os,sys,re,glob,random
import pandas as pd
import librosa as lb
import IPython.display as ipd
import soundfile as sf
import numpy as np
import cv2
import ast, joblib
from pathlib import Path

%matplotlib inline
import matplotlib.pyplot as plt
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

class Record:
    '''
    Records labels and predictions within one epoch
    '''
    def __init__(self):
        self.labels = []
        self.preds = []
        self.f1score = 0
        
        #padding array
        self.pad_rows = np.ones((1,len(unique_key)))
        
    def update(self, cur_logits, cur_labels):
        cur_labels = cur_labels.detach().cpu().numpy()
        cur_preds = cur_logits.sigmoid().detach().cpu().numpy()
        self.labels.append(cur_labels)
        self.preds.append(cur_preds)

    def get_f1score(self):
        labels = np.concatenate(self.labels,axis=0).astype(float)
        preds = np.concatenate(self.preds,axis=0).astype(float)
        for _ in range(5):
            labels = np.append(labels, self.pad_rows, axis=0)
            preds  = np.append(preds,  self.pad_rows, axis=0)
        score = sklearn.metrics.average_precision_score(
            labels, preds, average='macro'
        )
        return score
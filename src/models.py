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
import torch, torchaudio
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

import timm

class Model(nn.Module):
    def __init__(self,CFG,pretrained=True,path=None,training=True):
        super(Model, self).__init__()
        self.model = timm.create_model(CFG.model_name,pretrained=pretrained, drop_rate=0.2, drop_path_rate=0.2, in_chans=1)
        self.model.reset_classifier(num_classes=0)
        if path is not None:
          self.model.load_state_dict(torch.load(path))
        
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, CFG.CLASS_NUM)
        self.dropout = nn.Dropout(p=0.2)
        
        self.loss_fn = nn.BCEWithLogitsLoss()#(reduction='none')
        self.training = training
        
        #wav to image helper
        self.mel = torchaudio.transforms.MelSpectrogram(
            n_mels = CFG.n_mel, 
            sample_rate= CFG.sr, 
            f_min = CFG.fmin, 
            f_max = CFG.fmax, 
            n_fft = CFG.n_fft, 
            hop_length=CFG.hop_len,
            norm = None,
            power = CFG.power,
            mel_scale = 'htk')
        
        self.ptodb = torchaudio.transforms.AmplitudeToDB(top_db=CFG.top_db)
        
    def torch_mono_to_color(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)

        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = torch.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.to(torch.uint8)
        else:
            V = torch.zeros_like(X, dtype=torch.uint8)

        return V
    
    def wavtoimg(self, wav):
        melimg= self.mel(wav)
        dbimg = self.ptodb(melimg)
        cimg = self.torch_mono_to_color(dbimg)
        img = dbimg.to(torch.float32) / 255.0

        return img

    def forward(self, x, y=None, w=None):
        if self.training:
            # shape:(b, outm, inm, time)
            # inner mixup (0)
            x1 = 0.5*self.wavtoimg(x[:,0,0,:]) + 0.5*self.wavtoimg(x[:,0,1,:])
            # inner mixup (1)
            x2 = 0.5*self.wavtoimg(x[:,1,0,:]) + 0.5*self.wavtoimg(x[:,1,1,:])
            # outer mixup
            x = 0.5*x1 + 0.5*x2
            y = 0.5*y[:,0,:] + 0.5*y[:,1,:]
        else:
            x = self.wavtoimg(x)
        
        x = self.model(x[:,None,:,:])
        x = self.dropout(x)
        x = self.fc(x)
        if (y is not None)&(w is not None):
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            return x
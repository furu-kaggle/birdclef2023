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

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self):
        lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
        return lam, 1 - lam

class Model(nn.Module):
    def __init__(self,CFG,pretrained=True,path=None,training=True):
        super(Model, self).__init__()
        self.model = timm.create_model(
            CFG.model_name,
            pretrained=pretrained, 
            drop_rate=0.2, 
            drop_path_rate=0.2, 
            in_chans=1,
            global_pool="",
            num_classes=0
        )
        if path is not None:
          self.model.load_state_dict(torch.load(path))
        
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, CFG.CLASS_NUM)
        self.dropout = nn.Dropout(p=0.2)
        
        self.loss_fn = nn.BCEWithLogitsLoss()#(reduction='none')
        self.training = training

        self.mixup_in = Mixup(mixup_alpha=2.0)
        self.mixup_out = Mixup(mixup_alpha=2.0)
        
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
    
    def wavtoimg(self, wav, power=2):
        self.mel.power = power
        melimg= self.mel(wav)
        dbimg = self.ptodb(melimg)
        img = (dbimg.to(torch.float32) + 80)/80
        return img

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def forward(self, x, y=None, w=None):
        if self.training:
            # shape:(b, outm, inm, time)
            # inner mixup (0)
            power = random.uniform(1.9,2.1)
            lam1, lam2 = self.mixup_in.get_lambda()
            x1 = lam1*self.wavtoimg(x[:,0,0,:], power) + lam2*self.wavtoimg(x[:,0,1,:], power)

            # inner mixup (1)
            lam1, lam2 = self.mixup_in.get_lambda()
            x2 = lam1*self.wavtoimg(x[:,1,0,:], power) + lam2*self.wavtoimg(x[:,1,1,:], power)
            
            # outer mixup
            lam1, lam2 = self.mixup_out.get_lambda()
            x = lam1*x1 + lam2*x2
            y = lam1*y[:,0,:] + lam2*y[:,1,:]
        else:
            x = self.wavtoimg(x)
        
        x = self.model(x[:,None,:,:])
        x = self.gem(x)[:,:,0,0]
        x = self.dropout(x)
        x = self.fc(x)
        if (y is not None)&(w is not None):
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            return x
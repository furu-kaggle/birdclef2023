import os,sys,re,glob,random
import pandas as pd
import librosa as lb
import IPython.display as ipd
import soundfile as sf
import numpy as np
import ast, joblib
from pathlib import Path
import torchaudio as ta

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


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        lams = []
        inv_lams = []
        for _ in range(batch_size):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            lams.append(lam)
            inv_lams.append(1.0-lam)
        return torch.tensor(lams, dtype=torch.float32), torch.tensor(inv_lams, dtype=torch.float32)

class Model(nn.Module):
    def __init__(self,CFG,pretrained=False,training=True):
        super(Model, self).__init__()
        self.cfg = CFG
        self.model = timm.create_model(
            CFG.model_name,
            pretrained=pretrained, 
            drop_rate=CFG.backbone_dropout, 
            drop_path_rate=CFG.backbone_droppath, 
            in_chans=1, 
            global_pool="",
            num_classes=0,
        )
        
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, CFG.CLASS_NUM)
        init_layer(self.fc)
        self.dropout = nn.Dropout(p=CFG.head_dropout)
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.training = training
        self.factor = 6
        self.frame = CFG.frame

        self.mixup_in = Mixup(mixup_alpha=CFG.mixup_alpha_in)
        self.mixup_out = Mixup(mixup_alpha=CFG.mixup_alpha_out)
        
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
        self.freq_mask = ta.transforms.FrequencyMasking(12, iid_masks=True)
    
    def wavtoimg(self, wav, power=2):
        self.mel.power = power
        melimg= self.mel(wav)
        dbimg = self.ptodb(melimg)
        img = (dbimg.to(torch.float32) + 80)/80
        if (self.training)&(random.uniform(0,1) < 0.25):
            img = self.freq_mask(img)
        return img

    def gem_pooling(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def inner_mixup(self, x, x_mix, batch_size):        
        perms = torch.randperm(self.factor).to(x.device)
        for i, perm in enumerate(perms):
            x_mix[:,:,i*self.frame:(i+1)*self.frame] = x[:,:,perm*self.frame:(perm+1)*self.frame]

        lam1, lam2 = self.mixup_in.get_lambda(batch_size)
        lam1, lam2 = lam1.to(x.device), lam2.to(x.device)
        x = lam1[:,None,None]*x + lam2[:,None,None]*x_mix
        return x

    def forward(self, x, y=None, w=None):
        if self.training:
            power = random.uniform(self.cfg.augpower_min,self.cfg.augpower_min)
            batch_size = x.shape[0]

            if (random.uniform(0,1) < self.cfg.mixup_in_prob1):
                x0, x0_mix = self.wavtoimg(x[:,0,0,:], power), self.wavtoimg(x[:,0,1,:], power)
                x0 = self.inner_mixup(x0, x0_mix, batch_size)
            else:
                x0 = self.wavtoimg(x[:,0,0,:], power)

            if (random.uniform(0,1) < self.cfg.mixup_in_prob2):
                x1, x1_mix = self.wavtoimg(x[:,1,0,:], power), self.wavtoimg(x[:,1,1,:], power)
                x1 = self.inner_mixup(x1, x1_mix, batch_size)
            else:
                x1 = self.wavtoimg(x[:,1,0,:], power)
            
            if (random.uniform(0,1) < self.cfg.mixup_out_prob):
                lam1, lam2 = self.mixup_out.get_lambda(batch_size)
                lam1, lam2 = lam1.to(x.device), lam2.to(x.device)
                x = lam1[:,None,None]*x0 + lam2[:,None,None]*x1
                y = lam1[:,None]*y[:,0,:] + lam2[:,None]*y[:,1,:]
            else:
                x = x0
                y = y[:,0,:]
        else:
            x = self.wavtoimg(x)
        x  = x[:,None,:,:-1]
        if  (self.training)&(self.factor > 3):
            #print(x.shape)
            b, c, f, t = x.shape
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(b*self.factor, t//self.factor, f, c)
            x = x.permute(0, 3, 2, 1)
            #print(x.shape)
            x = self.model(x)
            b, c, f, t = x.shape
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(b//self.factor, t*self.factor, f, c)
            x = x.permute(0, 3, 2, 1)
        else:
            x = self.model(x)

        x = self.gem_pooling(x)[:,:,0,0]
        x = self.dropout(x)
        x = self.fc(x)
        if (y is not None):
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            return x
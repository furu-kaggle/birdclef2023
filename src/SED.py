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

class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()

        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (bs, channel(2304), n_time(47))
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.cla(x) #(bs, class(2304 to 264),time(47))
        x = torch.sum(norm_att * cla, dim=2) #(bs, class(2304 to 264)) from time(47).sum()
        return x, norm_att, cla

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
    def __init__(self,CFG,pretrained=False,path=None,training=True):
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
        if path is not None:
          self.model.load_state_dict(torch.load(path))
        
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, CFG.CLASS_NUM)
        init_layer(self.fc)
        self.att_block = AttBlockV2(in_features, in_features)
        self.dropout = nn.Dropout(p=CFG.head_dropout)
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.training = training

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
    
    def wavtoimg(self, wav, power=2):
        self.mel.power = power
        melimg= self.mel(wav)
        dbimg = self.ptodb(melimg)
        img = (dbimg.to(torch.float32) + 80)/80
        return img

    def forward(self, x, y=None, w=None):
        if self.training:
            power = random.uniform(self.cfg.augpower_min,self.cfg.augpower_min)
            batch_size = x.shape[0]
            # shape:(b, outm, inm, time)
            # inner mixup (0)
            lam1, lam2 = self.mixup_in.get_lambda(batch_size)
            lam1, lam2 = lam1.to(x.device), lam2.to(x.device)
            x1 = lam1[:,None,None]*self.wavtoimg(x[:,0,0,:], power) + lam2[:,None,None]*self.wavtoimg(x[:,0,1,:], power)
            

            # inner mixup (1)
            lam1, lam2 = self.mixup_in.get_lambda(batch_size)
            lam1, lam2 = lam1.to(x.device), lam2.to(x.device)
            x2 = lam1[:,None,None]*self.wavtoimg(x[:,1,0,:], power) + lam2[:,None,None]*self.wavtoimg(x[:,1,1,:], power)

            if (random.uniform(0,1) < self.cfg.mixup_out_prob):
                lam1, lam2 = self.mixup_out.get_lambda(batch_size)
                lam1, lam2 = lam1.to(x.device), lam2.to(x.device)
                x = lam1[:,None,None]*x1 + lam2[:,None,None]*x2
                y = lam1[:,None]*y[:,0,:] + lam2[:,None]*y[:,1,:]
            else:
                x = x1
                y = y[:,0,:]
        else:
            x = self.wavtoimg(x)
        x  = x[:,None,:,:-1]
        x = self.model(x)
        x = torch.mean(x, dim=2)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2 #(batch_size, channel(2304), time(47))
        (x, norm_att, segmentwise_output) = self.att_block(x)
        x = self.dropout(x)
        x = self.fc(x) #(batch_size, channel(2304), time(47))
        if (y is not None):
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            return x
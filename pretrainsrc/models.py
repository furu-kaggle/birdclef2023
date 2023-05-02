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

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def gem_pooling(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def forward(self, x):
        ret = self.gem_pooling(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

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
        self.dropout = nn.Dropout(p=CFG.head_dropout)
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.training = training
        self.factor = 6
        self.frame = CFG.frame

        self.mixup_in = Mixup(mixup_alpha=CFG.mixup_alpha_in)
        self.mixup_out = Mixup(mixup_alpha=CFG.mixup_alpha_out)
        self.gem = GeM()

    def forward(self, x, y=None, w=None):
        if self.training:
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

        x = self.gem(x)[:,:,0,0]
        x = self.dropout(x)
        x = self.fc(x)
        if (y is not None):
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            return x
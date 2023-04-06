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
from timm.models.nfnet import ScaledStdConv2d

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
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

    def forward(self, x, mask=None):
        # x: (bs, channel(2304), n_time(47))
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1) #(bs, class(2304 to 264),time(47))
        #ここでtimeにマスクすれば良い？
        if mask is not None:
            norm_att = norm_att * mask[:,None,:]
        cla = self.nonlinear_transform(self.cla(x)) #(bs, class(2304 to 264),time(47))
        x = torch.sum(norm_att * cla, dim=2) #(bs, class(2304 to 264)) from time(47).sum()
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            print("beware of sigmoid")
            return torch.sigmoid(x)

##################################################
# Binary Focall Loss
##################################################

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=1):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        bce_loss = self.loss_fct(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.5, (1. - probas)**self.gamma * bce_loss, probas**self.gamma * bce_loss)
        loss = loss.mean()
        return loss

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self):
        lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
        return lam, 1 - lam

class Model(nn.Module):
    def __init__(self,CFG,pretrained=False,path=None,training=True):
        super(Model, self).__init__()
        self.model = timm.create_model(
            CFG.model_name,
            pretrained=pretrained, 
            drop_rate=0.2, 
            drop_path_rate=0.2, 
            in_chans=1, 
            global_pool="",
            num_classes=0,
        )
        if path is not None:
          self.model.load_state_dict(torch.load(path))
        
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, in_features)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.att_block = AttBlockV2(in_features, CFG.CLASS_NUM)
        #self.dropout = nn.Dropout(p=0.2)
        
        self.loss_fn = nn.BCEWithLogitsLoss()#(reduction='none')
        #self.loss_fn = BCEFocalLoss(gamma=2)
        self.training = training

        self.mixup = Mixup(mixup_alpha=2.0)

        self.freq_mask = ta.transforms.FrequencyMasking(12, iid_masks=True)
        self.time_mask = ta.transforms.TimeMasking(50, iid_masks=True)
        
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
        img = (dbimg.to(torch.float32) + 80)/80
        if (self.training)&(random.uniform(0,1) < 0.5):
            img = self.freq_mask(img)
        if (self.training)&(random.uniform(0,1) < 0.5):
            img = self.time_mask(img)


        return img

    def forward(self, x, y=None, w=None):
        if self.training:
            # shape:(b, outm, inm, time)
            # inner mixup (0)
            lam1, lam2 = self.mixup.get_lambda()
            x1 = lam1*self.wavtoimg(x[:,0,0,:]) + lam2*self.wavtoimg(x[:,0,1,:])

            # inner mixup (1)
            lam1, lam2 = self.mixup.get_lambda()
            x2 = lam1*self.wavtoimg(x[:,1,0,:]) + lam2*self.wavtoimg(x[:,1,1,:])
            
            # outer mixup
            lam1, lam2 = self.mixup.get_lambda()
            x = lam1*x1 + lam2*x2
            y = lam1*y[:,0,:] + lam2*y[:,1,:]
        else:
            x = self.wavtoimg(x)
        
        x = self.model(x[:,None,:,:])
        x = torch.mean(x, dim=2)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2 #(batch_size, channel(2304), time(47))
        x = self.dropout1(x).transpose(1, 2)
        x = F.relu_(self.fc(x)) #(batch_size, channel(2304), time(47))
        x = self.dropout2(x).transpose(1, 2)
        (x, norm_att, segmentwise_output) = self.att_block(x, None)
        segx = segmentwise_output.max(dim=2).values
        if (y is not None)&(w is not None):
            loss = self.loss_fn(x, y) #+ 0.5*self.loss_fn(segx, y)
            return x, loss
        else:
            return x
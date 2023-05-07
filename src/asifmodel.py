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

class MixStripes(nn.Module):
    def __init__(self, dim, mix_width, stripes_num):
        """Mix stripes.
        Args:
          dim: int, dimension along which to mix
          mix_width: int, maximum width of stripes to mix
          stripes_num: int, how many stripes to mix
        """
        super(MixStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.mix_width = mix_width
        self.stripes_num = stripes_num
        
    def transform_slice(self, e, r, total_width):
        """e: (channels, time_steps, freq_bins)"""
        """r: (channels, time_steps, freq_bins)"""
        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.mix_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = .5*e[:, bgn : bgn + distance, :] + .5*r[:, bgn : bgn + distance, :]
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = .5*e[:, :, bgn : bgn + distance] + .5*r[:, :, bgn : bgn + distance]

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]
            index = torch.randperm(batch_size)
            rand_ = input[index, :]
            for n in range(batch_size):
                self.transform_slice(input[n], rand_[n], total_width)

            return input
        
class ASIFSpecMixAugmentation(nn.Module):
    def __init__(self, time_mix_width, time_stripes_num, freq_mix_width,
        freq_stripes_num, factor=6):

        super(ASIFSpecMixAugmentation, self).__init__()
        layers = [
            MixStripes(dim=2, mix_width=freq_mix_width, stripes_num=freq_stripes_num),
            MixStripes(dim=3, mix_width=time_mix_width,stripes_num=time_stripes_num)
        ]
        self.mixer = nn.Sequential(*layers)
        self.factor = factor

    def forward(self, x):
        bs = x.shape[0]
        x_mix = torch.zeros_like(x)
        for i in range(0, bs, self.factor):
            x_mix[i: i+self.factor] = self.mixer(x[i:i+self.factor])

        return x

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

        #self.model.stem.conv1 = ScaledStdConv2d(1, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        
        in_features = self.model.num_features
        self.fc = nn.Linear(in_features, CFG.CLASS_NUM)
        self.dropout = nn.Dropout(p=0.2)
        
        self.loss_fn = nn.BCEWithLogitsLoss()#(reduction='none')
        self.training = training

        self.mixup_in = Mixup(mixup_alpha=2.0)
        self.mixup_out = Mixup(mixup_alpha=2.0)

        self.gem = GeM()
        
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
        self.factor = 6
        self.frame = 500
        self.asifspecmix_stem = ASIFSpecMixAugmentation(
            time_mix_width=20, 
            time_stripes_num=min(3,self.factor), 
            freq_mix_width=4,
            freq_stripes_num=min(3,self.factor),
            factor = self.factor
        )
        self.asifspecmix_layer1 = ASIFSpecMixAugmentation(
            time_mix_width=20, 
            time_stripes_num=min(3,self.factor), 
            freq_mix_width=4,
            freq_stripes_num=min(3,self.factor),
            factor = self.factor
        )
        self.asifspecmix_layer2 = ASIFSpecMixAugmentation(
            time_mix_width=10, 
            time_stripes_num=min(3,self.factor), 
            freq_mix_width=2,
            freq_stripes_num=min(3,self.factor),
            factor = self.factor
        )
        self.asifspecmix_layer3 = ASIFSpecMixAugmentation(
            time_mix_width=4, 
            time_stripes_num=min(3,self.factor), 
            freq_mix_width=1,
            freq_stripes_num=min(3,self.factor),
            factor = self.factor
        )
        self.stem = self.model.stem
        self.stage0 = self.model.stages[0]
        self.stage1 = self.model.stages[1]
        self.stage2 = self.model.stages[2]
        self.stage3 = self.model.stages[3]
        self.final_conv = self.model.final_conv
        self.final_act = self.model.final_act
        
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

    def update_factor(self, factor):
        self.factor = factor
        self.asifspecmix_stem = ASIFSpecMixAugmentation(
            time_mix_width=10, 
            time_stripes_num=1, 
            freq_mix_width=2,
            freq_stripes_num=1,
            factor = self.factor
        )
        self.asifspecmix_layer1 = ASIFSpecMixAugmentation(
            time_mix_width=5, 
            time_stripes_num=1, 
            freq_mix_width=1,
            freq_stripes_num=1,
            factor = self.factor
        )
        self.asifspecmix_layer2 = ASIFSpecMixAugmentation(
            time_mix_width=2, 
            time_stripes_num=1, 
            freq_mix_width=1,
            freq_stripes_num=1,
            factor = self.factor
        )
    
    def wavtoimg(self, wav, power=2):
        self.mel.power = power
        melimg= self.mel(wav)
        dbimg = self.ptodb(melimg)
        img = (dbimg.to(torch.float32) + 80)/80
        return img

    def forward(self, x, y=None, w=None):
        if self.training:
            # shape:(b, outm, inm, time)
            # inner mixup (0)
            power = random.uniform(1.9,2.1)
            batch_size = x.shape[0]
            if random.uniform(0,1) < 1.0:
                lam1, lam2 = self.mixup_out.get_lambda(batch_size)
                lam1, lam2 = lam1.to(x.device), lam2.to(x.device)
                x = lam1[:,None,None]*self.wavtoimg(x[:,0,:], power) + lam2[:,None,None]*self.wavtoimg(x[:,1,:], power)
                y = lam1[:,None]*y[:,0,:] + lam2[:,None]*y[:,1,:]
            else:
                x = self.wavtoimg(x[:,0,:], power)
                y = y[:,0,:]
        else:
            x = self.wavtoimg(x)
        x  = x[:,None,:,:-1]
        if  self.training:
            if random.uniform(0,1) < 0.5:
                x_mix = torch.zeros_like(x).to(x.device)
                perms = torch.randperm(self.factor).to(x.device)
                for i, perm in enumerate(perms):
                    x_mix[:,:,:,i*self.frame:(i+1)*self.frame] = x[:,:,:,perm*self.frame:(perm+1)*self.frame]

                lam1, lam2 = self.mixup_in.get_lambda(batch_size)
                lam1, lam2 = lam1.to(x.device), lam2.to(x.device)
                x = lam1[:,None,None,None]*x + lam2[:,None,None,None]*x_mix
            
            #print(x.shape)
            b, c, f, t = x.shape
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(b*self.factor, t//self.factor, f, c)
            x = x.permute(0, 3, 2, 1)
            #print(x.shape)
            #x  = self.model(x)
            x = self.stem(x) #torch.Size([12, 128, 32, 125])
            if random.uniform(0,1) < 0.25:
                x = self.asifspecmix_stem(x)
            x = self.stage0(x) #(12, 256, 32, 125)
            if random.uniform(0,1) < 0.125:
                x = self.asifspecmix_layer1(x)
            x = self.stage1(x) ##(12, 512, 16, 63)
            if random.uniform(0,1) < 0.05:
                x = self.asifspecmix_layer2(x)
            x = self.stage2(x) #(12, 1536, 8, 32)
            #x = self.asifspecmix_layer3(x)
            x = self.stage3(x) #(12, 1536, 4, 16)
            x = self.final_conv(x)
            x = self.final_act(x)
            #print(x.shape)
            b, c, f, t = x.shape
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(b//self.factor, t*self.factor, f, c)
            x = x.permute(0, 3, 2, 1)
        else:
            x = self.model(x)

        x = self.gem(x)[:,:,0,0]
        x = self.dropout(x)
        x = self.fc(x)
        if (y is not None)&(w is not None):
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            return x
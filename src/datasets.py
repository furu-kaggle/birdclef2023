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

train_aug = AA.Compose(
    [
        AA.AddBackgroundNoise(
            sounds_path="data/ff1010bird_nocall/nocall", min_snr_in_db=5, max_snr_in_db=10, p=0.5
        ),
        AA.AddBackgroundNoise(
            sounds_path="data/train_soundscapes/nocall", min_snr_in_db=5, max_snr_in_db=10, p=0.5
        ),
        AA.AddBackgroundNoise(
            sounds_path="data/aicrowd2020_noise_30sec/noise_30sec",
            min_snr_in_db=5,
            max_snr_in_db=10,
            p=0.75,
        ),
        AA.AddBackgroundNoise(
            sounds_path="data/useesc50",
            min_snr_in_db=5,
            max_snr_in_db=10,
            p=0.75,
        ),
        AA.AddGaussianSNR(
            min_snr_in_db=5,max_snr_in_db=10.0,p=0.25
        ),
        AA.Shift(
            min_fraction=0.1, max_fraction=0.1, rollover=False, p=0.25
        ),
        AA.LowPassFilter(
            min_cutoff_freq=100, max_cutoff_freq=10000, p=0.25
        )
    ]
)


class WaveformDataset(Dataset):
    def __init__(self,
                 CFG,
                 df: pd.DataFrame,
                 prilabelp=1.0,
                 seclabelp=0.5,
                 mixup_prob = 0.15,
                 smooth=0.005,
                 train = True
                 ):
      
        self.df = df.reset_index(drop=True)
        self.CFG = CFG
        self.aug = train_aug
        self.sr = CFG.sr
        self.df["sort_index"] = self.df.index
        self.smooth = smooth
        self.prilabelp = prilabelp - smooth
        self.seclabelp = seclabelp - smooth
        self.train = train

        
        #Matrix Factorization (サブラベル同士は相関なしとして扱う)
        self.mfdf = self.df[(self.df.sec_num > 0)][["label_id","labels_id"]].explode("labels_id").reset_index(drop=True)
        
        #mixupするlabel_idリストを作成する
        self.mixup_idlist = self.mfdf.groupby("label_id").labels_id.apply(list).to_dict()
        
        #mixupする先はシングルラベルにする
        sdf = self.df[(self.df.sec_num==0)|(self.df.primary_label=="lotcor1")]
        
        #label_idリストからレコード番号を取得し、レコード番号からランダムサンプリングする
        self.id2record = sdf.groupby("label_id").sort_index.apply(list)
        
    def crop_or_pad(self, y, length, is_train=False, start=None):
        if len(y) < length//2:
            if is_train:
                wid = length//2 - len(y)
                start = np.random.randint(length//2, length//2 + wid)
                y_cp = np.zeros(length,dtype=np.float32)
                y_cp[start : start + len(y)] = y
                y = y_cp
            else:
                y = np.concatenate([y, np.zeros(length - len(y))])
        elif len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])

        elif len(y) > length:
            if not is_train:
                start = start or 0
            else:
                start = start or np.random.randint(len(y) - length)

            y = y[start:start + length]

        return y
        
        
    def __len__(self):
        return len(self.df)

    def load_audio(self,row):
        #データ読み込み
        data, sr = librosa.load(row.audio_paths, sr=self.sr, offset=0, mono=True)

        #augemnt1
        if (self.train)&(random.uniform(0,1) < row.weight):
             data = self.aug(samples=data, sample_rate=sr)

        #test datasetの最大長
        max_sec = len(data)//sr

        #0秒の場合は１秒として取り扱う
        max_sec = 1 if max_sec==0 else max_sec
        
        labels = torch.zeros(self.CFG.CLASS_NUM, dtype=torch.float32) + self.smooth
        if row.sec_num != 0:
            labels[row.labels_id] = self.seclabelp
        if row.label_id != -1:
            labels[row.label_id] = self.prilabelp
        

        return data, labels
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio1, label1 = self.load_audio(row)
        if self.train:
            if row.label_id in list(self.mixup_idlist.keys()):
                #FMからペアとなるラベルIDを取得
                pair_label_id = np.random.choice(self.mixup_idlist[row.label_id])
                pair_idx = np.random.choice(self.id2record[pair_label_id])
                row2 = self.df.iloc[pair_idx]
                audio2, label2 = self.load_audio(row2)
                label = torch.stack([label1,label2])
            else:
                audio2 = audio1
                label = torch.stack([label1,label1])
        else:
            audio = audio1
            label = label1
        weight = torch.tensor(row.weight, dtype=torch.float32)
        return audio1, audio2, label, weight


class DynamicalPaddingCollate:
    def __init__(self,CFG, quant_th):
            self.CFG = CFG
            self.quant_th = quant_th
            self.unit = self.CFG.period*self.CFG.sr
            self.cutoff = self.CFG.max_factor*self.CFG.period*self.CFG.sr
            
    def crop_or_pad(self, y, length, start=None,is_train=False):
        if len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])
            
        elif len(y) > length:
            if not is_train:
                start = start or 0
            else:
                start = start or np.random.randint(len(y) - length)

            y = y[start:start + length]

        return y
        
    def __call__(self, batch):
        audios1, audios2, labels, weights = list(zip(*batch))

        # calculate max time length of this batch
        time_array = np.append(
            np.array([len(ad) for ad in audios1]),
            np.array([len(ad) for ad in audios2]),
        axis=0)
        padding_len = int(np.quantile(time_array, self.quant_th))
        time_max = min(self.cutoff, padding_len)
        #frame Nomalization
        time_max = max(self.unit, time_max//self.unit * self.unit)
        audios1 = torch.stack([torch.tensor(self.crop_or_pad(ad, length = time_max),dtype=torch.float32) for ad in audios1])
        audios2 = torch.stack([torch.tensor(self.crop_or_pad(ad, length = time_max),dtype=torch.float32) for ad in audios2])
        audios = torch.stack([audios1, audios2])

        return audios, torch.stack(labels), torch.stack(weights)
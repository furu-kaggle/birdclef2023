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
        # AA.Shift(
        #     min_fraction=0.1, max_fraction=0.1, rollover=False, p=0.25
        # ),
        # AA.LowPassFilter(
        #     min_cutoff_freq=100, max_cutoff_freq=10000, p=0.25
        # )
    ]
)


class WaveformDataset(Dataset):
    def __init__(self,
                 CFG,
                 df: pd.DataFrame,
                 period = 5,
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
        self.period = period
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

        if (self.train)&(self.period >= 30):
            duration_seconds = librosa.get_duration(filename=row.audio_paths,sr=None)
            #訓練時にはランダムにスタートラインを変える(time shift augmentations)
            if (self.train)&(duration_seconds > max(35, self.period + 5)):
                offset = random.uniform(0, duration_seconds - self.period)
            else:
                offset = 0
        else:
            offset = 0
        #データ読み込み
        data, sr = librosa.load(row.audio_paths, sr=self.sr, offset=offset, duration=self.period, mono=True)

        #augemnt1
        if (self.train)&(random.uniform(0,1) < row.weight):
             data = self.aug(samples=data, sample_rate=sr)

        #test datasetの最大長
        max_sec = len(data)//sr

        #0秒の場合は１秒として取り扱う
        max_sec = 1 if max_sec==0 else max_sec
        
        data = self.crop_or_pad(data , length=sr*self.period,is_train=self.train)

        if self.train:
            #add train data
            if row.sec_num==0:
                pair_idx = np.random.choice(self.id2record[row.label_id])
                row_pair = self.df.iloc[pair_idx]
                data_pair, sr = librosa.load(row.audio_paths, sr=self.sr, offset=0, duration=self.period, mono=True)
                #augemnt2
                if (random.uniform(0,1) < row.weight):
                    data_pair = self.aug(samples=data_pair, sample_rate=sr)

            else:
                data_pair = data

            #test datasetの最大長
            max_sec = len(data_pair)//sr

            #0秒の場合は１秒として取り扱う
            max_sec = 1 if max_sec==0 else max_sec
        
            data_pair = self.crop_or_pad(data_pair, length=sr*self.period,is_train=False)

            data = np.stack([data, data_pair])
        
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
                audio = np.stack([audio1,audio2])
                label = np.stack([label1,label2])
            else:
                audio = np.stack([audio1,audio1])
                label = np.stack([label1,label1])
        else:
            audio = audio1
            label = label1
        weight = torch.tensor(row.weight, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        return audio, label, weight
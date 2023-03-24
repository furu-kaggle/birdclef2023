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
            sounds_path="data/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.5
        ),
        AA.AddBackgroundNoise(
            sounds_path="data/train_soundscapes/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
        AA.AddBackgroundNoise(
            sounds_path="data/aicrowd2020_noise_30sec/noise_30sec",
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.25,
        ),
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
        self.prilabelp = prilabelp - self.smooth
        self.seclabelp = seclabelp - self.smooth
        self.train = train

        
        #Matrix Factorization (サブラベル同士は相関なしとして扱う)
        self.mfdf = self.df[self.df.sec_num > 0][["label_id","labels_id"]].explode("labels_id").reset_index(drop=True)
        
        #mixupするlabel_idリストを作成する
        self.mixup_idlist = self.mfdf.groupby("label_id").labels_id.apply(list).to_dict()
        
        #mixupする先はシングルラベルにする
        sdf = self.df[(self.df.sec_num==0)|(self.df.primary_label=="lotcor1")]
        
        #label_idリストからレコード番号を取得し、レコード番号からランダムサンプリングする
        self.id2record = sdf.groupby("label_id").sort_index.apply(list)
        
    def crop_or_pad(self, y, length, is_train=False, start=None):
        if len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])

            n_repeats = length // len(y)
            epsilon = length % len(y)

            y = np.concatenate([y]*n_repeats + [y[:epsilon]])

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
        #訓練時にはランダムにスタートラインを変える(time shift augmentations)
        offset = 0
        #データ読み込み
        data, sr = librosa.load(row.audio_paths, sr=None, offset=offset, duration=30, mono=True)

        #augemnt
        if (self.train)&(random.uniform(0,1) < row.weight):
            data = self.aug(samples=data, sample_rate=sr)

        #test datasetの最大長
        max_sec = len(data)//sr 
        #0秒の場合は１秒として取り扱う
        max_sec = 1 if max_sec==0 else max_sec
        
        #データを5秒間隔でかつ7秒幅を取って区切る
        datas = [data[int(max(0, i) * sr):int(min(max_sec, i + self.period) * sr)] for i in range(0, max_sec, self.period)]

        #端は1秒短くなるので埋める
        datas[0] = self.crop_or_pad(datas[0] , length=sr*self.period)
        datas[-1] = self.crop_or_pad(datas[-1] , length=sr*self.period)

        datas = np.stack(datas)
        if (self.train):
            datas = datas[np.random.choice(len(datas),size=2)] #2つmixup用にサンプリング
        else:
            datas = datas[0]
        
        labels = torch.zeros(self.CFG.CLASS_NUM, dtype=torch.float32) + self.smooth
        labels[row.label_id] = self.prilabelp
        if row.sec_num != 0:
            labels[row.labels_id] = self.seclabelp

        return datas, labels
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio1, label1 = self.load_audio(row)
        if self.train:
            if row.label_id in list(self.mixup_idlist.keys()):
                #FMからペアとなるラベルIDを取得
                pair_label_id = np.random.choice(self.mixup_idlist[row.label_id])
                pair_idx = np.random.choice(self.id2record[pair_label_id])
                row2 = self.df.iloc[pair_idx]
                audio2, label2 = self.load_audio(row)
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
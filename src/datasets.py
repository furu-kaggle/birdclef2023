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
                 period = 5,
                 prilabelp=1.0,
                 seclabelp=0.5,
                 mixup_prob = 0.15,
                 smooth=0.005,
                 train = True,
                 mask = None
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
        if mask is not None:
            print("set mask")
            self.mask = mask
        else:
            self.mask = None

        self.period_idx = int(self.period*100)

        
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

    def get_audio(self, row, offset = 0):
        #データ読み込み
        data, sr = librosa.load(row.audio_paths, sr=self.sr, offset=offset, duration=self.period, mono=True)
        #augemnt1
        if (self.train)&(random.uniform(0,1) < row.weight):
             data = self.aug(samples=data, sample_rate=sr)

        #test datasetの最大長
        max_sec = len(data)//sr

        #0秒の場合は１秒として取り扱う
        max_sec = 1 if max_sec==0 else max_sec
        
        data = self.crop_or_pad(data , length=sr*self.period,is_train=False)
        """
        (1)マスクに対して対象のIDがあるかどうか判定
        No→処理終了し、画像サイズ(メル数、時間軸の長さ)となる全て１のマスクを生成する
        YES→(2)へ進む
        (2)tmp変数にマスクを取得するが、この場合、スペクトル画像全て(t,f)が入っており、入力画像サイズが時間軸方向で異なる場合がある。以下の場合分けが必要
        (i)時間軸方向が入力画像サイズの時間軸方向よりも小さい場合
        時間軸方向のパディングのみ適用し、余った余白は0を埋める
        (ii)時間軸方向が入力画像サイズの時間軸方向よりも大きい場合
        offsetからoffset + period_idxまで埋めるが、offset + period_idxが時間軸方向よりも長くなる場合は時間軸方向目一杯まで埋めた上で残りの余白を0埋めする
        """

        if self.mask is not None:
            if row.filename_id in self.mask:
                offset_idx = int(offset*100)
                tmp = self.mask[row.filename_id][:,offset_idx:]
                f, t = tmp.shape
                mask = np.zeros((f, self.period_idx))
                max_idx = min(t, self.period_idx)
                mask[:,:max_idx] = tmp[:, :max_idx]
            else:
                mask = np.ones((self.CFG.n_mel, self.period_idx))
        else:
            mask = None
        return data, mask

    def load_audio(self,row):
        data, mask = self.get_audio(row)
        if self.train:
            #add train data
            if row.sec_num==0:
                pair_idx = np.random.choice(self.id2record[row.label_id])
                row_pair = self.df.iloc[pair_idx]
                data_pair, mask_pair = self.get_audio(row_pair)
            else:
                duration_seconds = librosa.get_duration(filename=row.audio_paths,sr=None)
                if duration_seconds > self.period:
                    data_pair, mask_pair = self.get_audio(row, offset=random.uniform(0, duration_seconds - self.period))
                else:
                    data_pair, mask_pair = data, mask

            data = np.stack([data, data_pair])
            mask = np.stack([mask, mask_pair])
        
        labels = torch.zeros(self.CFG.CLASS_NUM, dtype=torch.float32) + self.smooth
        if row.sec_num != 0:
            labels[row.labels_id] = self.seclabelp
        if row.label_id != -1:
            labels[row.label_id] = self.prilabelp

        return data, labels, mask
        
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio1, label1, mask1 = self.load_audio(row)
        if self.train:
            if row.label_id in list(self.mixup_idlist.keys()):
                #FMからペアとなるラベルIDを取得
                pair_label_id = np.random.choice(self.mixup_idlist[row.label_id])
                pair_idx = np.random.choice(self.id2record[pair_label_id])
                row2 = self.df.iloc[pair_idx]
                audio2, label2, mask2 = self.load_audio(row)
                audio = np.stack([audio1,audio2])
                label = np.stack([label1,label2])
                mask = np.stack([mask1, mask2])
            else:
                audio = np.stack([audio1,audio1])
                label = np.stack([label1,label1])
                mask = np.stack([mask1, mask1])
        else:
            audio = audio1
            label = label1
        weight = torch.tensor(row.weight, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        if self.train:
            return audio, label, weight, mask
        else:
            return audio, label, weight
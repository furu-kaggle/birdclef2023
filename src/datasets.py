import os,sys,re,glob,random
import pandas as pd
import librosa as lb
import IPython.display as ipd
import soundfile as sf
import numpy as np
import cv2
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

from madgrad import MADGRAD

import audiomentations as AA
from audiomentations import (
    AddGaussianSNR,
)
import timm

class WaveformDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 prilabelp=1.0,
                 seclabelp=1.0,
                 mixup_prob = 0.15,
                 smooth=0.005,
                 train = True
                 ):
      
        self.df = df.reset_index(drop=True)
        self.df["sort_index"] = self.df.index
        self.smooth = smooth
        self.prilabelp = prilabelp - self.smooth
        self.seclabelp = seclabelp - self.smooth
        self.train = train
        
        #mixupprob (データ分析からは0.15~0.20程度が望ましい)
        self.mixup_prob = mixup_prob
        
        #Matrix Factorization (サブラベル同士は相関なしとして扱う)
        self.mfdf = self.df[self.df.sec_num > 0][["label_id","labels_id"]].explode("labels_id").reset_index(drop=True)
        
        #mixupするlabel_idリストを作成する
        self.mixup_idlist = self.mfdf.groupby("label_id").labels_id.apply(list).to_dict()
        
        #label_idリストからレコード番号を取得し、レコード番号からランダムサンプリングする
        self.id2record = self.df.groupby("label_id").sort_index.apply(list)

    def __len__(self):
        return len(self.df)

    def load_img(self,row):
        image = np.load(row.audio_paths)
        image = image.astype("float32", copy=False) / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        
        #15秒以上の場合は端点を外す
        image = image[1:-1] if row.sec > 15 else image
        
        #優先バイアス(録音終わりよりも録音始めの方が鳥の鳴き声が入っている確率が高い 1分以上の場合は開始30秒を学習)
        image = image[:6] if row.sec > 60 else image

        labels = torch.zeros(CFG.CLASS_NUM, dtype=torch.float32) + self.smooth
        labels[row.label_id] = self.prilabelp
        if row.sec_num != 0:
            labels[row.labels_id] = self.seclabelp

        return image, labels
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, label = self.load_img(row)
        if self.train:
            img = img[np.random.choice(len(img),size=2)]
            if (random.uniform(0,1) < self.mixup_prob)&(row.label_id in list(self.mixup_idlist.keys())):
                #FMからペアとなるラベルIDを取得
                pair_label_id = np.random.choice(self.mixup_idlist[row.label_id])
                pair_idx = np.random.choice(self.id2record[pair_label_id])
                row2 = self.df.iloc[pair_idx]
                img2, label2 = self.load_img(row)
                img2 = img2[np.random.choice(len(img2),size=2)]
                img = 0.5*img + 0.5*img2
                label = 0.5*label + 0.5*label2

            img = img.mean(axis=0)
        else:
            #1枚目のみ評価に使用する
            img = img[0]
        weight = row.weight
        return img[None,:,:], label, weight
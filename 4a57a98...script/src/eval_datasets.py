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

class EvalWaveformDataset(Dataset):
    def __init__(self,
                 CFG,
                 df: pd.DataFrame,
                 period = 5):
      
        self.df = df.reset_index(drop=True)


        self.CFG = CFG
        self.sr = CFG.sr
        self.period = period
        
        #Matrix Factorization (サブラベル同士は相関なしとして扱う)
        self.mfdf = self.df[(self.df.sec_num > 0)][["label_id","labels_id"]].explode("labels_id").reset_index(drop=True)
        
        #mixupするlabel_idリストを作成する
        self.mixup_idlist = self.mfdf.groupby("label_id").labels_id.apply(list).to_dict()
        
        #mixupする先はシングルラベルにする
        sdf = self.df[(self.df.sec_num==0)|(self.df.primary_label=="lotcor1")]
        
        #label_idリストからレコード番号を取得し、レコード番号からランダムサンプリングする
        self.id2record = sdf.groupby("label_id").sort_index.apply(list)


        #df postprocess
        self.evaldf = df.copy()

        self.evaldf["start_sec"] = self.evaldf.sec.apply(lambda x: [s for s in range(0, max(1,int(x)-1), 5)])
        self.evaldf = self.evaldf.explode("start_sec").reset_index(drop=True)
        self.evaldf["unique_id"] = self.evaldf["filename_id"] + "_" + self.evaldf["start_sec"].astype(str)
        self.evaldf = self.evaldf[self.evaldf.start_sec < 30].reset_index(drop=True)

    def crop_or_pad(self, y, length, is_train=False, start=None):
        if len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])

        elif len(y) > length:
            if not is_train:
                start = start or 0
            else:
                start = start or np.random.randint(len(y) - length)

            y = y[start:start + length]

        return y
        
        
    def __len__(self):
        return len(self.evaldf)

    def load_audio(self,row):
        #データ読み込み
        data, sr = librosa.load(row.audio_paths, sr=self.sr, offset=row.start_sec, duration=self.period, mono=True)

        #test datasetの最大長
        max_sec = len(data)//sr 
        #0秒の場合は１秒として取り扱う
        max_sec = 1 if max_sec==0 else max_sec
        
        data = self.crop_or_pad(data , length=sr*self.period,is_train=False)
        
        prim_labels = torch.zeros(self.CFG.CLASS_NUM, dtype=torch.float32)
        sec_labels = torch.zeros(self.CFG.CLASS_NUM, dtype=torch.float32)

        prim_labels[row.label_id] = 1.0
        sec_labels[row.label_id] = 1.0
        if row.sec_num != 0:
            sec_labels[row.labels_id] = 1.0

        return data, prim_labels, sec_labels
    
    def __getitem__(self, idx):
        row = self.evaldf.iloc[idx]
        audio, prim_labels, sec_labels = self.load_audio(row)
        audio = torch.tensor(audio, dtype=torch.float32)
        return audio, prim_labels, sec_labels, row.sec_num, row.unique_id
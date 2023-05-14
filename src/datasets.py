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
import concurrent.futures
from numpy.lib.stride_tricks import sliding_window_view
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
                 train = True
                 ):
      
        self.df = df.reset_index(drop=True)
        self.df["mixup_weight"] = self.df["sample_weight"]/self.df["sample_weight"].sum()
        #self.label_weight = self.df.set_index("label_id").["sample_weight"]
        self.CFG = CFG
        self.aug = train_aug
        self.sr = CFG.sr
        self.period = period
        self.df["sort_index"] = self.df.index
        self.smooth = smooth
        self.prilabelp = prilabelp - smooth
        self.seclabelp = seclabelp - smooth
        self.train = train

        # cand_df = self.df[["filename_id","latitude","longitude"]].dropna().drop_duplicates()

        # candpair_df = cand_df.merge(cand_df,on=["latitude","longitude"],suffixes=("","_cand"))
        # cand_df = cand_df[cand_df.filename_id!=cand_df.filename_id_cand].reset_index(drop=True)
        # self.cand_dict = cand_df.groupby("filename_id").filename_id_cand.apply(list).to_dict()
        

        
        #Matrix Factorization (サブラベル同士は相関なしとして扱う)
        self.mfdf = self.df[(self.df.sec_num > 0)][["label_id","labels_id"]].explode("labels_id").reset_index(drop=True)
        
        #mixupするlabel_idリストを作成する
        self.mixup_idlist = self.mfdf.groupby("label_id").labels_id.apply(list).to_dict()
        
        #mixupする先はシングルラベルにする
        sdf = self.df[(self.df.sec_num==0)|(self.df.primary_label=="lotcor1")]
        
        #label_idリストからレコード番号を取得し、レコード番号からランダムサンプリングする
        self.id2record = sdf.groupby("label_id").sort_index.apply(list)

        self.cache = {}
        self.max_sec = max(self.CFG.factors)*5.0

        boxdf = pd.read_csv("data/box.csv",index_col=0).reset_index().rename(columns={"index":"unique_id"})
        boxdf["filename_id"] = boxdf["unique_id"].apply(lambda x: x.split("_")[0])
        boxdf["time_id"] = boxdf["unique_id"].apply(lambda x: x.split("_")[1])
        boxdf["box_id"] = boxdf["unique_id"].apply(lambda x: x.split("_")[2])
        boxdf["start_x"] = boxdf["time_id"].astype(int)*500
        boxdf["x1"] = boxdf["x1"] + boxdf["start_x"]
        boxdf["y1"] = boxdf["y1"]
        boxdf["x2"] = boxdf["x2"] + boxdf["start_x"]
        boxdf["y2"] = boxdf["y2"]

        # Group the DataFrame by 'filename_id'
        grouped = boxdf.groupby('filename_id')

        # Create a dictionary where the keys are the filename_ids and the values are the corresponding sub-dataframes (gdf)
        self.gdf_dict = {name: group for name, group in grouped}
        
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

    def get_offset(self, row):
        #準備
        if row.filename_id in self.gdf_dict:
            gdf = self.gdf_dict[row.filename_id]
            mask = np.zeros((128, gdf.start_x.max() + 500),dtype=np.float64)
            for jdx, brow in gdf.iterrows():
                mask[int(brow.y1):int(brow.y2),int(brow.x1):int(brow.x2)] += brow.conf
            if mask.shape[1] > self.period*100:
                sampling_weights = sliding_window_view(mask.max(axis=0), self.period*100).sum(axis=1)[::100]
                sample_prob = sampling_weights/sampling_weights.sum()
            else:
                sample_prob = None
            mask_freq = np.argwhere(mask.sum(axis=1)==0)[:,0]
            if len(mask_freq) > 40:
                mask_freq = []
        else:
            sample_prob = None
            mask_freq = []

        #periodより長いか判定する
        duration_seconds = librosa.get_duration(filename=row.audio_paths,sr=None)
        if duration_seconds < self.period:
            #offsetを変えても情報が全て入っているので、0として扱う
            offset = 0
        else:
            if sample_prob is not None:
                #offsetを変えてシグナルから始まるようにする
                offset = np.random.choice(len(sample_prob),p=sample_prob)
            else:
                offset = random.uniform(0, duration_seconds - self.period)
        
        mask_freq_array = np.ones((self.CFG.n_mel, self.period*100),dtype=bool)
        mask_freq_array[mask_freq] = 0

        return offset, mask_freq_array

    def load_audio(self, row, offset):
        data, sr = librosa.load(row.audio_paths, sr=self.sr, offset=offset, duration=self.period, mono=True)
        return data

    def preprocess_audio(self, data, row):
        #augemnt1
        if (self.train)&(random.uniform(0,1) < row.weight):
             data = self.aug(samples=data, sample_rate=self.sr)

        #test datasetの最大長
        max_sec = len(data)//self.sr

        #0秒の場合は１秒として取り扱う
        max_sec = 1 if max_sec==0 else max_sec
        
        data = self.crop_or_pad(data , length=self.sr*self.period, is_train=self.train)

        return data

    def __len__(self):
        return len(self.df)

    def get_audio(self, row):
        offset, freqmask = self.get_offset(row)
        data = self.load_audio(row, offset)
        data = self.preprocess_audio(data, row)

        if self.train:
            #add train data
            if row.sec_num==0:
                pair_idx = np.random.choice(self.id2record[row.label_id])
                row_pair = self.df.iloc[pair_idx]
                offset, freqmask_pair = self.get_offset(row_pair)
                data_pair = self.load_audio(row_pair, offset)
                data_pair = self.preprocess_audio(data_pair, row_pair)

            else:
                offset, freqmask_pair = self.get_offset(row)
                data_pair = self.load_audio(row, offset)
                data_pair = self.preprocess_audio(data_pair, row)

            data = np.stack([data, data_pair])
            freqmask = np.stack([freqmask, freqmask_pair])
        
        labels = torch.zeros(self.CFG.CLASS_NUM, dtype=torch.float32) + self.smooth
        if row.sec_num != 0:
            labels[row.labels_id] = self.seclabelp
        if row.label_id != -1:
            labels[row.label_id] = self.prilabelp
        

        return data, labels, freqmask
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio1, label1, freqmask1 = self.get_audio(row)
        if self.train:
            # if (self.CFG.mixup_fm)&(row.filename_id in self.cand_dict):
            #     #FMからペアとなるラベルIDを取得
            #     filename_id = np.random.choice(self.cand_dict[row.filename_id])
            #     row2 = self.df[self.df.filename_id == filename_id].iloc[0]
            # else:
            pair_idx = np.random.choice(len(self.df), p=self.df["mixup_weight"].values)
            row2 = self.df.iloc[pair_idx]
            audio2, label2, freqmask2 = self.get_audio(row2)
            audio = np.stack([audio1,audio2])
            label = np.stack([label1,label2])
            freqmask = np.stack([freqmask1,freqmask2])
        else:
            audio = audio1
            label = label1
        weight = torch.tensor(row.weight, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        return audio, label, weight, freqmask
import os,sys,re,glob,random

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd
import librosa as lb
import IPython.display as ipd
import soundfile as sf
import numpy as np
import ast, joblib
from pathlib import Path
import itertools

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
from audiomentations import (
    AddGaussianSNR,
)
import timm

from src import Trainer, Model, WaveformDataset, CFG

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
def birds_stratified_split(df, target_col, test_size=0.2):
    class_counts = df[target_col].value_counts()
    low_count_classes = class_counts[class_counts < 10].index.tolist() ### Birds with single counts

    df['train'] = df[target_col].isin(low_count_classes)

    train_df, val_df = train_test_split(df[~df['train']], test_size=test_size, stratify=df[~df['train']][target_col], random_state=42)

    train_df = pd.concat([train_df, df[df['train']]], axis=0).reset_index(drop=True)

    # Remove the 'valid' column
    train_df.drop('train', axis=1, inplace=True)
    val_df.drop('train', axis=1, inplace=True)

    return train_df, val_df

device = torch.device("cuda")
def run(foldtrain=False):
    model = Model(CFG, path = CFG.pretrainpath).to(device)
    
    if foldtrain:
        train, test = birds_stratified_split(df, 'primary_label', 0.1)

        valid_set = WaveformDataset(
            CFG = CFG,
            df=test,
            smooth=0,
            mixup_prob=0,
            seclabelp=1.0,
            train = False,
            period = 5
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=CFG.batch_size//2,
            pin_memory=True,
            shuffle = False,
            drop_last=True,
            num_workers=CFG.workers,
        )

    else:
        train = df
        
    optimizer = CFG.get_optimizer(
        model, 
        CFG.lr, 
        CFG.lr_ratio
    )
    scheduler = CFG.get_scheduler(
        optimizer,
        epochs=CFG.epochs,
        min_lr=CFG.min_lr,
        warmupstep=CFG.warmupstep
    )
    
    trainer = Trainer(
        CFG = CFG,
        model=model,
        optimizer=optimizer,
        scheduler = scheduler,
        device=device,
    )
    #trainer.valid_one_cycle(valid_loader, 0)
    for epoch in range(CFG.epochs):
        model.factor = CFG.factors[epoch]
        train_set = WaveformDataset(
             CFG = CFG,
             df=train,
             smooth=CFG.smooth,
             period = int(5 * CFG.factors[epoch])
         )
        batch_factor = 1
        train_loader = DataLoader(
            train_set,
            batch_size=CFG.batch_size*batch_factor,
            drop_last=True,
            pin_memory=True,
            shuffle = True,
            num_workers=CFG.workers *batch_factor,
        )
        print(f"{'-'*35} EPOCH: {epoch}/{CFG.epochs} {'-'*35}")
        trainer.train_one_cycle(train_loader,epoch)
        if foldtrain:
            trainer.valid_one_cycle(valid_loader,epoch)
        else:
            #last save model
            savename = CFG.weight_dir + f"model_{CFG.key}_last.bin"
            torch.save(trainer.model.state_dict(),savename)
            if (epoch > 25):
                try:
                    savename = CFG.weight_dir + f"model_{epoch}.bin"
                    torch.save(trainer.model.state_dict(),savename)
                except:
                    pass
                    
pretrain_label = pd.concat([
    #pd.read_csv("data/train2020_noduplicates.csv",index_col=0),
    pd.read_csv("data/train2021_noduplicates.csv",index_col=0),
    pd.read_csv("data/train2022_noduplicates.csv",index_col=0),
]).dropna(subset=["primary_label"])
pretrain_label["secondary_labels"] = pretrain_label["secondary_labels"].apply(eval)
pretrain_label = pretrain_label.groupby("filename_id").agg(
    primary_labels = ("primary_label","unique"),
    secondary_labels = ("secondary_labels", list)
).reset_index()
pretrain_label["secondary_labels"] = pretrain_label.secondary_labels.apply(
    lambda x: list(
        set(
            list(
                itertools.chain.from_iterable(x)
            )
        )
    )
)

#train2020 = pd.read_csv("data/train2020_noduplicates.csv",index_col=0)
train2021 = pd.read_csv("data/train2021_noduplicates.csv",index_col=0)
train2022 = pd.read_csv("data/train2022_noduplicates.csv",index_col=0)
 
pdf = pd.DataFrame(glob.glob("data/bird**/train_*audio/**/*.ogg"),columns=["audio_paths"])
pdf["filename_id"] = pdf.audio_paths.apply(lambda x: x.split("/")[-1].split(".")[0])
train2021 = pd.merge(train2021,pdf,on=["filename_id"])
train2022 = pd.merge(train2022,pdf,on=["filename_id"])

#print(len(train2021),len(train2022))

#pdf = pd.DataFrame(glob.glob("data/bird**/train_*audio/**/*.mp3"),columns=["audio_paths"])
#pdf["filename_id"] = pdf.audio_paths.apply(lambda x: x.split("/")[-1].split(".")[0])
#train2020 = pd.merge(train2020,pdf,on=["filename_id"])

#print(len(train2020))
#print(len(train2021)+len(train2022)+len(train2020))

df = pd.concat([train2021,train2022]).reset_index(drop=True)
df = pd.merge(df.drop(["secondary_labels"],axis=1),pretrain_label,on=["filename_id"])

ex_ids = pd.read_csv("data/train.csv").filename_id.values
df = df[~df.filename_id.isin(ex_ids)].reset_index(drop=True)
print(len(df))

sec_unique = set(df.explode("secondary_labels").dropna(subset=["secondary_labels"]).secondary_labels.unique())
pri_unique = set(df.explode("primary_labels").dropna(subset=["primary_labels"]).primary_labels.unique())
unique_key = list(pri_unique|sec_unique)
print(f"unique_len:{len(unique_key)}")

label2id = {label: label_id for label_id, label in enumerate(sorted(unique_key))}
id2label = {val: key for key,val in label2id.items()}

df.loc[:,"label_id"] = df.loc[:,"primary_label"].map(label2id)
df.loc[:,"label_ids"] = df.loc[:,"primary_labels"].apply(lambda x: list(np.vectorize(
    lambda s: label2id[s])(x)) if len(x)!=0 else -1)
df.loc[:,"labels_id"] = df.loc[:,"secondary_labels"].apply(lambda x: list(np.vectorize(
    lambda s: label2id[s])(x)) if len(x)!=0 else -1)

df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

print(df)

#ユニークキー
CFG.unique_key = unique_key

#クラス数
CFG.CLASS_NUM = len(unique_key)

# CFG.key = "eval"
# CFG.epochs = 3
# CFG.factors = [3,3,3]
# run(foldtrain=True)

CFG.key = "all"
CFG.epochs = 30
run(foldtrain=False)
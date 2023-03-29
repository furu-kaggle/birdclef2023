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

device = torch.device("cuda")
def run(foldtrain=False):
    model = Model(CFG, path = CFG.pretrainpath).to(device)
    
    if foldtrain:
        train = df[~df["eval"].astype(bool)].reset_index(drop=True)
        test =  df[df["eval"].astype(bool)].reset_index(drop=True)

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
    

    train_set = WaveformDataset(
        CFG = CFG,
        df=train,
        smooth=CFG.smooth,
        period = CFG.period
    )
    train_loader = DataLoader(
          train_set,
          batch_size=CFG.batch_size,
          drop_last=True,
          pin_memory=True,
          shuffle = True,
          num_workers=CFG.workers,
    )
    
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
        print(f"{'-'*35} EPOCH: {epoch}/{CFG.epochs} {'-'*35}")
        trainer.train_one_cycle(train_loader,epoch)
        if foldtrain:
            trainer.valid_one_cycle(valid_loader,epoch)
        else:
            #last save model
            savename = CFG.weight_dir + f"model_{CFG.key}_last.bin"
            torch.save(trainer.model.state_dict(),savename)

df = pd.read_csv("data/train.csv")
df["labels_id"] = df.labels_id.apply(eval)

submission = pd.read_csv("data/sample_submission.csv")
unique_key = list(submission.columns[1:])

label2id = {label: label_id for label_id, label in enumerate(sorted(unique_key))}
id2label = {val: key for key,val in label2id.items()}

# #connect path
pathdf = pd.DataFrame(glob.glob("data/train_audio/**/*.ogg"),columns=["audio_paths"])
pathdf["filename_sec"] = pathdf.audio_paths.apply(lambda x: x.split("/")[-1].replace(".ogg",""))
pathdf["filename_id"] =pathdf["filename_sec"].apply(lambda x: x.split("_")[0])
df = pd.merge(df,pathdf[["filename_id","audio_paths"]],on=["filename_id"]).reset_index(drop=True)

addtrain = pd.read_csv("pretrain_src/add_train.csv",index_col=0).dropna(subset=["primary_label"])
addtrain["secondary_labels"] = addtrain["secondary_labels"].apply(eval)
addtrain.loc[:,"label_id"] = addtrain.loc[:,"primary_label"].map(label2id).fillna(-1).astype(int)
addtrain.loc[:,"labels_id"] = addtrain.loc[:,"secondary_labels"].apply(lambda x: np.vectorize(
    lambda s: label2id[s] if s in unique_key else -1)(x) if len(x)!=0 else np.array([-1]))
addtrain.loc[:,"labels_id"] = addtrain.loc[:,"labels_id"].apply(lambda x: list(x[x != -1]))
addtrain["sec_num"] = addtrain.loc[:,"labels_id"].apply(len)
addtrain["eval"] = 0

pathdf = pd.DataFrame(glob.glob("/home/furugori/train*/train*/**/XC*.*"),columns=["audio_paths"])
pathdf["filename_sec"] = pathdf.audio_paths.apply(lambda x: x.split("/")[-1].replace(".mp3","").replace(".ogg",""))
pathdf["filename_id"] =pathdf["filename_sec"].apply(lambda x: x.split("_")[0])
addtrain = pd.merge(addtrain,pathdf[["filename_id","audio_paths"]].drop_duplicates("filename_id"),on=["filename_id"]).reset_index(drop=True)

print(addtrain[["primary_label","secondary_labels","label_id","labels_id","audio_paths"]])

df = pd.concat([df,addtrain]).reset_index(drop=True)

df["weight"] = df["rating"] / df["rating"].max()

#ユニークキー
CFG.unique_key = unique_key

#クラス数
CFG.CLASS_NUM = len(unique_key)

CFG.key = "eval"
run(foldtrain=True)

CFG.key = "all"
run(foldtrain=False)
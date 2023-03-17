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

from madgrad import MADGRAD

import audiomentations as AA
from audiomentations import (
    AddGaussianSNR,
)
import timm

from src import Trainer, Model, WaveformDataset

device = torch.device("cuda")

def run(foldtrain=False):
    model = Model(CFG, path = CFG.pretrainpath).to(device)
    
    if foldtrain:
        train = df[df.fold != CFG.fold].reset_index(drop=True)
        test =  df[df.fold == CFG.fold].reset_index(drop=True)

        valid_set = WaveformDataset(
            CFG = CFG,
            df=test,
            smooth=0,
            mixup_prob=0,
            seclabelp=1.0,
            train = False
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=CFG.batch_size,
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
        smooth=CFG.smooth
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

df = pd.read_csv("data/train_metadata.csv")
df["secondary_labels"] = df.secondary_labels.apply(eval)
df["sec_num"] = df["secondary_labels"].apply(len)
df["filename_id"] = df["filename"].apply(lambda x: x.split("/")[-1].replace(".ogg",""))
df["sort_index"] = df.index

submission = pd.read_csv("data/sample_submission.csv")

unique_key = list(submission.columns[1:])
sec_unique_key = df.explode("secondary_labels").dropna(subset=["secondary_labels"])["secondary_labels"].unique()
#non-sympathetic bird
#ただサンプル数が少ないケースもある
nonsympathetic_key =  set(unique_key) - set(sec_unique_key) 

label2id = {label: label_id for label_id, label in enumerate(sorted(unique_key))}
id2label = {val: key for key,val in label2id.items()}

df.loc[:,"label_id"] = df.loc[:,"primary_label"].map(label2id)
df.loc[:,"labels_id"] = df.loc[:,"secondary_labels"].apply(lambda x: np.vectorize(
    lambda s: label2id[s])(x) if len(x)!=0 else -1)

# #connect path
pathdf = pd.DataFrame(glob.glob("data/train_audio/**/*.ogg"),columns=["audio_paths"])
pathdf["filename_sec"] = pathdf.audio_paths.apply(lambda x: x.split("/")[-1].replace(".ogg","").replace(".npy",""))
pathdf["filename_id"] =pathdf["filename_sec"].apply(lambda x: x.split("_")[0])
df = pd.merge(df,pathdf[["filename_id","audio_paths"]],on=["filename_id"]).reset_index(drop=True)
df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

#Matrix Factorization (サブラベル同士は相関なしとして扱う)
mfdf = df[df.sec_num > 0][["label_id","labels_id"]].explode("labels_id").reset_index(drop=True)

#50%以上1なので、一旦一様分布に近似して問題なさそう....？
mfdf["prob"] = 1
probdf = mfdf.groupby(["label_id","labels_id"]).prob.count().reset_index()

from sklearn.model_selection import KFold, StratifiedKFold
RANDOM_STATE = 35
skfold = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
splits= skfold.split(df ,y=df["author"])

for i,(_, test_index) in enumerate(splits):
    df.loc[test_index,"fold"] = i


class CFG:
    #image parameter
    sr = 32000
    period = 5
    n_mel = 128
    fmin = 20
    fmax = 16000
    power = 2
    top_db = 80

    time_len = 281

    # time_len = sr[1/s] * time[s] /hop_len = sr[1/s] * time[s] 4/n_fft 
    n_fft = int(sr * period * 4/time_len)

    hop_len = n_fft//4

    #imgsize
    imagesize = (n_mel, time_len)
    
    #クラス数
    CLASS_NUM = len(unique_key)

    #ユニークキー
    unique_key = unique_key
    
    #バッチサイズ
    batch_size = 16

    #前処理CPUコア数
    workers = 8

    #学習率 (best range 5e-9~2e-4)
    lr = 1e-3

    #スケジューラーの最小学習率
    min_lr = 1e-6

    #ウォームアップステップ
    warmupstep = 0

    #エポック数
    epochs = 20

    #lr ratio (best fit 3)
    lr_ratio = 3

    #label smoothing rate
    smooth = 0.005

    #model name
    model_name = 'eca_nfnet_l0'

    #pretrain model path
    pretrainpath = "pretrain_weight/eca_nfnet_l0_pretrainmodel_70k_p7.bin"

    #重みを保存するディレクトリ
    weight_dir = "weight/eca_nfnet_l0/"

    #テストfold
    fold = 0

    def get_optimizer(model, learning_rate, ratio, decay=0):
        return  MADGRAD(params=[
            {"params": model.model.parameters(), "lr": learning_rate/ratio},
            {"params": model.fc.parameters(),    "lr": learning_rate},
        ],weight_decay=decay)

    def get_scheduler(optimizer,min_lr, epochs, warmupstep=0,warmup_lr_init=5e-5):
        # base lr は optimizerを継承
        # document:https://timm.fast.ai/SGDR
        return CosineLRScheduler(
            optimizer, 
            t_initial=epochs, 
            lr_min=min_lr, 
            warmup_t=warmupstep, 
            warmup_lr_init=warmup_lr_init, 
            warmup_prefix=True
        )

for fold in [0]:
    CFG.fold = fold
    CFG.key = fold
    run(foldtrain=True)
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

from pretrainsrc import Trainer, Model, WaveformDataset, CFG, EvalWaveformDataset

device = torch.device("cuda")
def run(foldtrain=False):
    model = Model(CFG, path = CFG.pretrainpath).to(device)
    
    if foldtrain:
        train = df[~df["eval"].astype(bool)].reset_index(drop=True)
        test =  df[df["eval"].astype(bool)].reset_index(drop=True)

        valid_set = EvalWaveformDataset(
            CFG = CFG,
            df=test,
            period = 5
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
        device=device
    )
    primary_label_counts_map = train["label_id"].value_counts().to_dict()
    secondary_label_counts_map = train["labels_id"].explode().value_counts().to_dict()
    train["primary_count"] = train["label_id"].map(primary_label_counts_map)
    train["secondary_count"] = train["label_id"].map(secondary_label_counts_map).fillna(0)
    train["label_count"] = train["primary_count"] + train["secondary_count"]
    train["sample_weight"] = train["label_count"]**(1/4)/train["label_count"]
    #trainer.valid_one_cycle(valid_loader, 0)
    for epoch in range(CFG.epochs):
        # downsample_train = pd.concat([
        #         train[train['label_id'] == label].sample(min(CFG.sample_size, count), random_state=epoch, replace=False)
        #                         for label, count in train['label_id'].value_counts().items()             
        # ]).reset_index(drop=True)
        print("set sampler")
        train_sampler = torch.utils.data.WeightedRandomSampler(
            list(train["sample_weight"].values),
            len(train),
            replacement=True
        )
        model.factor = CFG.factors[epoch]
        train_set = WaveformDataset(
             CFG = CFG,
             df=train,
             prilabelp = CFG.prilabelp,
             seclabelp = CFG.seclabelp,
             smooth=CFG.smooth,
             period = int(5 * CFG.factors[epoch])
         )
        batch_factor = min(2, int(max(CFG.factors)/CFG.factors[epoch]))
        train_loader = DataLoader(
            train_set,
            batch_size=CFG.batch_size*batch_factor,
            drop_last=True,
            pin_memory=True,
            #shuffle = True,
            num_workers=CFG.workers*batch_factor,
            sampler = train_sampler
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
                    

train = pd.read_csv("data/pretrain.csv",index_col=0)
pdf = pd.DataFrame(glob.glob("data/birdclef202*/pretrain_spec*/kaggle/audio_images/*.pt"),columns=["apath"])
pdf["filename_id"] = pdf.apath.apply(lambda x: x.split("/")[-1].split(".")[0])
df = pd.merge(train,pdf,on=["filename_id"])
df = pd.merge(df.drop(["primary_labels","secondary_labels"],axis=1),pretrain_label,on=["filename_id"])

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

#ユニークキー
CFG.unique_key = unique_key

#クラス数
CFG.CLASS_NUM = len(unique_key)

CFG.id2label = id2label

CFG.key = "eval"
run(foldtrain=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

set_seed(35)
CFG.key = "all_35"
run(foldtrain=False)

set_seed(355)
CFG.key = "all_355"
run(foldtrain=False)

set_seed(311)
CFG.key = "all_311"
run(foldtrain=False)
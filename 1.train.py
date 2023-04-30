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

from src import Trainer, Model, WaveformDataset, CFG, EvalWaveformDataset



device = torch.device("cuda")
def run():
    model = Model(CFG, path = CFG.pretrainpath).to(device)

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
        warmupstep=CFG.warmupstep,
        k_decay=CFG.k_decay
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

    for epoch in range(CFG.epochs):
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
        train_loader = DataLoader(
            train_set,
            batch_size=CFG.batch_size,
            drop_last=True,
            pin_memory=True,
            shuffle = False,
            num_workers=CFG.workers,
            sampler = train_sampler
        )
        print(f"{'-'*35} EPOCH: {epoch}/{CFG.epochs} {'-'*35}")
        trainer.train_one_cycle(train_loader,epoch)
        #last save model
        savename = CFG.weight_dir + f"model_{CFG.key}_last.bin"
        torch.save(trainer.model.state_dict(),savename)
        if (epoch > 25):
            try:
                savename = CFG.weight_dir + f"model_{epoch}.bin"
                torch.save(trainer.model.state_dict(),savename)
            except:
                pass
                    

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

addtrain = pd.read_csv("data/add_train.csv",index_col=0).dropna(subset=["primary_label"])
addtrain["secondary_labels"] = addtrain["secondary_labels"].apply(eval)
addtrain.loc[:,"label_id"] = addtrain.loc[:,"primary_label"].map(label2id).fillna(-1).astype(int)
addtrain.loc[:,"labels_id"] = addtrain.loc[:,"secondary_labels"].apply(lambda x: np.vectorize(
    lambda s: label2id[s] if s in unique_key else -1)(x) if len(x)!=0 else np.array([-1]))
addtrain.loc[:,"labels_id"] = addtrain.loc[:,"labels_id"].apply(lambda x: list(x[x != -1]))
addtrain["sec_num"] = addtrain.loc[:,"labels_id"].apply(len)
addtrain["eval"] = 0

pathdf = pd.DataFrame(glob.glob("data/addtrain_audio/XC*.*"),columns=["audio_paths"])
pathdf["filename_sec"] = pathdf.audio_paths.apply(lambda x: x.split("/")[-1].replace(".mp3","").replace(".ogg",""))
pathdf["filename_id"] =pathdf["filename_sec"].apply(lambda x: x.split("_")[0])
addtrain = pd.merge(addtrain,pathdf[["filename_id","audio_paths"]].drop_duplicates("filename_id"),on=["filename_id"]).reset_index(drop=True)
#addtrain["start_sec"] = 0

print(addtrain[["primary_label","secondary_labels","label_id","labels_id","audio_paths"]])

df = pd.concat([df,addtrain]).reset_index(drop=True)

print(df)

df["weight"] = df["rating"] / df["rating"].max()

#df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

loopaugdf = pd.read_csv("data/loopaugdf.csv")
df = pd.merge(df, loopaugdf, on=["filename_id"], how="left").fillna(False)

#ユニークキー
CFG.unique_key = unique_key

#クラス数
CFG.CLASS_NUM = len(unique_key)

CFG.id2label = id2label

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
run()

set_seed(355)
CFG.key = "all_355"
run()

set_seed(311)
CFG.key = "all_311"
run()
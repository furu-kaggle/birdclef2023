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

from madgrad import MADGRAD

import audiomentations as AA
from audiomentations import (
    AddGaussianSNR,
)
import timm

from .Record import Record

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, CFG):
        """
        Constructor for Trainer class
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.device = device
        self.CFG = CFG
    
    def train_one_cycle(self, train_loader, epoch):
        """
        Runs one epoch of training, backpropagation and optimization
        """
        self.model.train() 
        self.model.training = True

        total_loss = 0
        total_nums = 0
        record = Record(self.CFG)

        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        for idx,(data, label, weight) in pbar:
            data =  data.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.float)
            weight = weight.to(self.device, dtype=torch.float)
            
            self.optimizer.zero_grad()
            pred, loss = self.model(data, label, weight)       
            record.update(pred, label)
            total_loss += (loss.detach().item() * label.size(0))
            total_nums += label.size(0)
            
            loss.backward()
            self.optimizer.step()

            pbar.set_description("[loss %f, lr %e last_lr %e]" % (total_loss / total_nums, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))

            if  self.scheduler is not None:
                self.scheduler.step(epoch + idx/len(train_loader))
            
        self.model.eval()
        print(f"epoch:{epoch} train result, loss:{total_loss / total_nums}", file=codecs.open(self.CFG.weight_dir + 'logging.txt', 'a', 'utf-8'))

    def valid_one_cycle(self, valid_loader, epoch):
        """
        Runs one epoch of prediction
        """
        self.model.eval()
        self.model.training = False
        total_loss = 0
        total_nums = 0
        record = Record(self.CFG)
        
        #get feature vectors array
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader))
        with torch.no_grad():
            for idx, (xval, yval, weight) in pbar:
                xval = xval.to(self.device, dtype=torch.float)
                yval = yval.to(self.device, dtype=torch.float)
                weight = weight.to(self.device, dtype=torch.float)

                pred, loss = self.model(xval, yval, weight)
                record.update(pred, yval)
                total_loss += (loss.mean().detach().item() * yval.size(0))
                total_nums += yval.size(0)
                pbar.set_description("[loss %f]" % (total_loss / total_nums))

        cmAP = record.get_f1score()
        print(cmAP)

        print(f"epoch:{epoch} val result, get_cmAP:{cmAP:.3f}, loss:{total_loss / total_nums}", file=codecs.open(self.CFG.weight_dir + 'logging.txt', 'a', 'utf-8'))
        savename = self.CFG.weight_dir + f"model_{cmAP:.3f}_{self.CFG.key}.bin"
        torch.save(self.model.state_dict(),savename)
        for path in sorted(glob.glob(self.CFG.weight_dir + f"model_*_{self.CFG.key}.bin"), reverse=True)[1:]:
            os.remove(path)
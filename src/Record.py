import os,sys,re,glob,random
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from tqdm import tqdm
import sklearn.metrics

class Record:
    '''
    Records labels and predictions within one epoch
    '''
    def __init__(self, CFG):
        self.labels = []
        self.preds = []
        self.unique_ids = []
        self.f1score = 0
        self.unique_key = CFG.unique_key
        
        #padding array
        self.pad_rows = np.ones((1,len(self.unique_key)))
        
    def update(self, cur_logits, cur_labels):
        cur_labels = cur_labels.clone().detach().cpu().numpy()
        cur_preds = cur_logits.sigmoid().clone().detach().cpu().numpy()
        self.labels.append(cur_labels)
        self.preds.append(cur_preds)

    def eval_update(self, cur_logits, cur_labels, unique_id):
        cur_labels = cur_labels.clone().detach().cpu().numpy()
        cur_preds = cur_logits.sigmoid().clone().detach().cpu().numpy()
        self.labels.append(cur_labels)
        self.preds.append(cur_preds)
        self.unique_ids.append(unique_id)

    def get_f1score(self):
        labels = np.concatenate(self.labels,axis=0).astype(float)
        preds = np.concatenate(self.preds,axis=0).astype(float)
        for _ in range(5):
            labels = np.append(labels, self.pad_rows, axis=0)
            preds  = np.append(preds,  self.pad_rows, axis=0)
        score = sklearn.metrics.average_precision_score(
            labels, preds, average='macro'
        )
        return score

    def get_validation(self):
        labels = np.concatenate(self.labels,axis=0).astype(float)
        preds = np.concatenate(self.preds,axis=0).astype(float)
        for _ in range(5):
            labels = np.append(labels, self.pad_rows, axis=0)
            preds  = np.append(preds,  self.pad_rows, axis=0)
        score_array = sklearn.metrics.average_precision_score(
            labels, preds, average=None
        )
        print(f"{score_array.mean()}")
        score_df = pd.DataFrame(score_array, index=self.unique_key, columns=["ac_score"])
        return score_df
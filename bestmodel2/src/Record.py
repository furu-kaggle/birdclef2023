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
        self.f1score = 0
        
        self.ptrues = []
        self.strues = []
        self.uids = []
        self.secnum = []
        self.plosses = 0
        self.slosses = 0
        self.total = 0
        self.sample_size = 8400
        
        #padding array
        self.pad_rows = np.ones((1,len(CFG.unique_key)))
        self.id2label = CFG.id2label
        self.unique_key = CFG.unique_key
        self.weight_dir = CFG.weight_dir
        
    def update(self, cur_logits, cur_labels):
        cur_labels = cur_labels.detach().cpu().numpy()
        cur_preds = cur_logits.sigmoid().detach().cpu().numpy()
        self.labels.append(cur_labels)
        self.preds.append(cur_preds)
        
    def eval_update(self, logits, ptrues, strues, plosses, slosses, uids,sec_num):
        self.plosses += plosses.detach().cpu().item() * ptrues.size(0)
        self.slosses += slosses.detach().cpu().item() * strues.size(0)
        self.total += ptrues.size(0) 
        plosses = plosses.detach().cpu()
        ptrues = ptrues.detach().cpu().numpy()
        strues = strues.detach().cpu().numpy()
        logits = logits.sigmoid().detach().cpu().numpy()
        self.ptrues.append(ptrues)
        self.strues.append(strues)
        self.preds.append(logits)
        self.uids.append(uids)
        self.secnum.append(sec_num)
    
    def get_loss(self):
        return self.plosses/self.total, self.slosses/self.total

    def get_score(self, preds, target, random_state, type="primary"):
        for _ in range(5):
            target = np.append(target, self.pad_rows, axis=0)
            preds  = np.append(preds,  self.pad_rows, axis=0)

        score = sklearn.metrics.average_precision_score(
            target, preds, average=None
        )
        score_df = pd.DataFrame(score, index=self.unique_key, columns=[f"{type}_score"])
        score_df.to_csv(f"{self.weight_dir}{type}_score_{random_state}.csv")
        return score.mean()

    def get_valdf(self):
        ptrues = np.concatenate(self.ptrues,axis=0).astype(float)
        strues = np.concatenate(self.strues,axis=0).astype(float)
        preds = np.concatenate(self.preds,axis=0).astype(float)
        uids = np.concatenate(self.uids,axis=0)
        secnum = np.concatenate(self.secnum,axis=0)
        
        ptdf = {}
        stdf = {}
        pdf = {}
        secdf = {}
        for uid, pr, st, pt, sc in zip(uids, preds, strues, ptrues,secnum): 
            ptdf[uid] = pt
            stdf[uid] = st
            pdf[uid] = pr
            secdf[uid] = sc
        
        pdf = pd.DataFrame(pdf).T.rename(columns=self.id2label).reset_index().rename(columns={"index":"row_id"})
        pdf["pred"] = pdf[self.unique_key].apply(lambda x: np.array(x),axis=1)
        
        stdf = pd.DataFrame(stdf).T.rename(columns=self.id2label).reset_index().rename(columns={"index":"row_id"})
        stdf["secondary_true"] = stdf[self.unique_key].apply(lambda x: np.array(x),axis=1)
        
        ptdf = pd.DataFrame(ptdf).T.rename(columns=self.id2label).reset_index().rename(columns={"index":"row_id"})
        ptdf["primary_true"] = ptdf[self.unique_key].apply(lambda x: np.array(x),axis=1)

        secdf = pd.DataFrame(secdf,index=["sec_num"]).T.reset_index().rename(columns={"index":"row_id"})
        
        self.result = pdf.merge(
            stdf[["row_id","secondary_true"]]
        ,on=["row_id"]).merge(
            ptdf[["row_id","primary_true"]]
        ,on=["row_id"]).merge(
            secdf[["row_id","sec_num"]]
        ,on=["row_id"])
        
        self.result.to_csv(f"{self.weight_dir}result.csv")

        primary_only_score = []
        for random_state in [2311,2551,8769,3772,11302,37115]:
            bootstrap_sample = self.result.sample(n=self.sample_size,random_state=random_state)
            pp = self.get_score(
                np.stack(bootstrap_sample["pred"].values),
                np.stack(bootstrap_sample["primary_true"].values),
                type="primary",
                random_state = random_state
            )
            ps = self.get_score(
                np.stack(bootstrap_sample["pred"].values),
                np.stack(bootstrap_sample["secondary_true"].values),
                type="secondary",
                random_state = random_state
            )
            bootstrap_sample_primary_only = self.result[self.result.sec_num==0].sample(n=self.sample_size,random_state=random_state)
            ppo = self.get_score(
                np.stack(bootstrap_sample_primary_only["pred"].values),
                np.stack(bootstrap_sample_primary_only["primary_true"].values),
                type="primary_only",
                random_state = random_state
            )
            primary_only_score.append(ppo)

        return np.array(primary_only_score)


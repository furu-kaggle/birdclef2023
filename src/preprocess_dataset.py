import numpy as np
import random,os, glob
import torch
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

import librosa
import joblib
from tqdm.notebook import tqdm

class generate_specimg:
    def __init__(self,
                 df: pd.DataFrame,
                 period,
                 sr, 
                 n_mels, 
                 fmin, 
                 fmax,
                 pp_dir
                 ):
        
        self.df = df
        
        #make Melspectrum
        self.sr = sr
        self.period = period
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.pp_dir = pp_dir
        kwargs = {}
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs
    
    def make_melspec(self, y):

        melspec = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sr, 
            n_mels=self.n_mels, 
            fmin=self.fmin, 
            fmax=self.fmax, 
            **self.kwargs,
        )

        melspec = librosa.power_to_db(melspec).astype(np.float32)
        return melspec
    
    def mono_to_color(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)

        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else:
            V = np.zeros_like(X, dtype=np.uint8)

        return V

    def crop_or_pad(self, y, length, is_train=False, start=None):
        if len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])

            n_repeats = length // len(y)
            epsilon = length % len(y)

            y = np.concatenate([y]*n_repeats + [y[:epsilon]])

        elif len(y) > length:
            if not is_train:
                start = start or 0
            else:
                start = start or np.random.randint(len(y) - length)

            y = y[start:start + length]

        return y
    
    def audio_to_image(self, audio):
        melspec = self.make_melspec(audio) 
        image = self.mono_to_color(melspec)
        return image

    def __call__(self, path):
        try:
            #データ読み込み
            data, sr = librosa.load(path, sr=self.sr)

            #test datasetの最大長
            max_sec = len(data)//sr

            #予測フレーム
            pred_sec = 7


            #データを5秒間隔でかつ7秒幅を取って区切る
            datas = [data[int(max(0, i-1) * sr):int(min(max_sec, i+6) * sr)] for i in range(0, max_sec, self.period)]

            #端は1秒短くなるので埋める
            datas[0] = self.crop_or_pad(datas[0] , length=sr*pred_sec)
            if len(datas) > 1:
                datas[-1] = self.crop_or_pad(datas[-1] , length=sr*pred_sec)

            #データをメル周波数によって画像化
            images = [self.audio_to_image(audio) for audio in datas]
            images = np.stack(images)

            #保存
            filename = path.split("/")[-1]
            path = f"{self.pp_dir}/{filename}_{max_sec}.npy"
            np.save(path, images)
        except:
            pass
        
def get_audios_as_images(cfg, paths):
    pool = joblib.Parallel(os.cpu_count())
    
    converter = generate_specimg(
        paths,
        period=cfg.period,
        sr=cfg.sr,
        n_mels=cfg.n_mels,
        fmin = cfg.fmin,
        fmax = cfg.fmax,
        pp_dir = cfg.pp_dir
    )
    mapper = joblib.delayed(converter)
    tasks = [mapper(path) for path in tqdm(paths)]
    pool(tqdm(tasks))
#Deep learning from pytorch
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler
from madgrad import MADGRAD

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
    epochs = 15

    #lr ratio (best fit 3)
    lr_ratio = 3

    #label smoothing rate
    smooth = 0.005

    #model name
    model_name = 'eca_nfnet_l0'

    #pretrain model path
    pretrainpath = "pretrain_weight/eca_nfnet_l0_pretrainmodel_70k_p7.bin"

    #重みを保存するディレクトリ
    weight_dir = "src/weight/eca_nfnet_l0/"

    #テストfold
    fold = 0

    def get_optimizer(model, learning_rate, ratio, decay=0):
        return  MADGRAD(params=[
            {"params": model.model.parameters(), "lr": learning_rate/ratio},
            {"params": model.fc.parameters(),    "lr": learning_rate},
        ],weight_decay=decay)

    def get_scheduler(optimizer, min_lr, epochs, warmupstep=0,warmup_lr_init=5e-5):
        # base lr は optimizerを継承
        # document:https://timm.fast.ai/SGDR
        return CosineLRScheduler(
            optimizer, 
            t_initial=20, 
            lr_min=min_lr, 
            warmup_t=warmupstep, 
            warmup_lr_init=warmup_lr_init, 
            warmup_prefix=True
        )
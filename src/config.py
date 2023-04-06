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
    period = 10
    n_mel = 128
    fmin = 50
    fmax = 14000
    power = 2
    top_db = None

    time_len = 281

    # time_len = sr[1/s] * time[s] /hop_len = sr[1/s] * time[s] 4/n_fft 
    n_fft = 1024

    hop_len = 320
    
    #バッチサイズ
    batch_size = 30

    #前処理CPUコア数
    workers = 30

    #学習率 (best range 5e-9~2e-4)
    lr = 1e-2

    #スケジューラーの最小学習率
    min_lr = 7e-5

    #ウォームアップステップ
    warmupstep = 0

    #エポック数
    epochs = 80

    #lr ratio (best fit 3)
    lr_ratio = 5

    #label smoothing rate
    smooth = 0.005

    #model name
    model_name = 'eca_nfnet_l0'

    #pretrain model path
    pretrainpath = "data/pp_nmel128f415fft1024hop320e60/model_0_60.bin"

    #重みを保存するディレクトリ
    weight_dir = "src/weight/exp/"

    #テストfold
    fold = 0

    updater = [
        30,30,30,20,20,#first 5e
        15,15,15,15,15,#next 5e
        14,14,14,14,14,#continuous
        13,13,13,13,13,#continuous
        12,12,12,12,12,#continuous
        11,11,11,11,11,#continuous
        10,10,10,10,10,#continuous
        10,10,10,10,10,#continuous
        10,10,10,10,10,#continuous
        9,9,9,9,9,#continuous
        8,8,8,8,8,#continuous
        7,7,7,7,7,#continuous
        6,6,6,6,6,#continuous
        5,5,5,5,5,#continuous
        5,5,5,5,5,#continuous
        5,5,5,5,5,#continuous
    ]

    def get_optimizer(model, learning_rate, ratio, decay=0):
        return  MADGRAD(params=[
            {"params": model.model.parameters(), "lr": learning_rate/ratio},
            {"params": model.fc.parameters(),    "lr": learning_rate},
        ],weight_decay=decay)

    def get_scheduler(optimizer, min_lr, epochs, warmupstep=3,warmup_lr_init=1e-5):
        # base lr は optimizerを継承
        # document:https://timm.fast.ai/SGDR
        return CosineLRScheduler(
            optimizer, 
            t_initial=100, 
            lr_min=min_lr, 
            warmup_t=warmupstep, 
            warmup_lr_init=warmup_lr_init, 
            warmup_prefix=True
        )
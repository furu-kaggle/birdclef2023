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
    batch_size = 8

    #前処理CPUコア数
    workers = 8

    #学習率 (best range 5e-9~2e-4)
    lr = 7e-4

    #スケジューラーの最小学習率
    min_lr = 7e-6

    #ウォームアップステップ
    warmupstep = 0

    #エポック数
    epochs = 23

    #lr ratio (best fit 3)
    lr_ratio = 3

    #label smoothing rate
    smooth = 0.005

    #model name
    model_name = 'eca_nfnet_l0'

    #pretrain model path
    pretrainpath = "pretrain_weight/eca_nfnet_l0_pretrainmodel_70k_p7.bin"

    #重みを保存するディレクトリ
    weight_dir = "src/weight/exp/"

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
            t_initial=30, 
            lr_min=min_lr, 
            warmup_t=warmupstep, 
            warmup_lr_init=warmup_lr_init, 
            warmup_prefix=True
        )
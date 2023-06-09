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
    period = 30
    n_mel = 128
    fmin = 50
    fmax = 14000
    power = 2
    top_db = None
    prilabelp = 1.0
    seclabelp = 0.5
    frame = 500
    augpower_min = 2
    augpower_max = 2
    mixup_out_prob = 0.5
    mixup_in_prob = 0.5
    backbone_dropout = 0.2
    backbone_droppath = 0.2
    head_dropout = 0.2

    mixup_alpha_in = 2.0
    mixup_alpha_out = 2.0
    sample_size = 300
    # time_len = sr[1/s] * time[s] /hop_len = sr[1/s] * time[s] 4/n_fft 
    n_fft = 1024

    hop_len = 320
    
    #バッチサイズ
    batch_size = 8

    #前処理CPUコア数
    workers = 16

    #学習率 (best range 5e-9~2e-4)
    lr = 5e-3

    #スケジューラーの最小学習率
    min_lr = 1e-5

    #ウォームアップステップ
    warmupstep = 0

    #エポック数
    epochs = 40

    #factor update
    factors = list([15,14,13,12,11,10,10,9,9,9,8,8,8,7,7,7,6,6,6]) + list([max(1, 6 - i//3) for i in range(30)])

    #lr ratio (best fit 3)
    lr_ratio = 5

    #label smoothing rate
    smooth = 0.005

    #model name
    model_name = 'eca_nfnet_l0'

    #pretrain model path
    pretrainpath = "data/pretrain_weightmodel_all_last.bin"

    #重みを保存するディレクトリ
    weight_dir = "src/weight/exp/"

    def get_optimizer(model, learning_rate, ratio, decay=0):
        return  MADGRAD(params=[
            {"params": model.model.parameters(), "lr": learning_rate/ratio},
            {"params": model.fc.parameters(),    "lr": learning_rate},
        ],weight_decay=decay)

    def get_scheduler(optimizer, min_lr, epochs, warmupstep=0,warmup_lr_init=1e-5):
        # base lr は optimizerを継承
        # document:https://timm.fast.ai/SGDR
        return CosineLRScheduler(
            optimizer, 
            t_initial=epochs+5, 
            lr_min=min_lr, 
            warmup_t=warmupstep, 
            warmup_lr_init=warmup_lr_init, 
            warmup_prefix=True
        )
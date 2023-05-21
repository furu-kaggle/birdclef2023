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
    seclabelp = 0.75
    frame = 500
    augpower_min = 2
    augpower_max = 2
    mixup_out_prob = 1.0
    mixup_in_prob1 = 0.5
    mixup_in_prob2 = 0.5
    noise_aug_p = 0.5
    geometric_mixup_p = 0.75
    fm_prob =  0.5
    backbone_dropout = 0.2
    backbone_droppath = 0.2
    head_dropout = 0

    mixup_alpha_in = 2.0
    mixup_alpha_out = 2.0
    sample_size = 300
    # time_len = sr[1/s] * time[s] /hop_len = sr[1/s] * time[s] 4/n_fft 
    n_fft = 1024

    hop_len = 320
    
    #バッチサイズ
    batch_size = 15

    #前処理CPUコア数
    workers = 30

    #学習率 (best range 5e-9~2e-4)
    lr = 5e-3

    #スケジューラーの最小学習率
    min_lr = 3e-5

    #ウォームアップステップ
    warmupstep = 10

    #エポック数
    epochs = 105

    #factor update
    factors = list([25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10]) + list([max(1, 9 - i//8) for i in range(epochs)])

    batch_factor = {
        25:1,24:1,23:1,22:1,21:1,20:1,19:1,18:1,17:1,16:1,15:1,14:1,13:1,12:1,11:1,10:2,9:2,8:2,
        7:2,6:2,5:4,4:4,3:4,2:4,1:5
    }

    mixup_fm = True

    #lr ratio (best fit 3)
    lr_ratio = 5

    #label smoothing rate
    smooth = 0.01

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

    def get_scheduler(optimizer, min_lr, epochs, warmupstep=warmupstep,warmup_lr_init=1e-5):
        # base lr は optimizerを継承
        # document:https://timm.fast.ai/SGDR
        return CosineLRScheduler(
            optimizer, 
            t_initial=110, 
            lr_min=min_lr, 
            warmup_t=warmupstep, 
            warmup_lr_init=warmup_lr_init, 
            warmup_prefix=True,
            k_decay = 1
        )
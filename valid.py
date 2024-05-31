import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from model.LSTM_attention import *
from model.RIFE import Model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from model.VimeoSeptuplet import *

device = torch.device("cuda")

log_path = 'train_log'
intrain_path = 'intrain_log'

from model.pretrained_RIFE_loader import IFNet_update
from model.pretrained_RIFE_loader import convert_load




def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_tea_pred_list = []
    loss_mse_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    addImageFlag = 1

    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        timestep = timestep.to(device, non_blocking=True)
        # Changed for Septuplet 0 1 2 3 4 5 6
        b_data_gpu, _, h_data_gpu, w_data_gpu = data_gpu.shape
        display_all = data_gpu.view(b_data_gpu, 7, 3, h_data_gpu, w_data_gpu)
        gt = []
        pred = []
        with torch.no_grad():
            pred_all, teapred, info = model.update(data_gpu, training=False)
            merged_img = info['merged_tea']
        for iframe in range(3):
            gt.append(data_gpu[:, 6 * iframe + 3:6 * iframe + 6])
            gt = torch.stack(gt, dim=1)  # B*N*C*H*W
            pred.append(pred_all[:, 6 * iframe + 3:6 * iframe + 6])
            pred = torch.stack(pred, dim=1)  # B*N*C*H*W

        for j in range(gt.shape[0]):
            sep_PSNR = torch.tensor(0.0)
            for k in range(3):
                epsilon = 1e-9  # Small value to avoid log10(0)
                mse = torch.mean((gt[j][k] - pred[j][k]) ** 2).cpu().data  # Better to use **2 for clarity
                single_psnr = -10 * math.log10(mse + epsilon)
                sep_PSNR = sep_PSNR + single_psnr
            sep_PSNR = sep_PSNR / 3
            print("AVERAGE PSNR:", sep_PSNR, "currently at ", i)
            psnr_list.append(sep_PSNR)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=7, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(args.local_rank)

    torch.backends.cudnn.benchmark = True
    pretrained_path = 'RIFE_log'
    checkpoint = convert_load(torch.load(f'{pretrained_path}/flownet.pkl', map_location=device))
    Ori_IFNet_loaded = IFNet_update()
    Ori_IFNet_loaded.load_state_dict(checkpoint)
    for param in Ori_IFNet_loaded.parameters():
        param.requires_grad = False

    model = Model(Ori_IFNet_loaded, args.local_rank)
    # Dummy parameters
    step = 5555
    dataset_val = VimeoDatasetSep('test')
    val_data = DataLoader(dataset_val, batch_size=4, pin_memory=True, num_workers=1)
    writer_val = SummaryWriter('validate')

    pretrained_model_path = '/root/attention/CBAM/RIFE_LSTM_Context/intrain_log'
    model.load_model(pretrained_model_path)
    print("Loaded ConvLSTM model")
    model.eval()

    evaluate(model, val_data, step, args.local_rank, writer_val)

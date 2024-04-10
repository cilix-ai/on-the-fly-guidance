import os, glob, torch
import numpy as np
from natsort import natsorted


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES, power), 8)


def calc_learning_rate(epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    return round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

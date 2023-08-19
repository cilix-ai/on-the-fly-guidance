from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.ViTVNet import CONFIGS as CONFIGS_ViT
from models.ViTVNet import ViTVNet
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from models.VoxelMorph import VxmDense_1

from models.Optron import Optron
from csv_logger import log_csv

import argparse


# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', type=str, default='../autodl-fs/IXI_data/Train/')
parser.add_argument('--val_dir', type=str, default='../autodl-fs/IXI_data/Val/')
parser.add_argument('--dataset', type=str, default='IXI')
parser.add_argument('--atlas_dir', type=str, default='../autodl-fs/IXI_data/atlas.pkl')

parser.add_argument('--model', type=str, default='TransMorph')

parser.add_argument('--training_lr', type=float, default=1e-4)
parser.add_argument('--optron_lr', type=float, default=1e-2)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=500)

parser.add_argument('--optron_epoch', type=int, default=10)

args = parser.parse_args()


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    batch_size = 1
    atlas_dir = args.atlas_dir
    train_dir = args.train_dir
    val_dir = args.val_dir
    weights = [1, 1] # loss weights
    save_dir = '{}_ncc_{}_diffusion_{}/'.format(args.model, weights[0], weights[1])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = args.training_lr # learning rate
    epoch_start = args.epoch_start
    max_epoch = args.max_epoch #max traning epoch

    '''
    Initialize model
    '''
    img_size = (160, 192, 224)
    if args.model == "TransMorph":
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorph(config)
    elif args.model == "VoxelMorph":
        model = VxmDense_1(img_size)
    elif args.model == "ViTVNet":
        config_vit = CONFIGS_ViT['ViT-V-Net']
        model = ViTVNet(config_vit, img_size=img_size)

    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if epoch_start:
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    if args.dataset == "IXI":
        train_composed = transforms.Compose([trans.RandomFlip(0),
                                            trans.NumpyType((np.float32, np.float32)),
                                            ])

        val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                        trans.NumpyType((np.float32, np.int16))])
        
        train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
        val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    elif args.dataset == "OASIS":
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
        val_set = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    criterions += [nn.MSELoss()]
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            if args.dataset == "OASIS":
                x_seg = data[2]
                y_seg = data[3]
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)

            optron1 = Optron(output[1].clone().detach())
            Optron_optimizer = optim.Adam(optron1.parameters(), lr=args.optron_lr, weight_decay=0, amsgrad=True)
            adjust_learning_rate(Optron_optimizer, epoch, max_epoch, args.optron_lr)
            for i in range(args.optron_epoch):
                x_warped, optimized_flow = optron1(x)
                Optron_loss_ncc = criterions[0](x_warped, y) * weights[0]
                Optron_loss_reg = criterions[1](optimized_flow, y) * weights[1]
                Optron_loss = Optron_loss_ncc + Optron_loss_reg

                Optron_optimizer.zero_grad()
                Optron_loss.backward()
                Optron_optimizer.step()

            x_warped, optimized_flow = optron1(x)
            loss_mse = criterions[2](output[1], optimized_flow)
            loss_reg = criterions[1](optimized_flow, y) * 0.02
            loss = loss_mse + loss_reg
            loss_vals = [loss_mse, loss_reg]
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.dataset == "OASIS":
                y_in = torch.cat((y, x), dim=1)
                output = model(y_in)

                optron2 = Optron(output[1].clone().detach())
                Optron_optimizer = optim.Adam(optron2.parameters(), lr=args.optron_lr, weight_decay=0, amsgrad=True)
                adjust_learning_rate(Optron_optimizer, epoch, max_epoch, args.optron_lr)

                for i in range(args.optron_epoch):
                    y_warped, optimized_flow = optron2(y)
                    Optron_loss_ncc = criterions[0](y_warped, x) * weights[0]
                    Optron_loss_reg = criterions[1](optimized_flow, x) * weights[1]
                    Optron_loss = Optron_loss_ncc + Optron_loss_reg

                    Optron_optimizer.zero_grad()
                    Optron_loss.backward()
                    Optron_optimizer.step()

                loss_mse = criterions[2](optimized_flow, output[1])
                loss_reg = criterions[1](optimized_flow, x) * 0.02
                loss = loss_mse + loss_reg
                loss_vals = [loss_mse, loss_reg]
                
                loss_all.update(loss.item(), x.numel())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x, y, x_seg, y_seg = data
                x_in = torch.cat((x, y), dim=1)
                output = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        log_csv(epoch, eval_dsc.avg, loss_all.avg)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        loss_all.reset()
    writer.close()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()

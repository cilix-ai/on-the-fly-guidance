import matplotlib.pyplot as plt
import os, glob
import numpy as np
from natsort import natsorted
from data import datasets, trans
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models.ViTVNet import CONFIGS as CONFIGS_ViT
from models.ViTVNet import ViTVNet
from models.TransMorph import CONFIGS as CONFIGS_TM
from models.TransMorph import TransMorph
from models.VoxelMorph import VoxelMorph
from models.Optron import Optron
import utils.utils as utils
import utils.losses as losses
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='TransMorph')
parser.add_argument('--weights', type=str, default='./checkpoints/LPBA/trm_opt/dsc0.684_epoch332.pth.tar')

parser.add_argument('--dataset', type=str, default='LPBA')
parser.add_argument('--val_dir', type=str, default='./datasets/LPBA40/test/')
parser.add_argument('--label_dir', type=str, default='./datasets/LPBA40/label/')
parser.add_argument('--atlas_dir', type=str, default='./datasets/LPBA40/atlas.nii.gz')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--step_size', type=int, default=0.2)
parser.add_argument('--resolution', type=int, default=10)
args = parser.parse_args()

def visualize_loss_landscape(model, dataloader, dir1, dir2, res, step_size, optim_point):
    x_grid, y_grid = np.meshgrid(np.arange(-res, res, 1), np.arange(-res, res, 1))
    loss_grid = np.zeros_like(x_grid).astype('float')
    shapes = get_param_shapes(model)

    # Compute the loss for each point in the grid
    print('Computing losses for each point in the grid...')
    with torch.no_grad():
        for i in range(-res, res):
            for j in range(-res, res):
                params_new = (
                    optim_point + 
                    i * step_size * dir1 +
                    j * step_size * dir2
                )
                init_from_flat_params(model, params_new, shapes)
                loss = compute_loss(model, dataloader)
                loss_grid[i][j] = loss
                print(f'({i}, {j}): {loss}')
    
    # normalize loss_grid
    # print(loss_grid.min(), loss_grid.max(), loss_grid.max() - loss_grid.min())
    # print(loss_grid - loss_grid.min())
    loss_grid = (loss_grid - loss_grid.min()) / (loss_grid.max() - loss_grid.min())
    print(loss_grid)
    
    # Plot the loss landscape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, loss_grid, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('dir1')
    ax.set_ylabel('dir2')
    ax.set_zlabel('Loss')
    # ax.set_zlim(loss_grid.min(), loss_grid.max())
    plt.show()
    # scatter = ax.scatter(x, y, z, c=colors, cmap='viridis')  # cmap 参数指定了颜色的映射方式，这里使用了 'viridis'
    # plt.colorbar(scatter, label='Z Value')
    plt.savefig('./loss_landscape.png')
      
def compute_loss(model, dataloader):
    model.eval()
    loss_all = utils.AverageMeter()
    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')
    with torch.no_grad():
        for data in dataloader:
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)
            loss_ncc = criterion_ncc(output[0], y)
            loss_reg = criterion_reg(output[1], y)
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())
    return loss_all.avg
    
def load_model(img_size):
    if args.model == 'TransMorph':
        config = CONFIGS_TM['TransMorph']
        if args.dataset == 'LPBA':
            config.img_size = img_size
            config.window_size = (5, 6, 5, 5)
        model = TransMorph(config)
    elif args.model == 'VoxelMorph':
        model = VoxelMorph(img_size)
    elif args.model == 'ViTVNet':
        config = CONFIGS_ViT['ViT-V-Net']
        model = ViTVNet(config, img_size=img_size)
    model.cuda()
    
    return model

def load_data():
    if args.dataset == "IXI":
        val_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))])    
        val_set = datasets.IXIBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), args.atlas_dir, transforms=val_composed)
    elif args.dataset == "OASIS":
        val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        val_set = datasets.OASISBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), transforms=val_composed)
    elif args.dataset == "LPBA":
        val_set = datasets.LPBAInferDataset(glob.glob(args.val_dir + '*.nii.gz'), args.atlas_dir, args.label_dir, transforms=None)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    
    return val_loader

def get_flat_params(model):
    """Get flattened and concatenated params of the model."""
    params = model.state_dict()
    flat_params = torch.Tensor().cuda()
    for _, param in params.items():
        flat_params = torch.cat((flat_params, torch.flatten(param)))
    return flat_params

def init_from_flat_params(model, flat_params, shapes):
    """Set all model parameters from the flattened form."""
    if not isinstance(flat_params, torch.Tensor):
        raise AttributeError(
            "Argument to init_from_flat_params() must be torch.Tensor"
        )
    state_dict = unflatten_to_state_dict(flat_params, shapes)
    model.load_state_dict(state_dict, strict=True)

def get_param_shapes(model):
    shapes = []
    for name, param in model.state_dict().items():
        shapes.append((name, param.shape, param.numel()))
    return shapes

def unflatten_to_state_dict(flat_w, shapes):
    state_dict = {}
    counter = 0
    for shape in shapes:
        name, tsize, tnum = shape
        param = flat_w[counter : counter + tnum].reshape(tsize)
        state_dict[name] = torch.nn.Parameter(param)
        counter += tnum
    assert counter == len(flat_w), "counter must reach the end of weight vector"
    return state_dict

    
if __name__ == '__main__':
    img_size = (160, 192, 160) if args.dataset == 'LPBA' else (160, 192, 224)
    model = load_model(img_size)
    model.cuda()
    pretrained = torch.load(args.weights)['state_dict']
    model.load_state_dict(pretrained)
    
    val_loader = load_data()

    '''
    generate 2 random direction vectors
    '''
    optim_params = get_flat_params(model)
    np.random.seed(args.seed)
    dir1 = np.random.random(optim_params.shape)
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = np.random.random(optim_params.shape)
    dir2 = dir2 / np.linalg.norm(dir2)
    dir1 = torch.from_numpy(dir1).cuda()
    dir2 = torch.from_numpy(dir2).cuda()
    
    visualize_loss_landscape(model, val_loader, dir1, dir2, args.resolution, args.step_size, optim_params)
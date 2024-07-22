import os, utils.utils as utils, glob, sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.ViTVNet import CONFIGS as CONFIGS_ViT
from models.ViTVNet import ViTVNet
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from models.VoxelMorph import VoxelMorph
import argparse
import matplotlib.pyplot as plt
import nibabel as nib



# parse the commandline
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IXI')
parser.add_argument('--test_dir', type=str, default='../datasets/IXI_data/Val/')
parser.add_argument('--atlas_dir', type=str, default='../datasets/IXI_data/atlas.pkl')
parser.add_argument('--label_dir', type=str, default='../datasets/LPBA40/label/')
parser.add_argument('--model', type=str, default='TransMorph')
parser.add_argument('--model_dir', type=str, default='./checkpoints/trm/')
parser.add_argument('--model_opt_dir', type=str, default='./checkpoints/trm_opt/')
parser.add_argument('--save_dir', type=str, default='./results/')
args = parser.parse_args()

def main():
    save_dir = args.save_dir + args.dataset + '/' + args.model + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    """Initialize model"""
    img_size = (160, 192, 160) if args.dataset == "LPBA" else (160, 192, 224)
    if args.model == "TransMorph":
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorph(config)
        model_opt = TransMorph.TransMorph(config)
    elif args.model == "VoxelMorph":
        model = VoxelMorph(img_size)
        model_opt = VoxelMorph(img_size)
    elif args.model == "ViTVNet":
        config_vit = CONFIGS_ViT['ViT-V-Net']
        model = ViTVNet(config_vit, img_size=img_size)
        model_opt = ViTVNet(config_vit, img_size=img_size)
    
    """Load model weights"""
    if args.model_dir is None:
        raise ValueError("model_dir is None")
    elif args.model_opt_dir is None:
        raise ValueError("model_opt_dir is None")
    else:
        model_dir = args.model_dir
        model_opt_dir = args.model_opt_dir
    model_idx = -1
    
    # load weights of regular model
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    model.load_state_dict(best_model)
    model.eval()
    model.cuda()
    
    # load weights of model with ofg
    best_model = torch.load(model_opt_dir + natsorted(os.listdir(model_opt_dir))[model_idx])['state_dict']
    model_opt.load_state_dict(best_model)
    model_opt.eval()
    model_opt.cuda()
    
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()
    
    
    """load test dataset"""
    test_dir = args.test_dir 
    atlas_dir = args.atlas_dir
    if args.dataset == 'IXI':
        test_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16)),])
        test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    elif args.dataset == 'OASIS':
        test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
        test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    elif args.dataset == "LPBA":
        test_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.LPBAInferDataset(glob.glob(test_dir + '*.nii.gz'), atlas_dir, args.label_dir, transforms=test_composed)
    else:
        raise ValueError("Dataset name is wrong!")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    
    
    '''Default: plot the registration results of the first image pair in test_loader'''
    grid_img = mk_grid_img(8, 1, img_size)
    with torch.no_grad():
        for data in test_loader:
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_in = torch.cat((x,y),dim=1)
            
            # model
            x_def, flow = model(x_in)
            def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
            def_seg = reg_model([x_seg.cuda().float(), flow.cuda()])
            flow = flow.cpu().detach().numpy()[0]
            
            # model_opt
            x_def_opt, flow_opt = model_opt(x_in)
            def_grid_opt = reg_model_bilin([grid_img.float(), flow_opt.cuda()])
            def_seg_opt = reg_model([x_seg.cuda().float(), flow_opt.cuda()])
            flow_opt = flow_opt.cpu().detach().numpy()[0]

            # compute the min and max value of each channel of flow and flow_opt for normalization later
            flow_min = np.min(flow, axis=(1, 2, 3))
            flow_max = np.max(flow, axis=(1, 2, 3))
            flow_opt_min = np.min(flow_opt, axis=(1, 2, 3))
            flow_opt_max = np.max(flow_opt, axis=(1, 2, 3))
            v_min = np.stack((flow_min, flow_opt_min)).min(axis=0)
            v_max = np.stack((flow_max, flow_opt_max)).max(axis=0)

            # transform the deformation field to RGB image
            rgb = def2rgb(flow, v_min, v_max)
            rgb_opt = def2rgb(flow_opt, v_min, v_max)
            break
    
    var = [x, y, x_seg, y_seg, x_def, x_def_opt, def_seg, def_seg_opt, def_grid, def_grid_opt]
    for i, _ in enumerate(var):
        var[i] = var[i].squeeze(0).squeeze(0).cpu().detach().numpy()
    x, y, x_seg, y_seg, x_def, x_def_opt, def_seg, def_seg_opt, def_grid, def_grid_opt = var

    # save volumes and use other apps, like 3D-slicer, to open
    var = [x, y, x_seg, y_seg, x_def, x_def_opt, def_seg, def_seg_opt, flow[None, ...].transpose(3, 4, 2, 0, 1), flow_opt[None, ...].transpose(3, 4, 2, 0, 1)]
    file_name = ['x', 'y', 'x_seg', 'y_seg', 'x_def', 'x_def_ofg', 'def_seg', 'def_seg_ofg', 'disp', 'disp_ofg']
    for d, name in list(zip(var, file_name)):
        nib_write(d, save_dir+name+'.nii.gz')
    
    # save images
    idx = 0
    var = [x, y, x_def, x_def_opt, rgb, rgb_opt, def_grid, def_grid_opt]
    file_name = ['fixed', 'moving', 'warped', 'warped_ofg', 'def_field', 'def_field_ofg', 'def_grid', 'def_grid_ofg']
    for v, name in list(zip(var, file_name)):
        plt.figure(++idx)
        plt.axis('off')
        if v.shape[-1] == 3:
            plt.imshow(v[:, 96, :, :].astype('uint8'))
        else:
            plt.imshow(v[:,  96, :], cmap='gray')
        plt.savefig(save_dir+name+'.png')


def def2rgb(disp, v_min, v_max):
    # Normalize deformation field
    C, H, W, L = disp.shape
    for i in range(C):
        disp[i] = (disp[i] - v_min[i]) / (v_max[i] - v_min[i]) # normalize to [0, 1]
        disp[i] = disp[i] * 255
    return np.transpose(disp, (3, 2, 1, 0))


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def nib_write(data, save_dir):
    data_nib = nib.Nifti1Image(data, np.eye(4))
    if data.shape[-1] == 3:
        data_nib.header.set_intent(1007)
    else:
        data_nib.header.get_xyzt_units()
    data_nib.to_filename(save_dir)


if __name__ == '__main__':
    main()
import os, utils, glob, sys
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
from models.VoxelMorph import VxmDense_1
import argparse
import matplotlib.pyplot as plt
import nibabel as nib


# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='IXI')
parser.add_argument('--test_dir', type=str, default='../dataset/IXI_data/Val/')
parser.add_argument('--atlas_dir', type=str, default='../dataset/IXI_data/atlas.pkl')
parser.add_argument('--model', type=str, default='TransMorph')
parser.add_argument('--model_dir', type=str, default='./experiments/trm/')
parser.add_argument('--model_opt_dir', type=str, default='./experiments/trm_opt/')

args = parser.parse_args()

def main():
    """Initialize model"""
    img_size = (160, 192, 224)
    if args.model == "TransMorph":
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorph(config)
        model_opt = TransMorph.TransMorph(config)
    elif args.model == "VoxelMorph":
        model = VxmDense_1(img_size)
        model_opt = VxmDense_1(img_size)
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
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.eval()
    model.cuda()
    
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
    if args.dataset == 'IXI':
        atlas_dir = args.atlas_dir
        test_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16)),])
        test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    elif args.dataset == 'OASIS':
        test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
        test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    else:
        raise ValueError("Dataset name is wrong!")
    
    '''Default: plot the registration results of the first image pair in test_loader'''
    with torch.no_grad():
        for data in test_loader:
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            grid_img = mk_grid_img(8, 1, img_size)
            x_in = torch.cat((x,y),dim=1)
            
            # model
            x_def, flow = model(x_in)
            def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
            def_seg = reg_model([x_seg.cuda().float(), flow.cuda()])
            flow = flow.cpu().detach().numpy()
            rgb = def2rgb(flow[0])
            
            # model_opt
            x_def_opt, flow_opt = model_opt(x_in)
            def_grid_opt = reg_model_bilin([grid_img.float(), flow_opt.cuda()])
            def_seg_opt = reg_model([x_seg.cuda().float(), flow_opt.cuda()])
            flow_opt = flow_opt.cpu().detach().numpy()
            rgb_opt = def2rgb(flow_opt[0])
            
            break
    
    var = [x, y, x_seg, y_seg, x_def, x_def_opt, def_seg, def_seg_opt, def_grid, def_grid_opt]
    for i, _ in enumerate(var):
        var[i] = var[i].cpu().detach().numpy()
    x, y, x_seg, y_seg, x_def, x_def_opt, def_seg, def_seg_opt, def_grid, def_grid_opt = var

    affine = np.eye(4)
    x_nib = nib.Nifti1Image(x.squeeze(0).squeeze(0), affine)
    x_nib.header.get_xyzt_units()
    x_nib.to_filename('./results/outputs/x.nii.gz')
    y_nib = nib.Nifti1Image(y.squeeze(0).squeeze(0), affine)
    y_nib.header.get_xyzt_units()
    y_nib.to_filename('./results/outputs/y.nii.gz')
    
    x_seg_nib = nib.Nifti1Image(x_seg.squeeze(0).squeeze(0), affine)
    x_seg_nib.header.get_xyzt_units()
    x_seg_nib.to_filename('./results/outputs/x_seg.nii.gz')
    y_seg_nib = nib.Nifti1Image(y_seg.squeeze(0).squeeze(0), affine)
    y_seg_nib.header.get_xyzt_units()
    y_seg_nib.to_filename('./results/outputs/y_seg.nii.gz')

    def_seg_nib = nib.Nifti1Image(def_seg.squeeze(0).squeeze(0), affine)
    def_seg_nib.header.get_xyzt_units()
    def_seg_nib.to_filename('./results/outputs/def_seg.nii.gz')
    def_seg_opt_nib = nib.Nifti1Image(def_seg_opt.squeeze(0).squeeze(0), affine)
    def_seg_opt_nib.header.get_xyzt_units()
    def_seg_opt_nib.to_filename('./results/outputs/def_seg_opt.nii.gz')
    
    num_row, num_col = 3, 4
    fig, axs = plt.subplots(num_row, num_col, figsize=(7, 5), squeeze=False, tight_layout={'pad': 0})

    # fixed image
    for j in range(num_row):
        axs[j, 0].imshow(y[0, 0, :, 96, :], cmap='gray')
        axs[j, 0].axis('off')
        
    axs[0, 0].set_title('Fixed')
    axs[0, 1].set_title('Moving')
    # axs[1, 0].set_ylabel('TRM')
    # axs[2, 0].set_ylabel("TRM_Opt")

    # moving image
    axs[0, 1].imshow(x[0, 0, :, 96, :], cmap='gray')
    axs[0, 1].axis('off')
    
    axs[0, 2].set_visible(False)
    axs[0, 3].set_visible(False)
    
    # warped moving image of model
    axs[1, 1].imshow(x_def[0, 0, :, 96, :], cmap='gray')
    axs[1, 1].axis('off')
    
    # warped moving image of model_opt
    axs[2, 1].imshow(x_def_opt[0, 0, :, 96, :], cmap='gray')
    axs[2, 1].axis('off')
    
    # deformation field of model
    axs[1, 2].imshow(rgb[:, 96, :].astype('uint8'))
    axs[1, 2].axis('off')    
    
    # deformation field of model_opt
    axs[2, 2].imshow(rgb_opt[:, 96, :].astype('uint8'))
    axs[2, 2].axis('off')  
    
    # grid image of model
    # axs[1, 3].set_visible(False)
    axs[1, 3].imshow(def_grid[0, 0, :, 96, :], cmap='gray')
    axs[1, 3].axis('off')    
    
    # grid image of model_opt
    # axs[2, 3].set_visible(False)
    axs[2, 3].imshow(def_grid_opt[0, 0, :, 96, :], cmap='gray')
    axs[2, 3].axis('off')
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.00005, hspace=0.005)
    plt.show()               


def def2rgb(disp):
    # Normalize deformation field
    C, H, W, L = disp.shape
    for i in range(C):
        min = disp[i].min()
        max = disp[i].max()
        disp[i] = (disp[i] - min) / (max - min) # normalize to [0, 1]
        disp[i] = (disp[i] * 255) # normalize to [0, 255]
    
    return np.transpose(disp, (1, 2, 3, 0))

# def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
#     grid_img = np.zeros(grid_sz)
#     for j in range(0, grid_img.shape[1], grid_step):
#         grid_img[:, j+line_thickness-1, :] = 1
#     for i in range(0, grid_img.shape[2], grid_step):
#         grid_img[:, :, i+line_thickness-1] = 1
#     grid_img = grid_img[None, None, ...]
#     grid_img = torch.from_numpy(grid_img).cuda()
#     return grid_img

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img
        

if __name__ == '__main__':
    main()
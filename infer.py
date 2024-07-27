import os, glob, time
import numpy as np
from natsort import natsorted
import nibabel as nib

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.ViTVNet import CONFIGS as CONFIGS_ViT
from models.ViTVNet import ViTVNet
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from models.VoxelMorph import VoxelMorph

import utils.utils as utils
from utils.csv_logger import infer_csv

from data import datasets, trans

import argparse

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IXI')
parser.add_argument('--test_dir', type=str, default='../datasets/IXI_data/Val/')
parser.add_argument('--label_dir', type=str, default='../datasets/LPBA40/label/')
parser.add_argument('--atlas_dir', type=str, default='../datasets/IXI_data/atlas.pkl')
parser.add_argument('--save_dir', type=str, default='./infer_results/', help='The directory to save infer results')
parser.add_argument('--model', type=str, default='TransMorph')
parser.add_argument('--model_dir', type=str, default=None, help='The directory path that saves model weights')
parser.add_argument('--ofg', action='store_true', help="use ofg or not")
args = parser.parse_args()


def main():
    if args.opt:
        csv_name = args.model + '_opt.csv'
    else:
        csv_name = args.model + '.csv'

    save_dir = args.save_dir + args.dataset + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """Initialize model"""
    img_size = (160, 192, 160) if args.dataset == "LPBA" else (160, 192, 224)
    if args.model == "TransMorph":
        config = CONFIGS_TM['TransMorph']
        if args.dataset == 'LPBA':
            config.img_size = img_size
            config.window_size = (5, 6, 5, 5)
        model = TransMorph.TransMorph(config)
    elif args.model == "VoxelMorph":
        model = VoxelMorph(img_size)
    elif args.model == "ViTVNet":
        config_vit = CONFIGS_ViT['ViT-V-Net']
        model = ViTVNet(config_vit, img_size=img_size)
    else:
        raise ValueError("{} doesn't exist!".format(args.model))

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    """Load model weights"""
    if args.model_dir is None:
        raise ValueError("model_dir is None")
    else:
        model_dir = args.model_dir
    model_idx = -1
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

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
    elif args.dataset == "AbdomenCTCT":
        # !!! for AbdomenCTCT, please supply the most outside dir as the test path, i.e., the path of the entire dataset
        test_set = datasets.AbdomenCTCT(path=os.path.join(args.test_dir, 'test.json'), img_dir=os.path.join(args.test_dir, 'imagesTr/'), labels_dir=os.path.join(args.test_dir, 'labelsTr/'))
    else:
        raise ValueError("Unknown dataset")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

    """start inferring"""    
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    eval_time = utils.AverageMeter()

    print("Start Inferring\n")
    with torch.no_grad():
        idx = 0
        for data in test_loader:
            idx += 1
            print("Test image:", idx)
            model.eval()
            data = [t.cuda() for t in data]
            x, y, x_seg, y_seg = data[0], data[1], data[2], data[3]
            x_in = torch.cat((x,y),dim=1)

            # evaluate infer time
            time_start = time.time()
            x_def, flow = model(x_in)
            time_end = time.time()
            eval_time.update(time_end - time_start, x.size(0))
            print("{}s".format(time_end - time_start))

            # evaluate Jdet
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            Jdet = np.sum(jac_det <= 0) / np.prod(tar.shape)
            eval_det.update(Jdet, x.size(0))
            print('det < 0: {}'.format(Jdet))

            # evaluate DSC
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            if args.dataset == "OASIS":
                dsc_trans = utils.dice_OASIS(def_out.long(), y_seg.long())
                dsc_raw = utils.dice_OASIS(x_seg.long(), y_seg.long())
            elif args.dataset == "IXI":
                dsc_trans = utils.dice_IXI(def_out.long(), y_seg.long())
                dsc_raw = utils.dice_IXI(x_seg.long(), y_seg.long())
            elif args.dataset == "LPBA":
                dsc_trans = utils.dice_LPBA(y_seg.cpu().detach().numpy(), def_out[0, 0, ...].cpu().detach().numpy())
                dsc_raw = utils.dice_LPBA(y_seg.cpu().detach().numpy(), x_seg.cpu().detach().numpy())
            elif args.dataset == "AbdomenCTCT":
                dsc_1 = utils.dice_AbdomenCTCT(y_seg.contiguous(), def_out.contiguous(), 14).cpu() # trans
                dsc_2 = utils.dice_AbdomenCTCT(y_seg.contiguous(), x_seg.contiguous(), 14).cpu() # raw
                dsc_ident = utils.dice_AbdomenCTCT(y_seg.contiguous(), y_seg.contiguous(), 14).cpu() * \
                            utils.dice_AbdomenCTCT(x_seg.contiguous(), x_seg.contiguous(), 14).cpu()
                dsc_trans = dsc_1.sum() / (dsc_ident > 0.1).sum()
                dsc_raw = dsc_2.sum() / (dsc_ident > 0.1).sum()

            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

            infer_csv(save_dir, csv_name, idx, dsc_raw.item(), dsc_trans.item(), Jdet, time_end - time_start)
            print()

        print('Average:')
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('time: {}s'.format(eval_time.avg))
        infer_csv(save_dir, csv_name, 'avg', eval_dsc_raw.avg, eval_dsc_def.avg, eval_det.avg, eval_time.avg)


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

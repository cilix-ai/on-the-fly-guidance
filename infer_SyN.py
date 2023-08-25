import glob
import os, utils, torch
import sys, ants
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
from torchvision import transforms
import nibabel as nib
import argparse
import time


# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument('--val_dir', type=str, default='../IXI_data/Val/')
parser.add_argument('--label_dir', type=str, default='../LPBA40/label/')
parser.add_argument('--dataset', type=str, default='IXI')
parser.add_argument('--atlas_dir', type=str, default='../IXI_data/atlas.pkl')
args = parser.parse_args()

def main():
    '''
    Initialize training
    '''
    if args.dataset == "IXI":
        val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                        trans.NumpyType((np.float32, np.int16))])
        
        val_set = datasets.IXIBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), args.atlas_dir, transforms=val_composed)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    elif args.dataset == "OASIS":
        val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        val_set = datasets.OASISBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), transforms=val_composed)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    elif args.dataset == "LPBA":
        val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                        trans.NumpyType((np.float32, np.int16))])
        
        val_set = datasets.LPBAInferDataset(glob.glob(args.val_dir + '*.nii.gz'), args.atlas_dir, args.label_dir, transforms=val_composed)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    
    eval_dsc = utils.AverageMeter()
    eval_Jdet = utils.AverageMeter()
    eval_time = utils.AverageMeter()
    with torch.no_grad():
        i = 0
        for data in val_loader:
            i += 1
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            x_seg = data[2].squeeze(0).squeeze(0).detach().cpu().numpy()
            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()
            x = ants.from_numpy(x)
            y = ants.from_numpy(y)

            x_ants = ants.from_numpy(x_seg.astype(np.float32))
            y_ants = ants.from_numpy(y_seg.astype(np.float32))

            time_start = time.time()
            reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
            time_end = time.time()
            eval_time.update(time_end - time_start, 1)
            
            def_seg = ants.apply_transforms(fixed=y_ants,
                                            moving=x_ants,
                                            transformlist=reg12['fwdtransforms'],
                                            interpolator='nearestNeighbor',)
                                            #whichtoinvert=[True, False, True, False]

            flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:,:,:,0,:].transpose(3, 0, 1, 2)
            def_seg = def_seg.numpy()
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            y_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_trans = utils.dice_val(def_seg.long(), y_seg.long(), 46)
            eval_dsc.update(dsc_trans.item(), 1)
            jac_det = utils.jacobian_determinant_vxm(flow)
            Jdet = np.sum(jac_det <= 0) / np.prod(y_seg.shape)
            eval_Jdet.update(Jdet, 1)
            
            print(i)
            print('Jdet < 0: {}'.format(Jdet))
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            print("{:d}s".format(time_end-time_start))
            print()

        print("Average:")
        print('Jdet < 0: {} +- {}'/format(eval_Jdet.avg, eval_Jdet.std))
        print('DSC: {:.5f} +- {:.5f}'.format(eval_dsc.avg, eval_dsc.std))
        print('time: {:d}s'.format(eval_time.avg))

        
def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    main()
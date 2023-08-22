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
import nibabel as nib

# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument('--test_dir', type=str, default='../autodl-fs/IXI_data/Test/')
parser.add_argument('--save_dir', type=str, default='./results/')
parser.add_argument('--dataset', type=str, default='IXI')
parser.add_argument('--atlas_dir', type=str, default='../autodl-fs/IXI_data/atlas.pkl')
parser.add_argument('--model', type=str, default='TransMorph')
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--opt', action='store_true', help="whether use optimizer during training")
parser.add_argument('--debug', action='store_true', help="if true, only infer the first val pair and save the deformation field")

args = parser.parse_args()

def main():
    if args.opt:
        csv_name = args.model + '_opt.csv'
    else:
        csv_name = args.model + '.csv'
    
    save_dir = args.save_dir + args.dataset + '/'
    # if not os.path.exists(save_dir + 'deformation_fields/'):
    #     os.makedirs(save_dir + 'deformation_fields/')
        
    """Initialize model"""
    img_size = (160, 192, 224)
    if args.model == "TransMorph":
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorph(config)
    elif args.model == "VoxelMorph":
        model = VxmDense_1(img_size)
    elif args.model == "ViTVNet":
        config_vit = CONFIGS_ViT['ViT-V-Net']
        model = ViTVNet(config_vit, img_size=img_size)
    
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
    
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    
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
    
    """start infering"""    
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    print("Start Inferring\n")
    with torch.no_grad():
        idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)
            
            if args.debug:
                # save deformation field
                def_field = flow.cpu().detach().numpy()
                def_field = np.transpose(def_field, (2, 3, 4, 1, 0))
                affine = np.eye(4)
                nii = nib.Nifti1Image(def_field, affine)
                nib.save(nii, save_dir + 'def_field.nii.gz')
                sys.exit(0)
                        
            #! more accurate
            # x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            # x_seg_oh = torch.squeeze(x_seg_oh, 1)
            # x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            # def_out = model.spatial_trans(x_seg.float(), flow.float())
            # x_segs = []
            # for i in range(46):
            #     def_seg = reg_model([x_seg_oh[:, i:i + 1, ...].float(), flow.float()])
            #     x_segs.append(def_seg)
            # x_segs = torch.cat(x_segs, dim=1)
            # def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            # del x_segs, x_seg_oh
            
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            Jdet = np.sum(jac_det <= 0) / np.prod(tar.shape)
            eval_det.update(Jdet, x.size(0))
            print('det < 0: {}'.format(Jdet))
            dsc_trans = utils.dice_IXI(def_out.long(), y_seg.long()) if args.dataset == 'IXI' else utils.dice_OASIS(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_IXI(def_out.long(), y_seg.long()) if args.dataset == 'IXI' else utils.dice_OASIS(def_out.long(), y_seg.long())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}\n'.format(dsc_trans.item(),dsc_raw.item()))
            
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            
            write_csv(save_dir + csv_name, idx, dsc_trans.item(), Jdet)
            idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))


def write_csv(save_dir, idx, dsc, Jdet):
    # check if directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir, 'a') as f:
        f.write('{},{},{}\n'.format(idx, dsc, Jdet))
        

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

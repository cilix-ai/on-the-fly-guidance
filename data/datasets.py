import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import random
import numpy as np
import SimpleITK as sitk


class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        # data_path is a list
        data_path.sort()
        random.seed(42)
        self.paths = random.sample(data_path, 200)
        self.paths.sort()
        
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        # data_path is a list
        data_path.sort()
        random.seed(42)
        self.paths = random.sample(data_path, 20)
        self.paths.sort()
        
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    

class OASISBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        # data_path is a list
        data_path.sort()
        random.seed(42)
        self.paths = random.sample(data_path, 200)
        self.paths.sort()
        
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class OASISBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        # data_path is a list
        data_path.sort()
        random.seed(42)
        self.paths = random.sample(data_path, 19)
        self.paths.sort()
        
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class LPBADataset(Dataset):
    def __init__(self, paths, atlas_path, transforms):
        self.paths = paths
        self.atlas_path = atlas_path
        self.transforms = transforms
    
    def __getitem__(self, index):
        path = self.paths[index]
        x = sitk.GetArrayFromImage(sitk.ReadImage(path))[None, ...]
        y = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_path))[None, ...]
        x = np.ascontiguousarray(x) # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x = torch.from_numpy(x).cuda().float()
        y = torch.from_numpy(y).cuda().float()
        return x, y

    def __len__(self):
        return len(self.paths)


class LPBAInferDataset(Dataset):
    def __init__(self, paths, atlas_path, label_dir, transforms):
        self.paths = paths
        self.atlas_path = atlas_path
        self.label_dir = label_dir
        self.transforms = transforms
    
    def __getitem__(self, index):
        path = self.paths[index]
        name = os.path.split(path)[1]
        x = sitk.GetArrayFromImage(sitk.ReadImage(path))[None, ...]
        y = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_path))[None, ...]
        x = torch.from_numpy(x).cuda().float()
        y = torch.from_numpy(y).cuda().float()
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(self.label_dir, name[:3] + "*"))[0]))[None, ...]
        x_seg = torch.from_numpy(x_seg).cuda().float()
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.label_dir, "S01.delineation.structure.label.nii.gz")))

        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    
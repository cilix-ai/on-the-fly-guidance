import torch
import torch.nn as nn

from models.VoxelMorph import VxmDense_1
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph


class CascadeOpt_Vxm(nn.Module):
    def __init__(self, img_size, blk_num=2):
        super(CascadeOpt_Vxm, self).__init__()

        self.img_size = img_size
        self.blk_num = blk_num

        self.blocks = nn.ModuleList()
        for i in range(self.blk_num):
            self.blocks.append(VxmDense_1(img_size))
        
    def forward(self, x):
        y = x[:, 1:2, :, :]

        for i in range(self.blk_num):
            x_warped, flow = self.blocks[i](x)
            x = torch.cat([x_warped, y], dim=1)
        
        return x_warped, flow


class CascadeOpt_Trans(nn.Module):
    def __init__(self, img_size=(160, 192, 160), window_size=(5, 6, 5, 5), blk_num=2):
        super(CascadeOpt_Trans, self).__init__()

        config = CONFIGS_TM['TransMorph']

        config.img_size = img_size
        config.window_size = window_size

        self.blk_num = blk_num

        self.blocks = nn.ModuleList()
        for i in range(self.blk_num):
            self.blocks.append(TransMorph.TransMorph(config))
    
    def forward(self, x):
        y = x[:, 1:2, :, :]

        for i in range(self.blk_num):
            x_warped, flow = self.blocks[i](x)
            x = torch.cat([x_warped, y], dim=1)
        
        return x_warped, flow

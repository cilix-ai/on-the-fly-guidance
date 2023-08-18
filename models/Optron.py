import sys
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import utils

class optron(nn.Module):
    def __init__(self, initial_flow):
        self.flow = nn.Parameter(initial_flow.clone())
        
        img_size = (160, 192, 224)
        mode = 'bilinear'
        self.spatial_trans = utils.SpatialTransformer(img_size, mode)
        
    def forward(self, data):
        x, _ = data
        x_warped = self.spatial_trans(x, self.flow)
        return x_warped
        

import numpy as np
import torch.nn as nn
import utils

class Optron(nn.Module):
    def __init__(self, initial_flow):
        self.flow = nn.Parameter(initial_flow)
        
        img_size = (160, 192, 224)
        mode = 'bilinear'
        self.spatial_trans = utils.SpatialTransformer(img_size, mode)
        
    def forward(self, x):
        x_warped = self.spatial_trans(x, self.flow)
        return x_warped, self.flow

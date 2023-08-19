import numpy as np
import torch.nn as nn
import utils


class Optron(nn.Module):
    def __init__(self, initial_flow):
        """
        Optimization module for displacements field
        Used to provide pseudo ground truth for training

        Args:
            initial_flow (torch.Tensor): initial flow field
        """
        super(Optron, self).__init__()
        
        self.flow = nn.Parameter(initial_flow)
        self.img_size = (160, 192, 224)
        self.mode = 'bilinear'

        self.spatial_trans = utils.SpatialTransformer(self.img_size, self.mode)

    def forward(self, x):
        x_warped = self.spatial_trans(x, self.flow)
        return x_warped, self.flow

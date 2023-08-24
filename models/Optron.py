import numpy as np
import torch.nn as nn
import utils


class Optron(nn.Module):
    """
    Optimization module for displacements field
    Used to provide pseudo ground truth for training
    """
    def __init__(self, img_size, initial_flow):
        """
        Args:
            initial_flow (torch.Tensor): initial flow field
        """
        super(Optron, self).__init__()

        self.img_size = img_size
        self.mode = 'bilinear'

        self.flow = nn.Parameter(initial_flow)
        self.spatial_trans = utils.SpatialTransformer(self.img_size, self.mode)

    def forward(self, x):
        x_warped = self.spatial_trans(x, self.flow)
        return x_warped, self.flow

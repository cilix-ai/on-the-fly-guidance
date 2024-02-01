import torch.optim as optim
import torch.nn as nn
import utils.losses as losses
import utils.utils as utils


class OFG(nn.Module):
    """
    Optimization module for displacements field
    Used to provide pseudo ground truth for training
    """

    def __init__(self, img_size, initial_flow, mode='bilinear'):
        """
        Args:
            img_size (tuple): shape of the input image
            initial_flow (torch.Tensor): initial flow field
        """
        super(OFG, self).__init__()

        self.img_size = img_size
        self.mode = mode

        self.flow = nn.Parameter(initial_flow)
        self.spatial_trans = utils.SpatialTransformer(self.img_size, self.mode)

    def forward(self, x):
        x_warped = self.spatial_trans(x, self.flow)
        return x_warped, self.flow


class OFGLoss(nn.Module):
    """
    OFG loss function
    """
    def __init__(self, iter_count=5, reg_weight=1):
        """
        Args:
            iter_count (int): number of steps for optimization
            reg_weight (float): weight of regularization term
        """
        super(OFGLoss, self).__init__()
        self.iter_count = iter_count
        self.reg_weight = reg_weight
        self.ncc = losses.NCC_vxm()
        self.reg = losses.Grad3d(penalty='l2')
        self.mse = nn.MSELoss()

    def forward(self, x, y, initial_flow):
        """
        Args:
            x (torch.Tensor): moving image
            y (torch.Tensor): fixed image
            initial_flow (torch.Tensor): initial deformation field
        """
        _, _, H, W, D = x.shape
        img_size = (H, W, D)
        ofg = OFG(img_size, initial_flow)
        ofg_optimizer = optim.Adam(self.ofg.parameters(), lr=0.1, 
                                   weight_decay=0, amsgrad=True)

        for _ in range(self.iter_count):
            x_warped, optimized_flow = ofg(x)
            loss_ncc = self.ncc(x_warped, y) * 1
            loss_reg = self.reg(optimized_flow, y) * self.reg_weight
            loss = loss_ncc + loss_reg

            ofg_optimizer.zero_grad()
            loss.backward()
            ofg_optimizer.step()

        ofg_loss = self.mse(optimized_flow, initial_flow)

        return ofg_loss


if __name__ == '__main__':
    criterion_ofg = OFGLoss()

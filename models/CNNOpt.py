import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(ResBlock, self).__init__()

        self.expansion = 1

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)
        
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x

        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))

        out += shortcut

        return out


class CNNOpt(nn.Module):
    def __init__(self, img_size, in_channels, start_channels, out_channels):
        super(CNNOpt, self).__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.start_channels = start_channels
        self.out_channels = out_channels

        self.input_encoder = self.input_feature_extract(self.in_channels, self.start_channels * 4, bias=False)
        self.down_conv = nn.Conv3d(self.start_channels * 4, self.start_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.resblock_group = self.resblocks(self.start_channels * 4, bias_opt=False)
        self.up_conv = nn.ConvTranspose3d(self.start_channels * 4, self.start_channels * 4, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.output = self.outputs(self.start_channels * 8, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.spatial_trans = utils.SpatialTransformer(self.img_size, 'bilinear')

    def resblocks(self, in_channels, bias_opt=False):
        layers = nn.Sequential(
            ResBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            ResBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            ResBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            ResBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            ResBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layers
    
    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layers = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            )
        else:
            layers = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            )
        return layers
    
    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=False):
        if batchnorm:
            layers = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh()
            )
        else:
            layers = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels / 2), kernel_size, stride, padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels / 2), out_channels, kernel_size, stride, padding, bias=bias),
                nn.Softsign()
            )
        return layers

    def forward(self, x):
        source = x[:, 0:1, ...]
        fea = self.input_encoder(x)

        e = self.down_conv(fea)
        e = self.resblock_group(e)
        e = self.up_conv(e)

        out = self.output(torch.cat([fea, e], dim=1))

        x_warped = self.spatial_trans(source, out)

        return x_warped, out

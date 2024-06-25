import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1,
                                 bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x




c = 16


class OriUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = Conv2(17, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)


class Unet(nn.Module):
    def __init__(self, hidden_dim=32, shift_dim=32):
        # forward_shiftedFeature, backward_shftedFeature, forwardContext, backwardContext
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shift_dim = shift_dim
        self.hs_dim = self.hidden_dim + self.shift_dim  # 64
        self.down0 = Conv2(in_planes=self.hs_dim * 2, out_planes=self.hs_dim)  # *2 for ori_f0/f1 32+32
        self.down1 = Conv2(self.hs_dim * 2, self.hs_dim * 2)    # 64*2 --> 64*2
        self.down2 = Conv2(self.hs_dim * 4, self.hs_dim * 4)    # 64*4 --> 64*4
        self.down3 = Conv2(self.hs_dim * 4, self.hs_dim * 4)
        self.up0 = deconv(self.hs_dim * 4, self.hs_dim * 4)
        self.up1 = deconv(self.hs_dim * 8, self.hs_dim * 2)
        self.up2 = deconv(self.hs_dim * 4, self.hs_dim * 1)
        self.up3 = deconv(self.hs_dim * 2, self.hs_dim * 2)
        self.conv = nn.Conv2d(self.hs_dim * 2, 3, 3, 1, 1)

    def forward(self, ori_f0_features, ori_f1_features, forward_shiftedFeature, backward_shiftedFeature,
                forwardContext_d2, forwardContext_d4, backwardContext_d2, backwardContext_d4):
        s0 = self.down0(torch.cat([ori_f0_features, ori_f1_features, forward_shiftedFeature, backward_shiftedFeature],
                                  dim=1))  # (32+32 )*2
        s1 = self.down1(torch.cat([s0, forwardContext_d2, backwardContext_d2], dim=1))  # 64 + (32+32)
        s2 = self.down2(torch.cat([s1, forwardContext_d4, backwardContext_d4], dim=1))  # 128 + (64+64)
        s3 = self.down3(s2)
        x = self.up0(s3)
        x = self.up1(torch.cat([x, s2], 1))
        x = self.up2(torch.cat([x, s1], 1))
        x = self.up3(torch.cat([x, s0], 1))
        x = self.conv(x)

        return torch.sigmoid(x)


class Unet_for_3Pyramid(nn.Module):
    def __init__(self, hidden_dim=32, shift_dim=32):
        """
        forward_shiftedFeature, backward_shftedFeature, forwardContext, backwardContext
        The tensor dimensions of pyramid are [c * H * W, 2c * H // 2 * W//2, 4c *H // 4 * W//4], where c is hidden_dim
        The input dimension of warped feature is shift_dim* H * W
        In current image pyramid
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shift_dim = shift_dim
        self.hs_dim = self.hidden_dim + self.shift_dim  # 64
        self.downDimension = nn.Conv2d(self.hs_dim * 2, self.hs_dim, 3,1,1)
        self.startBlock = nn.Conv2d(in_channels=self.hs_dim*2, out_channels=self.hs_dim, kernel_size=3, stride=1, padding=1)  #hs*2 -> hs*2 = 128
        self.simplified_startBlock = None
        self.down1 = Conv2(self.hs_dim * 2, self.hs_dim * 2)    # hs*2  == 64 + (32*2) = 128
        self.down2 = Conv2(self.hs_dim * 4, self.hs_dim * 4)    # hs*4  == 256 + (128*2) = 256
        self.down3 = Conv2(self.hs_dim * 8, self.hs_dim * 8)    # hs*8  == 512 + (256*2) = 512
        self.up0 = deconv(self.hs_dim * 8, self.hs_dim * 8)     # hs*8 => hs*4
        self.up1 = deconv(self.hs_dim * (8+4), self.hs_dim * 4)     # hs*4 => hs*2
        self.up2 = deconv(self.hs_dim * (4+2), self.hs_dim * 2)
        self.endBlock = nn.Conv2d(self.hs_dim * (2+1), self.hs_dim * 2, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(self.hs_dim * 2, 3, 3, 1, 1)

    def forward(self, ori_f0_features, ori_f1_features, forward_shiftedFeature, backward_shiftedFeature, forwardContext_d0,
                forwardContext_d2, forwardContext_d4, backwardContext_d0, backwardContext_d2, backwardContext_d4):
        s0 = self.startBlock(torch.cat([ori_f0_features, ori_f1_features, forward_shiftedFeature, backward_shiftedFeature],
                                       dim=1))  # (32+32 )*2 *H * W
        s1 = self.down1(torch.cat([s0, forwardContext_d0, backwardContext_d0], dim=1))  # 64 + (32+32)  cat
        s2 = self.down2(torch.cat([s1, forwardContext_d2, backwardContext_d2], dim=1))  # 128 + (64+64) cat H//2 * W//2 => H//4 W//4
        s3 = self.down3(torch.cat([s2, forwardContext_d4, backwardContext_d4], dim=1))  # 256 + (128+128) cat H//4 * W//4 => H//8 W//8
        x = self.up0(s3) # H//8 * W //8 ==> H//4 * W//4
        x = self.up1(torch.cat([x, s2], 1))  # H//4 ==> H//2
        x = self.up2(torch.cat([x, s1], 1))  # H//2 ==> H
        x = self.endBlock(torch.cat([x, s0], 1))  # H ==> H
        x = self.conv(x)

        return torch.sigmoid(x)

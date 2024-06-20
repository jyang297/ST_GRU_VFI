import torch
import torch.nn as nn
from model.Pyramid import FeaturePyramid as FPyramid
from model.Pyramid import ImagePyramid as IPyramid


class PyramidFBwardExtractor(nn.Module):
    # input 3, output 6
    def __init__(self, in_plane=3, hidden_pyramid=128, pyramid="image"):
        super().__init__()
        # set stride = 2 to downsample
        # as for 224*224, the current output shape is 112*112
        self.hidden_pyramid = hidden_pyramid
        self.in_plane = in_plane
        if pyramid == "image":
            self.pyramid = IPyramid(c=self.hidden_pyramid)
        elif pyramid == "feature":
            self.pyramid = FPyramid(c=self.hidden_pyramid)  # pyramid d1: c*h/2*w/2, ; pyramid 2: 2c * h/4* d/4

    def forward(self, allframes, flag_st='stu', pyramid="image"):
        # all frames: B*21*H*W  -->
        # x is concated frames [0,2,4,6] -> [(4*3),112,112]

        forwardFeatureList_d0 = []
        forwardFeatureList_d2 = []
        forwardFeatureList_d4 = []
        if flag_st == 'stu':
            range_Frames = 4
            skip_Frames = 6
        else:
            range_Frames = 7
            skip_Frames = 3
        for i in range(0, range_Frames):
            x = allframes[:, skip_Frames * i:skip_Frames * i + 3].clone()

            if pyramid == "image":
                y_d0, y_d2, y_d4 = self.pyramid(x)
                forwardFeatureList_d0.append(y_d0)
                forwardFeatureList_d2.append(y_d2)
                forwardFeatureList_d4.append(y_d4)

            elif pyramid == "feature":
                y_d2, y_d4 = self.pyramid(x)
                forwardFeatureList_d2.append(y_d2)
                forwardFeatureList_d4.append(y_d4)
        if pyramid == "image":    
            return forwardFeatureList_d0, forwardFeatureList_d2, forwardFeatureList_d4
        elif pyramid == "feature":
            return forwardFeatureList_d2, forwardFeatureList_d4

    # Output: B*N*C*H*W


class unitConvGRU(nn.Module):
    # Formula:
    # I_t = Sigmoid(Conv(x_t;W_{xi}) + Conv(h_{t-1};W_{hi}) + b_i)
    # F_t = Sigmoid(Conv(x_t;W_{xf}) + Conv(h_{t-1};W_{hi}) + b_i)
    def __init__(self, hidden_dim=128, input_dim=128):
        # 192 = 4*4*12
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


def Gru_ini_Image_resol(hidden_dimension, image):
    h, w = image.size()[-2:]
    b = image.size()[0]
    device = image.device()

    hidden_tensor = torch.zeros((b, hidden_dimension, h, w), device=device)

    return hidden_tensor
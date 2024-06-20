import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.Attenions import CBAM

class SecondFrame_Fusion(nn.Module):
    def __init__(self, fusion_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels + out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.att = CBAM(out_channels)


    def forward(self, temporal_features, optical_flow, current_frame_feature):
        warped_fea = warp(temporal_features, optical_flow)
        fusion_feature = self.fusion(warped_fea, current_frame_feature)
        res_fusion = fusion_feature + current_frame_feature
        temporal_next = nn.LeakyReLU(res_fusion)
        att_fusion = self.att(res_fusion)

        return temporal_next, att_fusion






import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramid(nn.Module):
    """Two-level feature pyramid
    1) remove high-level feature pyramid (compared to PWC-Net), and add more conv layers to stage 2;
    2) do not increase the output channel of stage 2, in order to keep the cost of corr volume under control.
    """

    def __init__(self, c=24):
        super().__init__()
        self.c = c
        self.conv_stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))
        self.conv_stage2 = nn.Sequential(
            nn.Conv2d(in_channels=self.c, out_channels=2 * self.c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=2 * self.c, out_channels=2 * self.c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=2 * self.c, out_channels=2 * self.c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=2 * self.c, out_channels=2 * self.c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=2 * self.c, out_channels=2 * self.c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=2 * self.c, out_channels=2 * self.c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, img):
        pyramid_layer_0 = self.conv_stage1(img)
        pyramid_layer_1 = self.conv_stage2(pyramid_layer_0)

        return [pyramid_layer_0, pyramid_layer_1]


class ImagePyramid(nn.Module):
    """
    Three level pyramid. The output size depends on the input parameter c.
    The output shapes are [ c * H * W, 2c * H/2 * W/2, 4c * H/2 * W/2] where H and W are image shape
    """
    def __init__(self, c=24):
        super().__init__()
        self.c = c
        self.conv_stage1 = self._make_conv_stage(out_channels=self.c)       # self.c = 24
        self.conv_stage2 = self._make_conv_stage(out_channels=self.c*2)     # 2* self.c = 48
        self.conv_stage3 = self._make_conv_stage(out_channels=self.c*4)     # 4* self.c = 96

    @staticmethod
    def _make_conv_stage(out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

    def forward(self, img):
        pyramid_layer_0 = self.conv_stage1(img)
        down_image_1 = F.interpolate(img, scale_factor=0.5, mode="bilinear", align_corners=False,
                                     recompute_scale_factor=False) * 0.5

        pyramid_layer_1 = self.conv_stage2(down_image_1)

        down_image_2 = F.interpolate(down_image_1, scale_factor=0.5, mode="bilinear", align_corners=False,
                                     recompute_scale_factor=False) * 0.5
        pyramid_layer_2 = self.conv_stage3(down_image_2)

        return [pyramid_layer_0, pyramid_layer_1, pyramid_layer_2]

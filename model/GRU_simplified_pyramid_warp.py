import model.laplacian as modelLap
from model.warplayer import warp
from model.refine import *
from model.loss import *
from model.myLossset import *
from model.ConvGRU import PyramidFBwardExtractor as Pyramid_direction_extractor
from model.ConvGRU import unitConvGRU
# Attention test
import model.Attenions as att
from model.myLossset import CensusLoss as census
import model.STloss as ST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c = 48


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False,
                          recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False)
        return flow, mask


class downHidden_attention(nn.Module):
    def __init__(self, hidden, hidden_down, att_mode="se"):
        super().__init__()
        self.hidden = hidden
        self.hidden_down = hidden_down
        self.down = conv(self.hidden, self.hidden_down)
        if att_mode == 'se':
            self.attention = att.SELayer(channel=self.hidden_down, reduction=16, pool_mode='avg')

            #  self.attention_d2 = att.SELayer(channel=self.hidden_down//2, reduction=16,pool_mode='avg')
            # self.attention_d4 = att.SELayer(channel=self.out_plane, reduction=16,pool_mode='avg')
        elif att_mode == 'cbam':
            self.attention = att.CBAM(in_channel=self.out_plane, ratio=4, kernel_size=7)
        elif att_mode == 'none':
            self.attention = nn.Sequential()

    def forward(self, x):
        x = self.down(x)
        x = self.attention(x)
        return x


# c = 48
class ConvGRUFeatures(nn.Module):
    """
    Student ConvGRU feature extractor
    """

    def __init__(self, hidden_dim=64, pyramid="image"):
        super().__init__()
        # current encoder: all frames ==> all features
        self.pyramid = pyramid
        self.hidden_dim = hidden_dim
        self.output_plane = self.hidden_dim //2
        self.img2Fencoder = Pyramid_direction_extractor(in_plane=3, hidden_pyramid=self.hidden_dim,
                                                        pyramid=self.pyramid)
        self.img2Bencoder = Pyramid_direction_extractor(in_plane=3, hidden_pyramid=self.hidden_dim,
                                                        pyramid=self.pyramid)
        self.hidden_dim_d0 = hidden_dim
        self.hidden_dim_d2 = hidden_dim * 2
        self.hidden_dim_d4 = hidden_dim * 4
        self.forwardgru_d0 = unitConvGRU(hidden_dim=self.hidden_dim_d0, input_dim=self.hidden_dim_d0)
        self.backwardgru_d0 = unitConvGRU(hidden_dim=self.hidden_dim_d0, input_dim=self.hidden_dim_d0)
        self.forwardgru_d2 = unitConvGRU(hidden_dim=self.hidden_dim_d2, input_dim=self.hidden_dim_d2)
        self.backwardgru_d2 = unitConvGRU(hidden_dim=self.hidden_dim_d2, input_dim=self.hidden_dim_d2)
        self.forwardgru_d4 = unitConvGRU(hidden_dim=self.hidden_dim_d4, input_dim=self.hidden_dim_d4)
        self.backwardgru_d4 = unitConvGRU(hidden_dim=self.hidden_dim_d4, input_dim=self.hidden_dim_d4)

        self.fdown_c = downHidden_attention(self.hidden_dim, self.output_plane)
        self.fdown_2c = downHidden_attention(self.hidden_dim * 2, 2 * self.output_plane)

        self.bdown_c = downHidden_attention(self.hidden_dim, self.output_plane)
        self.bdown_2c = downHidden_attention(self.hidden_dim * 2, 2 * self.output_plane)

        if pyramid == "image":
            self.fdown_4c = downHidden_attention(self.hidden_dim * 4, 4 * self.output_plane)
            self.bdown_4c = downHidden_attention(self.hidden_dim * 4, 4 * self.output_plane)

    def forward(self, allframes):
        # aframes = allframes_N.view(b,n*c,h,w)
        # Output: BNCHW
        fcontextlist_d0 = []  # c = 64
        bcontextlist_d0 = []  # c = 64
        fcontextlist_d2 = []  # 2c = 128
        bcontextlist_d2 = []  # 2c = 128
        fcontextlist_d4 = []  # 4c = 256
        bcontextlist_d4 = []  # 4c = 256
        if self.pyramid == "image":
            fallfeatures_d0, fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes, pyramid="image")
            ballfeatures_d0, ballfeatures_d2, ballfeatures_d4 = self.img2Bencoder(allframes, pyramid="image")
        elif self.pyramid == "feature":
            fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes)
            ballfeatures_d2, ballfeatures_d4 = self.img2Bencoder(allframes)
        b, _, h, w = allframes.size()

        # forward GRU
        # Method A: zero initialize Hiddenlayer
        if self.pyramid == "image":
            forward_hidden_initial_d0 = torch.zeros((b, self.hidden_dim_d0, h, w), device=device)
            backward_hidden_initial_d0 = torch.zeros((b, self.hidden_dim_d0, h, w), device=device)
        forward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h // 2, w // 2), device=device)
        backward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h // 2, w // 2), device=device)
        forward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h // 4, w // 4), device=device)
        backward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h // 4, w // 4), device=device)
        # n=4
        # I skipped the 0 -> first image

        if self.pyramid == "image":
            for i in range(0, 4):
                if i == 0:
                    # for d0 layer
                    fhidden = self.forwardgru_d0(forward_hidden_initial_d0, fallfeatures_d0[i])
                    bhidden = self.backwardgru_d0(backward_hidden_initial_d0, ballfeatures_d0[-i - 1])

                else:
                    fhidden = self.forwardgru_d0(fhidden, fallfeatures_d0[i])
                    bhidden = self.backwardgru_d0(bhidden, ballfeatures_d0[-i - 1])
                    fhidden_down = self.fdown_c(fhidden)
                    bhidden_down = self.bdown_c(bhidden)
                    fcontextlist_d0.append(fhidden_down)
                    bcontextlist_d0.append(bhidden_down)

        for i in range(0, 4):
            if i == 0:
                # for d2 layer
                fhidden = self.forwardgru_d2(forward_hidden_initial_d2, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(backward_hidden_initial_d2, ballfeatures_d2[-i - 1])
            else:
                fhidden = self.forwardgru_d2(fhidden, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(bhidden, ballfeatures_d2[-i - 1])
                fhidden_down = self.fdown_2c(fhidden)
                bhidden_down = self.bdown_2c(bhidden)
                fcontextlist_d2.append(fhidden_down)
                bcontextlist_d2.append(bhidden_down)
        for i in range(0, 4):
            if i == 0:
                # for d4 layer
                fhidden = self.forwardgru_d4(forward_hidden_initial_d4, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(backward_hidden_initial_d4, ballfeatures_d4[-i - 1])
            else:
                fhidden = self.forwardgru_d4(fhidden, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(bhidden, ballfeatures_d4[-i - 1])
                fhidden_down = self.fdown_4c(fhidden)
                bhidden_down = self.bdown_4c(bhidden)
                fcontextlist_d4.append(fhidden_down)
                bcontextlist_d4.append(bhidden_down)
        if self.pyramid == "image":
            return fcontextlist_d0, fcontextlist_d2, fcontextlist_d4, bcontextlist_d0, bcontextlist_d2, bcontextlist_d4
        elif self.pyramid == "feature":
            return fcontextlist_d2, fcontextlist_d4, bcontextlist_d2, bcontextlist_d4
        # return forwardFeature, backwardFeature
        # Now iterate through septuplet and get three inter frames


class TeacherConvGRUFeatures(ConvGRUFeatures):
    def __init__(self, hidden_dim=64, output_plane=32, pyramid="image"):
        super().__init__()

    def forward(self, allframes):
        # aframes = allframes_N.view(b,n*c,h,w)
        # Output: BNCHW
        b, _, h, w = allframes.size()
        tea_fcontextlist_d0 = []  # c = 64
        tea_bcontextlist_d0 = []  # c = 64
        tea_fcontextlist_d2 = []  # 2c
        tea_bcontextlist_d2 = []  # 2c
        tea_fcontextlist_d4 = []  # 4c
        tea_bcontextlist_d4 = []  # 4c

        if self.pyramid == "image":
            fallfeatures_d0, fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes, pyramid="image",
                                                                                  flag_st='tea')
            ballfeatures_d0, ballfeatures_d2, ballfeatures_d4 = self.img2Bencoder(allframes, pyramid="image",
                                                                                  flag_st='tea')
        elif self.pyramid == "feature":
            fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes, flag_st='tea')
            ballfeatures_d2, ballfeatures_d4 = self.img2Bencoder(allframes, flag_st='tea')

        forward_hidden_initial_d0 = torch.zeros((b, self.hidden_dim_d0, h, w), device=device)
        backward_hidden_initial_d0 = torch.zeros((b, self.hidden_dim_d0, h, w), device=device)
        forward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h // 2, w // 2), device=device)
        backward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h // 2, w // 2), device=device)
        forward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h // 4, w // 4), device=device)
        backward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h // 4, w // 4), device=device)
   
        if self.pyramid == "image":
            for i in range(0, 7):
                if i == 0:
                    fhidden = self.forwardgru_d0(forward_hidden_initial_d0, fallfeatures_d0[i])
                    bhidden = self.backwardgru_d0(backward_hidden_initial_d0, ballfeatures_d0[-i - 1])
                else:
                    fhidden = self.forwardgru_d0(fhidden, fallfeatures_d0[i])
                    bhidden = self.backwardgru_d0(bhidden, ballfeatures_d0[-i - 1])
                    if i % 2 == 0:
                        fhidden_down = self.fdown_c(fhidden)
                        bhidden_down = self.bdown_c(bhidden)
                        tea_fcontextlist_d0.append(fhidden_down)
                        tea_bcontextlist_d0.append(bhidden_down)
        for i in range(0, 7):
            if i == 0:
                fhidden = self.forwardgru_d2(forward_hidden_initial_d2, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(backward_hidden_initial_d2, ballfeatures_d2[-i - 1])
            else:
                fhidden = self.forwardgru_d2(fhidden, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(bhidden, ballfeatures_d2[-i - 1])
                if i % 2 == 0:
                    fhidden_down = self.fdown_2c(fhidden)
                    bhidden_down = self.bdown_2c(bhidden)
                    tea_fcontextlist_d2.append(fhidden_down)
                    tea_bcontextlist_d2.append(bhidden_down)

        for i in range(0, 7):
            if i == 0:
                fhidden = self.forwardgru_d4(forward_hidden_initial_d4, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(backward_hidden_initial_d4, ballfeatures_d4[-i - 1])
            else:
                fhidden = self.forwardgru_d4(fhidden, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(bhidden, ballfeatures_d4[-i - 1])
                if i % 2 == 0:
                    fhidden_down_down = self.fdown_4c(fhidden)
                    bhidden_down_down = self.bdown_4c(bhidden)
                    tea_fcontextlist_d4.append(fhidden_down_down)
                    tea_bcontextlist_d4.append(bhidden_down_down)

        return tea_fcontextlist_d0, tea_fcontextlist_d2, tea_fcontextlist_d4, tea_bcontextlist_d0, tea_bcontextlist_d2, tea_bcontextlist_d4
        # return forwardFeature, backwardFeature
        # Now iterate through septuplet and get three inter frames


class SingleImageExtractor(nn.Module):
    def __init__(self, in_plane=3, out_plane=32, att_mode='se'):
        super().__init__()
        # set stride = 2 to downsample
        # as for 224*224, the current output shape is 112*112
        self.out_plane = out_plane
        self.fromimage = conv(in_plane, 32, kernel_size=3, stride=1, padding=1)
        self.downsample = conv(32, 2 * 32, kernel_size=3, stride=1, padding=1)

        self.conv0 = nn.Sequential(
            conv(2 * 32, 2 * 32, 3, 1, 1),
            conv(2 * 32, 4 * 32, 3, 1, 1),
            conv(4 * 32, 4 * 32, 3, 1, 1),
            conv(4 * 32, self.out_plane, 3, 1, 1),
        )

    def forward(self, single):
        s = self.fromimage(single)
        s = self.downsample(s)  # not downsample for now
        s = self.conv0(s)
        s = torch.tanh(s)
        return s


class Loaded_Modified_IFNet(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        '''
        self.block0 = IFBlock(6+1, c=240)
        self.block1 = IFBlock(13+4+1, c=150)
        self.block2 = IFBlock(13+4+1, c=90)
        self.block_tea = IFBlock(16+4+1, c=90)
        self.contextnet = Contextnet()
        '''
        # Notice that the training start at Contextnet

        self.block0 = pretrained_model.block0
        self.block1 = pretrained_model.block1
        self.block2 = pretrained_model.block2
        self.block_tea = pretrained_model.block_tea
        self.InterpolationEncoder = SingleImageExtractor()

        # self.contextnet = pretrained_model.contextnet()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, scale_list=[4, 2, 1]):
        # forwardContext/backwardContext is forwardFeature[i], only pick up the one for the current interpolation
        # final_merged, loss = self.mIFnetframesset[i], for(wardFeature[3*i], backwardFeature[3*i+2])
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        # stdv = np.random.uniform(0.0, 5.0)
        # img0 = (img0 + stdv * torch.randn(*img0.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp
        # img1 = (img1 + stdv * torch.randn(*img1.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp       

        loss_tea_pred = 0
        merged_features = []
        ori_features = []
        # eps = 1e-8
        # ----------------

        flow_list = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])

            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1),
                              torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            # merged.append((warped_img0, warped_img1))
        interframe_f0 = self.InterpolationEncoder(img0)
        shifted_interframe_f0 = warp(interframe_f0, flow[:, :2] * 0.5)
        merged_features.append(shifted_interframe_f0)
        ori_features.append(interframe_f0)
        interframe_f1 = self.InterpolationEncoder(img1)
        shifted_interframe_f1 = warp(interframe_f1, flow[:, 2:4] * 0.5)
        merged_features.append(shifted_interframe_f1)
        ori_features.append(interframe_f1)

        return merged_features, ori_features


# ----------------


class newMergeIFnet(nn.Module):
    def __init__(self, pretrained_model, shift_dim=32, hidden_dim=32, pyramid="image"):
        super().__init__()
        self.pyramid = pyramid
        self.hidden_dim = hidden_dim
        self.shift_dim = shift_dim
        self.lap = modelLap.LapLoss()
        self.feature_ofnet = Loaded_Modified_IFNet(pretrained_model=pretrained_model)
        if self.pyramid == "image":
            self.unet_0to1 = Unet_for_3Pyramid(hidden_dim=self.hidden_dim, shift_dim=self.shift_dim)
        elif self.pyramid == "feature":
            self.unet_0to1 = Unet(hidden_dim=self.hidden_dim, shift_dim=self.shift_dim)
        self.decoder = nn.Sequential()
        self.epsilon = 1e-6
        self.loss_census = census()

    def forward(self, x,
                forwardContext_d0=0, forwardContext_d2=0, forwardContext_d4=0, backwardContext_d0=0,
                backwardContext_d2=0, backwardContext_d4=0,
                tea_forwardContext_d0=0, tea_forwardContext_d2=0, tea_forwardContext_d4=0, tea_backwardContext_d0=0,
                tea_backwardContext_d2=0, tea_backwardContext_d4=0,
                pyramid="image", training=True):

        gt = x[:, 6:]  # In inference time, gt is None
        merged_features, ori_features = self.feature_ofnet(x)  # 32*h*w
        forward_shiftedFeature, backward_shiftedFeature = merged_features
        ori_f0_features, ori_f1_features = ori_features
        if pyramid == "image":
            featureUnet = self.unet_0to1(ori_f0_features, ori_f1_features, forward_shiftedFeature,
                                         backward_shiftedFeature,
                                         forwardContext_d0, forwardContext_d2, forwardContext_d4, backwardContext_d0,
                                         backwardContext_d2, backwardContext_d4)

        elif pyramid == "feature":
            featureUnet = self.unet_0to1(ori_f0_features, ori_f1_features, forward_shiftedFeature,
                                         backward_shiftedFeature,
                                         forwardContext_d2, forwardContext_d4, backwardContext_d2, backwardContext_d4)

        predictimage = self.decoder(featureUnet)
        if training:
            if pyramid == "image":
                tea_featureUnet = self.unet_0to1(

                    ori_f0_features, ori_f1_features, forward_shiftedFeature,
                    backward_shiftedFeature,
                    tea_forwardContext_d0, tea_forwardContext_d2, tea_forwardContext_d4,
                    tea_backwardContext_d0, tea_backwardContext_d2, tea_backwardContext_d4)
            elif pyramid == "feature":
                tea_featureUnet = self.unet_0to1(ori_f0_features, ori_f1_features, forward_shiftedFeature,
                                                 backward_shiftedFeature, tea_forwardContext_d2, tea_forwardContext_d4,
                                                 tea_backwardContext_d2, tea_backwardContext_d4)
            tea_predictimage = self.decoder(tea_featureUnet)

        if training:
            loss_tea_pred = torch.mean(
                torch.sqrt(torch.pow((tea_predictimage - gt), 2) + self.epsilon ** 2)) + self.loss_census(
                tea_predictimage, gt)
        else:
            loss_tea_pred = 0

        loss_pred = torch.mean(torch.sqrt(torch.pow((predictimage - gt), 2) + self.epsilon ** 2)) + self.loss_census(
            predictimage, gt)
        loss_mse = ((predictimage - gt) ** 2).detach()
        loss_mse = loss_mse.mean()

        # loss_tea = 0
        merged_teacher = tea_predictimage  # not used. just to avoid error
        flow_teacher = predictimage * 0  # not used. just to avoid error
        mask_list = [flow_teacher, flow_teacher, flow_teacher]
        flow_list = [flow_teacher, flow_teacher, flow_teacher]
        return flow_list, merged_teacher, predictimage, flow_teacher, merged_teacher, loss_tea_pred, loss_mse, loss_pred


class VSRbackbone(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.pyramid = "image"
        self.feature_ofnet = newMergeIFnet(shift_dim=32, pretrained_model=pretrained)
        self.hidden = 64
        self.Stu_ConvGRU = ConvGRUFeatures(hidden_dim=self.hidden, pyramid=self.pyramid)
        self.tea_ConvGRU = TeacherConvGRUFeatures(hidden_dim=self.hidden, pyramid=self.pyramid)

    def forward(self, allframes, training_flag=True):
        # allframes 0<1>2<3>4<5>6

        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_tea_pred = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_mse = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')

        output_allframes = []
        output_teacher = []
        flow_list = []
        mask_list = []
        if self.pyramid == "feature":
            fallfeatures_2d, fallfeatures_4d, ballfeatures_2d, ballfeatures_4d = self.Stu_ConvGRU(allframes)
            if training_flag:
                flow_teacher_list = []
                output_onlyteacher = []
                tea_fallfeatures_2d, tea_fallfeatures_4d, tea_ballfeatures_2d, tea_ballfeatures_4d = self.tea_ConvGRU(
                    allframes)
            f2d = f4d = b2d = b4d = 0
            for i in range(0, 3):
                f2d += ST.mse(fallfeatures_2d[i], tea_fallfeatures_2d[i])
                f4d += ST.mse(fallfeatures_4d[i], tea_fallfeatures_4d[i])
                b2d += ST.mse(ballfeatures_2d[i], tea_ballfeatures_2d[i])
                b4d += ST.mse(ballfeatures_4d[i], tea_ballfeatures_4d[i])
            loss_dist = 0.5 * (f2d + b2d) + (f4d + b4d)
        elif self.pyramid == "image":
            fallfeatures_0d, fallfeatures_2d, fallfeatures_4d, ballfeatures_0d, ballfeatures_2d, ballfeatures_4d = self.Stu_ConvGRU(
                allframes)
            if training_flag:
                flow_teacher_list = []
                output_onlyteacher = []
                tea_fallfeatures_0d, tea_fallfeatures_2d, tea_fallfeatures_4d, tea_ballfeatures_0d, tea_ballfeatures_2d, tea_ballfeatures_4d = self.tea_ConvGRU(
                    allframes)
            f0d = b0d = f2d = f4d = b2d = b4d = 0
            for i in range(0, 3):
                f0d += ST.mse(fallfeatures_0d[i], tea_fallfeatures_0d[i])
                b0d += ST.mse(ballfeatures_0d[i], tea_ballfeatures_0d[i])
                f2d += ST.mse(fallfeatures_2d[i], tea_fallfeatures_2d[i])
                f4d += ST.mse(fallfeatures_4d[i], tea_fallfeatures_4d[i])
                b2d += ST.mse(ballfeatures_2d[i], tea_ballfeatures_2d[i])
                b4d += ST.mse(ballfeatures_4d[i], tea_ballfeatures_4d[i])
            loss_dist = (f0d + b0d) + 0.5 * (f2d + b2d) + 0.25 * (f4d + b4d)
        for i in range(0, 3, 1):
            img0 = allframes[:, 6 * i:6 * i + 3]
            gt = allframes[:, 6 * i + 3:6 * i + 6]
            img1 = allframes[:, 6 * i + 6:6 * i + 9]
            imgs_and_gt = torch.cat([img0, img1, gt], dim=1)
            if training_flag:
                if self.pyramid == "image":
                    flow, mask, merged, flow_teacher, merged_teacher, loss_tea_pred, loss_mse, loss_pred = self.feature_ofnet(
                        x=imgs_and_gt,
                        forwardContext_d0=fallfeatures_0d[i], forwardContext_d2=fallfeatures_2d[i],
                        forwardContext_d4=fallfeatures_4d[i],
                        backwardContext_d0=ballfeatures_0d[-(i + 1)], backwardContext_d2=ballfeatures_2d[-(i + 1)],
                        backwardContext_d4=ballfeatures_4d[-(i + 1)],
                        tea_forwardContext_d0=tea_fallfeatures_0d[i], tea_forwardContext_d2=tea_fallfeatures_2d[i],
                        tea_forwardContext_d4=tea_fallfeatures_4d[i],
                        tea_backwardContext_d0=tea_ballfeatures_0d[-(i + 1)],
                        tea_backwardContext_d2=tea_ballfeatures_2d[-(i + 1)],
                        tea_backwardContext_d4=tea_ballfeatures_4d[-(i + 1)],
                        pyramid="image", training=training_flag)
                elif self.pyramid == "feature":
                    flow, mask, merged, flow_teacher, merged_teacher, loss_tea_pred, loss_mse, loss_pred = self.feature_ofnet(
                        x=imgs_and_gt,
                        forwardContext_d0=0, forwardContext_d2=fallfeatures_2d[i],
                        forwardContext_d4=fallfeatures_4d[i],
                        backwardContext_d0=0, backwardContext_d2=ballfeatures_2d[-(i + 1)],
                        backwardContext_d4=ballfeatures_4d[-(i + 1)],
                        tea_forwardContext_d0=0, tea_forwardContext_d2=tea_fallfeatures_2d[i],
                        tea_forwardContext_d4=tea_fallfeatures_4d[i],
                        tea_backwardContext_d0=0,
                        tea_backwardContext_d2=tea_ballfeatures_2d[-(i + 1)],
                        tea_backwardContext_d4=tea_ballfeatures_4d[-(i + 1)],
                        pyramid="feature", training=training_flag)
            else:
                if self.pyramid == "image":
                    flow, mask, merged, flow_teacher, merged_teacher, loss_tea_pred, loss_mse, loss_pred = self.feature_ofnet(
                        x=imgs_and_gt,
                        forwardContext_d0=fallfeatures_0d[i], forwardContext_d2=fallfeatures_2d[i],
                        forwardContext_d4=fallfeatures_4d[i],
                        backwardContext_d0=ballfeatures_0d[-(i + 1)], backwardContext_d2=ballfeatures_2d[-(i + 1)],
                        backwardContext_d4=ballfeatures_4d[-(i + 1)],
                        pyramid="image", training=training_flag)
                elif self.pyramid == "feature":
                    flow, mask, merged, flow_teacher, merged_teacher, loss_tea_pred, loss_mse, loss_pred = self.feature_ofnet(
                        x=imgs_and_gt,
                        forwardContext_d0=0, forwardContext_d2=fallfeatures_2d[i],
                        forwardContext_d4=fallfeatures_4d[i],
                        backwardContext_d0=0, backwardContext_d2=ballfeatures_2d[-(i + 1)],
                        backwardContext_d4=ballfeatures_4d[-(i + 1)],
                        pyramid="feature", training=training_flag)

            # flow, mask, merged, flow_teacher, merged_teacher, loss_tea_pred = self.flownet(allframes)
            Sum_loss_tea_pred += loss_tea_pred
            Sum_loss_context += loss_pred
            Sum_loss_mse += loss_mse
            Sum_loss_tea_pred += loss_tea_pred
            output_allframes.append(img0)
            output_teacher.append(img0)
            # output_allframes.append(merged[2])
            output_teacher.append(merged_teacher)
            output_allframes.append(merged)
            flow_list.append(flow)
            flow_teacher_list.append(flow_teacher)
            output_onlyteacher.append(merged_teacher)
            mask_list.append(mask)

            # The way RIFE compute prediction loss and 
            # loss_l1 = (self.lap(merged[2], gt)).mean()
            # loss_tea = (self.lap(merged_teacher, gt)).mean()

        img6 = allframes[:, -3:]
        output_allframes.append(img6)
        output_teacher.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        output_teacher_tensors = torch.stack(output_teacher, dim=1)


        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_teacher_tensors, Sum_loss_tea_pred, Sum_loss_context, Sum_loss_mse, loss_dist

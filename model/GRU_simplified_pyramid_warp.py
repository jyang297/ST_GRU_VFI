import model.laplacian as modelLap
from model.warplayer import warp
from model.refine import Unet_for_3Pyramid as unet3P
from model.loss import *
from model.myLossset import *
from model.ConvGRU import PyramidFBwardExtractor as Pyramid_direction_extractor
from model.ConvGRU import unitConvGRU
# Attention test
import model.Attenions as att
from model.myLossset import CensusLoss as census
import model.STloss as ST
from model.motionWarp import TwoWarpSecondFrame_Fusion, OneWarpSecondFrame_Fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c = 48


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )



class SimplifiedWarp_Pipeline(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.pyramid = "image"
        self.hidden = 64
        self.img2Fencoder = Pyramid_direction_extractor(in_plane=3, hidden_pyramid=self.hidden_dim,
                                                        pyramid=self.pyramid)
        self.img2Bencoder = Pyramid_direction_extractor(in_plane=3, hidden_pyramid=self.hidden_dim,
                                                        pyramid=self.pyramid)
        self.interpolate = unet3P()
        self.decoder = nn.Sequential()
        self.epsilon = 1e-6
        self.loss_census = census()

    def forward(self, allframes, training_flag=True):
        # allframes 0<1>2<3>4<5>6

        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_tea_pred = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_mse = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')

        output_allframes = []
        output_teacher = []
        flow_list = []
        mask_list = []

        # Use Pretrained SpyNet to do optical flow estimation

        list_of_forward_optical_flow_for_warp_list = []
        list_of_backward_optical_flow_for_warp_list = []
        for i in range(0, 4, 1):
            img0 = allframes[:, 6 * i:6 * i + 3]
            gt = allframes[:, 6 * i + 3:6 * i + 6]
            img1 = allframes[:, 6 * i + 6:6 * i + 9]
            imgs_and_gt = torch.cat([img0, img1, gt], dim=1)
            forward_optical_flow_for_warp_list = self.spynet(torch.cat([img0, img1], dim=1))
            list_of_forward_optical_flow_for_warp_list.append(forward_optical_flow_for_warp_list)
            backward_optical_flow_for_warp_list = self.spynet(torch.cat([img1, img0], dim=1))
            list_of_backward_optical_flow_for_warp_list.append(backward_optical_flow_for_warp_list)

        # Encoder pyramid
        fallfeatures_d0, fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes, pyramid="image",
                                                                              flag_st='stu')
        ballfeatures_d0, ballfeatures_d2, ballfeatures_d4 = self.img2Bencoder(allframes, pyramid="image",
                                                                              flag_st='stu')
        # initial first warp feature
        list_of_half_att_forward_warped_feature_list = []
        list_of_half_att_backward_warped_feature_list = []
        list_of_forward_warped_feature_list = []
        list_of_backward_warped_feature_list = []
        list_of_forward_warped_feature_list.append(list(fallfeatures_d4[0], fallfeatures_d2[0], fallfeatures_d0[0]))
        list_of_backward_warped_feature_list.append(list(ballfeatures_d4[-1], ballfeatures_d2[-1], ballfeatures_d0[-1]))
        att_forward_warped_feature_list = []
        att_backward_warped_feature_list = []

        full_forward_warped_feature_list = []
        full_backward_warped_feature_list = []
        for i in range(0, 3, 1):
            current_fw_OF_pyramid_list = list_of_forward_optical_flow_for_warp_list[i]
            img_1_pyramid_ori_feature_list = list(fallfeatures_d4[i + 1], fallfeatures_d2[i + 1],
                                                  fallfeatures_d0[i + 1])

            current_bw_OF_pyramid_list = list_of_backward_optical_flow_for_warp_list[i]
            img_0_pyramid_ori_feature_list = list(ballfeatures_d4[-(i + 1) - 1], ballfeatures_d2[-(i + 1) - 1],
                                                  ballfeatures_d0[-(i + 1) - 1])

            for j in range(0, 3, 1):
                half_att_forward_warped_feature, forward_temporal_next = self.Warp_Fusion(
                    list_of_forward_warped_feature_list[i][j],
                    current_fw_OF_pyramid_list[j],
                    img_1_pyramid_ori_feature_list[j])

                att_forward_warped_feature_list.append(half_att_forward_warped_feature.clone())
                full_forward_warped_feature_list.append(forward_temporal_next.clone())

                half_att_backward_warped_feature, backward_temporal_next = self.Warp_Fusion(
                    list_of_backward_warped_feature_list[i][j],
                    current_bw_OF_pyramid_list[j],
                    img_0_pyramid_ori_feature_list[j])

                att_backward_warped_feature_list.append(half_att_backward_warped_feature.clone())
                full_backward_warped_feature_list.append(backward_temporal_next.clone())

            list_of_half_att_forward_warped_feature_list.append(att_forward_warped_feature_list)
            list_of_half_att_backward_warped_feature_list.append(att_backward_warped_feature_list)
            list_of_forward_warped_feature_list.append(full_forward_warped_feature_list)
            list_of_backward_warped_feature_list.append(full_backward_warped_feature_list)

        # # remove the initial warped feature which is actually the same as the feature of starting frame
        # del list_of_forward_warped_feature_list[0]
        # del list_of_backward_warped_feature_list[0]

        """
        Order of list:
        1. F/B-ward_optical_flow_for_warp_list: from the first interval to the last interval: ( 0-2, 2-4, 4-6 and 2-0, 4-2, 6-4 )
        2. F/B-allfeature from self.img2Fencoder: from first image to the last image: ( 0 2 4 6 and 0 2 4 6)
        3. full_FB-warped_feature_list are inside list_of_FB-warped_feature_list. The list of FB-warped_feature_list: ( 0-2, 2-4, 4-6 and 6-4, 4-2, 2-0 )
        3. att_forward_warped_feature_list, full_forward_warped_feature_list. The ordinary order: ( 0-2, 2-4, 4-6 )
        4. att_backward_warped_feature_list, full_backward_warped_feature_list. The back feature in reversed order: ( d4, d2, d0 )
        """

        for i in range(3):
            if training_flag:
                gt = allframes[:, 6 * i + 3:6 * i + 6]

            # The [i][-1] is supposed to pick up the d0 level
            current_fw_OF = list_of_forward_optical_flow_for_warp_list[i][-1]
            current_bw_OF = list_of_backward_optical_flow_for_warp_list[i][-1]
            ori_img0_feature = fallfeatures_d0[i]
            ori_img1_feature = ballfeatures_d0[i]
            warped_fimg0_d0 = warp(fallfeatures_d0[i], 0.5 * current_fw_OF)
            warped_fimg1_d0 = warp(ballfeatures_d0[i], 0.5 * current_bw_OF)

            # Use Unet to create the three interpolated frame
            f_att_d4, f_att_d2, f_att_d0 = list_of_half_att_forward_warped_feature_list[i]
            b_att_d4, b_att_d2, b_att_d0 = list_of_half_att_backward_warped_feature_list[i]

            featureUnet = self.interpolate(ori_img0_feature, ori_img1_feature, warped_fimg0_d0,
                                           warped_fimg1_d0, f_att_d0,
                                           f_att_d2, f_att_d4, b_att_d0, b_att_d2, b_att_d4)
            # flow, mask, merged, flow_teacher, merged_teacher, loss_tea_pred = self.interpolate(allframes)
            predictimage = self.decoder(featureUnet)

            # Start loss computation
            loss_pred = torch.mean(
                torch.sqrt(torch.pow((predictimage - gt), 2) + self.epsilon ** 2)) + self.loss_census(
                predictimage, gt)
            loss_mse = ((predictimage - gt) ** 2).detach()
            loss_mse = loss_mse.mean()

            # loss_tea = 0
            merged_teacher = tea_predictimage  # not used. just to avoid error
            flow_teacher = predictimage * 0  # not used. just to avoid error
            mask_list = [flow_teacher, flow_teacher, flow_teacher]
            flow_list = [flow_teacher, flow_teacher, flow_teacher]

            # =======================================================
            Sum_loss_context += loss_pred
            Sum_loss_mse += loss_mse
            Sum_loss_tea_pred += loss_tea_pred
            output_allframes.append(img0)
            output_teacher.append(img0)
            # output_allframes.append(merged[2])
            output_allframes.append(merged)
            flow_list.append(flow)
            mask_list.append(mask)

            # The way RIFE compute prediction loss and 
            # loss_l1 = (self.lap(merged[2], gt)).mean()
            # loss_tea = (self.lap(merged_teacher, gt)).mean()

        img6 = allframes[:, -3:]
        output_allframes.append(img6)
        output_teacher.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        output_teacher_tensors = torch.stack(output_teacher, dim=1)

        # Dummy output
        flow_teacher_list = []
        loss_dist = 0

        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_teacher_tensors, Sum_loss_tea_pred, Sum_loss_context, Sum_loss_mse, loss_dist


class SimplifiedPyramid_Pipline_unit(nn.Module):
    def __init__(self, pyramid, warp_type="two"):
        super().__init__()
        self.spynet = None  # SPYNet()

        if warp_type == "two":
            self.Warp_Fusion = TwoWarpSecondFrame_Fusion()
        elif warp_type == "one":
            self.Warp_Fusion = OneWarpSecondFrame_Fusion()
        else:
            raise NotImplementedError(
                "Only one or two warp is written")

        self.unet = Unet_for_3Pyramid(hidden_dim=self.hidden_dim, shift_dim=self.shift_dim)  # unet()

    def forward(self, x, forward_warped_feature_list, backward_warped_feature_list, img_0_pyramid_ori_feature_list,
                img_1_pyramid_ori_feature_list):
        # encoder_pyramid_feature: [feature_d0, feature_d1, feature_d2]
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]

        # optical_flow_for_warp_list:
        # [flow_d0, flow_d1, flow_d2]

        for i in range(3):
            half_att_forward_warped_feature, forward_temporal_next = self.Warp_Fusion(forward_warped_feature_list[i],
                                                                                      forward_optical_flow_for_warp_list[
                                                                                          i],
                                                                                      img_1_pyramid_ori_feature_list[i])

            att_backward_warped_feature_list, backward_temporal_next = self.Warp_Fusion(backward_warped_feature_list[i],
                                                                                        backward_optical_flow_for_warp_list[
                                                                                            i],
                                                                                        img_0_pyramid_ori_feature_list[
                                                                                            i])

            att_forward_warped_feature_list.append(half_att_forward_warped_feature.clone())
            att_backward_warped_feature_list.append(att_backward_warped_feature_list.clone())

            full_forward_warped_feature_list.append(forward_temporal_next.clone())
            full_backward_warped_feature_list.append(backward_temporal_next.clone())

        frame_interpolated = self.unet(img0, img1,
                                       forward_optical_flow_for_warp_list[-1], backward_optical_flow_for_warp_list[-1],
                                       att_forward_warped_feature_list, att_backward_warped_feature_list)

        return frame_interpolated, full_forward_warped_feature_list, full_backward_warped_feature_list

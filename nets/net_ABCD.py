from .bilateralCBAM import BilateralConvCBAM
from .NNutils import *
import torch.nn.functional as F

import nets.resnet as RESNET

__all__ = ['ABCD']

class ABCD(nn.Module):

    def __init__(self, args):
        super(ABCD, self).__init__()

        # build BCL network
        self.scales_filter_map = args['scales_filter_map']
        assert len(self.scales_filter_map) == 5
        self.chunk_size = -1
        self.dim = 3
        self.pc_feat_channel = args['pc_feat_channel'] 
        self.device = args['DEVICE']

        self.pc_conv0 = nn.Sequential(
            conv_1x1(self.dim, 32, use_leaky=args['use_leaky']),
            conv_1x1(32, 32, use_leaky=args['use_leaky']),
            conv_1x1(32, 32, use_leaky=args['use_leaky']),
        )

        self.bcn1 = BilateralConvCBAM(self.dim, self.scales_filter_map[0][1],
                                      32 + self.dim + 1, [32, 32],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=self.chunk_size)

        self.bcn1_ = BilateralConvCBAM(self.dim, self.scales_filter_map[0][1],
                                       64 + 32 + self.dim + 1, [self.pc_feat_channel, self.pc_feat_channel],
                                       self.device,
                                       use_bias=args['bcn_use_bias'],
                                       use_leaky=args['use_leaky'],
                                       use_norm=args['bcn_use_norm'],
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args['last_relu'],
                                       chunk_size=self.chunk_size)

        self.bcn2 = BilateralConvCBAM(self.dim, self.scales_filter_map[1][1],
                                      32 + self.dim + 1, [64, 64],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=self.chunk_size)

        self.bcn2_ = BilateralConvCBAM(self.dim, self.scales_filter_map[1][1],
                                       128 + 64 + self.dim + 1, [64, 64],
                                       self.device,
                                       use_bias=args['bcn_use_bias'],
                                       use_leaky=args['use_leaky'],
                                       use_norm=args['bcn_use_norm'],
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args['last_relu'],
                                       chunk_size=self.chunk_size)

        self.bcn3 = BilateralConvCBAM(self.dim, self.scales_filter_map[2][1],
                                      64 + self.dim + 1, [128, 128],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=self.chunk_size)

        self.bcn3_ = BilateralConvCBAM(self.dim, self.scales_filter_map[2][1],
                                       256 + 128 + self.dim + 1, [128, 128],
                                       self.device,
                                       use_bias=args['bcn_use_bias'],
                                       use_leaky=args['use_leaky'],
                                       use_norm=args['bcn_use_norm'],
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args['last_relu'],
                                       chunk_size=self.chunk_size)

        self.bcn4 = BilateralConvCBAM(self.dim, self.scales_filter_map[3][1],
                                      128 + self.dim + 1, [256, 256],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=self.chunk_size)

        self.bcn4_ = BilateralConvCBAM(self.dim, self.scales_filter_map[3][1],
                                       256 + 256 + self.dim + 1, [256, 256],
                                       self.device,
                                       use_bias=args['bcn_use_bias'],
                                       use_leaky=args['use_leaky'],
                                       use_norm=args['bcn_use_norm'],
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args['last_relu'],
                                       chunk_size=self.chunk_size)

        self.bcn5 = BilateralConvCBAM(self.dim, self.scales_filter_map[4][1],
                                      256 + self.dim + 1, [256, 256],
                                      self.device,
                                      use_bias=args['bcn_use_bias'],
                                      use_leaky=args['use_leaky'],
                                      use_norm=args['bcn_use_norm'],
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args['last_relu'],
                                      chunk_size=self.chunk_size)

        self.bcn5_ = BilateralConvCBAM(self.dim, self.scales_filter_map[4][1],
                                       256, [256, 256],
                                       self.device,
                                       use_bias=args['bcn_use_bias'],
                                       use_leaky=args['use_leaky'],
                                       use_norm=args['bcn_use_norm'],
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args['last_relu'],
                                       chunk_size=self.chunk_size)

        # self.pc_mask1 = conv_bn_relu(self.pc_feat_channel, 1, kernel_size=1, stride=1, padding=0)
        self.pc_mask1 = nn.Sequential(
            nn.Conv2d(self.pc_feat_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.pc_conv2 = conv_bn_relu(self.pc_feat_channel, self.pc_feat_channel, kernel_size=3, stride=2, padding=1)
        self.pc_conv3 = conv_bn_relu(self.pc_feat_channel, self.pc_feat_channel, kernel_size=3, stride=2, padding=1)
        self.pc_conv4 = conv_bn_relu(self.pc_feat_channel, self.pc_feat_channel, kernel_size=3, stride=2, padding=1)

        # Base Network from Self-Supervised Depth Completion
        self.d_conv0 = conv_bn_relu(1,
                                    16,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.im_conv0 = conv_bn_relu(3,
                                    48,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        pretrained_model = RESNET.__dict__[args['resnet']](pretrained=False)
        pretrained_model.apply(init_weights)

        self.conv1 = pretrained_model._modules['layer1']
        self.conv2 = pretrained_model._modules['layer2']
        self.conv3 = pretrained_model._modules['layer3']
        self.conv4 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        self.conv5 = conv_bn_relu(512,
                                    512,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt4 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(512 + 256 + self.pc_feat_channel),
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(256 + 128 + self.pc_feat_channel),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=(128 + 64 + self.pc_feat_channel),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)

        # self.im_mask1 = conv_bn_relu(in_channels=(64 + 64 + self.pc_feat_channel),
        #                              out_channels=1,
        #                              kernel_size=1,
        #                              stride=1)
        self.im_mask1 = nn.Sequential(
            nn.Conv2d(64 + 64 + self.pc_feat_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.convf = conv_bn_relu(in_channels=(64 + 64 + 64),
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)

    def forward(self, pc, im, d, generated_data, o_hw, check=False):

        # -------------------- BCL Network --------------------

        pc_idx = pc.clone()

        pc_bcl0 = self.pc_conv0(pc)

        pc_bcl1 = self.bcn1(torch.cat((generated_data[0]['pc1_el_minus_gr'], pc_bcl0), dim=1),
                             in_barycentric=generated_data[0]['pc1_barycentric'],
                             in_lattice_offset=generated_data[0]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[0]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc_bcl2 = self.bcn2(torch.cat((generated_data[1]['pc1_el_minus_gr'], pc_bcl1), dim=1),
                             in_barycentric=generated_data[1]['pc1_barycentric'],
                             in_lattice_offset=generated_data[1]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[1]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc_bcl3 = self.bcn3(torch.cat((generated_data[2]['pc1_el_minus_gr'], pc_bcl2), dim=1),
                             in_barycentric=generated_data[2]['pc1_barycentric'],
                             in_lattice_offset=generated_data[2]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[2]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc_bcl4 = self.bcn4(torch.cat((generated_data[3]['pc1_el_minus_gr'], pc_bcl3), dim=1),
                             in_barycentric=generated_data[3]['pc1_barycentric'],
                             in_lattice_offset=generated_data[3]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[3]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc_bcl5 = self.bcn5(torch.cat((generated_data[4]['pc1_el_minus_gr'], pc_bcl4), dim=1),
                             in_barycentric=generated_data[4]['pc1_barycentric'],
                             in_lattice_offset=generated_data[4]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[4]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        if check:
            print("pc_bcl0     ", pc_bcl0.size())
            print("pc_bcl1     ", pc_bcl1.size())
            print("pc_bcl2     ", pc_bcl2.size())
            print("pc_bcl3     ", pc_bcl3.size())
            print("pc_bcl4     ", pc_bcl4.size())
            print("pc_bcl5     ", pc_bcl5.size())

        pc_bcl5_back = self.bcn5_(pc_bcl5,
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[4]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[4]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[4]['pc1_lattice_offset'],
                                   )
        pc_bcl4_back = self.bcn4_(torch.cat((generated_data[4]['pc1_el_minus_gr'], pc_bcl5_back, pc_bcl4), dim=1),
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[3]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[3]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[3]['pc1_lattice_offset'],
                                   )
        pc_bcl3_back = self.bcn3_(torch.cat((generated_data[3]['pc1_el_minus_gr'], pc_bcl4_back, pc_bcl3), dim=1),
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[2]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[2]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[2]['pc1_lattice_offset'],
                                   )
        pc_bcl2_back = self.bcn2_(torch.cat((generated_data[2]['pc1_el_minus_gr'], pc_bcl3_back, pc_bcl2), dim=1),
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[1]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[1]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[1]['pc1_lattice_offset'],
                                   )
        pc_bcl1_back = self.bcn1_(torch.cat((generated_data[1]['pc1_el_minus_gr'], pc_bcl2_back, pc_bcl1), dim=1),
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[0]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[0]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[0]['pc1_lattice_offset'],
                                   )
        if check:
            print("pc_bcl5_back", pc_bcl5_back.size())
            print("pc_bcl4_back", pc_bcl4_back.size())
            print("pc_bcl3_back", pc_bcl3_back.size())
            print("pc_bcl2_back", pc_bcl2_back.size())
            print("pc_bcl1_back", pc_bcl1_back.size())

        # -------------------- Projection --------------------

        pc_feat1 = pc_to_depth_torch(pc_idx, pc_bcl1_back, o_hw, self.device)  # batchsize * 32 * 352 * 1216
        pc_mask = self.pc_mask1(pc_feat1) 
        pc_feat2 = self.pc_conv2(pc_feat1)  # batchsize * 32 * 176 * 608
        pc_feat3 = self.pc_conv3(pc_feat2)  # batchsize * 32 * 88 * 304
        pc_feat4 = self.pc_conv4(pc_feat3)  # batchsize * 32 * 44 * 152

        if check:
            print("pc_feat1    ", pc_feat1.size())
            print("pc_feat2    ", pc_feat2.size())
            print("pc_feat3    ", pc_feat3.size())
            print("pc_feat4    ", pc_feat4.size()) 

        # -------------------- ResNet for sparse depth & color image --------------------

        d_conv0 = self.d_conv0(d)
        im_conv0 = self.im_conv0(im)
        conv0 = torch.cat((d_conv0, im_conv0), 1)  # batchsize * ? * 352 * 1216

        conv1 = self.conv1(conv0)  # batchsize * ? * 352 * 1216
        conv2 = self.conv2(conv1)  # batchsize * ? * 176 * 608
        conv3 = self.conv3(conv2)  # batchsize * ? * 88 * 304
        conv4 = self.conv4(conv3)  # batchsize * ? * 44 * 152
        conv5 = self.conv5(conv4)  # batchsize * ? * 22 * 76

        if check:
            print("conv0       ", conv0.size())
            print("conv1       ", conv1.size())
            print("conv2       ", conv2.size())
            print("conv3       ", conv3.size())
            print("conv4       ", conv4.size())
            print("conv5       ", conv5.size())

        # -------------------- Late Fusion Decoder --------------------

        convt4 = self.convt4(conv5)
        y = torch.cat((convt4, conv4, pc_feat4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3, pc_feat3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2, pc_feat2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1, pc_feat1), 1)

        im_mask = self.im_mask1(y)
        masks = torch.cat([pc_mask, im_mask], dim=1)
        masks = F.softmax(masks, dim=1)

        if check:            
            print("convt4      ", convt4.size())
            print("convt3      ", convt3.size())
            print("convt2      ", convt2.size())
            print("convt1      ", convt1.size())
            print("pc_mask     ", pc_mask.size())
            print("im_mask     ", im_mask.size())
            print("masks       ", masks.size())

        pc_feat = masks[:, 0, :, :] * pc_feat1
        im_feat = masks[:, 1, :, :] * y
        
        if check:
            print("im_feat     ", im_feat.size())
            print("pc_feat     ", pc_feat.size())

        y = torch.cat((im_feat, pc_feat), 1)
        y = self.convf(y)

        if check:
            print("depth       ", y.size())   

        out = {
            'depth': 100 * y,
            'im_mask': im_mask,
            'pc_mask': pc_mask,
        }     

        return out
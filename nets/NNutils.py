import torch
import torch.nn as nn

LEAKY_RATE = 0.1

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_1x1(in_channels, out_channels, kernel_size=1,
             stride=1, padding=0, use_leaky=False, bias=True):
    relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
    layers = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                           relu)
    # initialize the weights
    for m in layers.modules():
        init_weights(m)
    return layers


def conv_bn_relu(in_channels, out_channels, kernel_size, \
                 stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


def convt_bn_relu(in_channels, out_channels, kernel_size, \
                  stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    # Add one more layer.. Aurier
    layers.append(
        nn.Conv2d(out_channels,
                  out_channels,
                  3,
                  1,
                  1,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

# def pc_to_depth_torch(pc, feat, im_size, _s=2 ** 8):
#     """
#         Color pointcloud by assigning RGB pixel values
#         at projected points on the image
#         :param pc: torch Variable of xyz in shape (1, 3, N)
#         :param feat: torch Variable of pc_feat in shape (1, C, N)
#         :param im_size: tuple of image size (H, W)
#         :return im_feat: NumPy array of pcd in fov (1, C, H, W)
#     """
#     indices = pc[0, :2, :] * _s
#     indices = indices.type(torch.cuda.LongTensor)  # (2, N)
#     values = feat[0, :, :].t()  # (N, C)

#     im_feat = torch.cuda.sparse.FloatTensor(indices, values,
#                                             torch.Size([im_size[0], im_size[1], values.size()[-1]])).to_dense()
#     im_feat = im_feat.permute(2, 0, 1).unsqueeze(0)
#     im_feat = im_feat.type(torch.cuda.FloatTensor)
# #     im_feat = Variable(im_feat.type(torch.cuda.FloatTensor),
                      
#     return im_feat

def pc_to_depth_torch(pc, feat, im_size, device, s=2 ** 8):
    height, width = im_size[0], im_size[1]
    indices = pc[0, :2, :] * s
    indices = indices.to(device).long()  # (2, N)
    values = feat[0, :, :].t()  # (N, C)
    depth = torch.zeros((height, width, values.size(-1))).to(device)
    depth[indices[0].tolist(), indices[1].tolist()] = values
    depth = depth.permute(2, 0, 1).unsqueeze(0)                 
    return depth
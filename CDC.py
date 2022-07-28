import tensorflow as tf
import numpy as np
import math
import pdb
from tensorflow.python.keras.layers import Conv3D, AveragePooling3D

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Parameter
import math
import pdb


#pytorch
# class CDC_ST(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False, theta=0.6, weight=None):
#
#         super(CDC_ST, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.theta = theta
#         self.conv.weight = torch.nn.Parameter(torch.from_numpy(weight.transpose(4, 3, 0, 1, 2)))
#
#     def forward(self, x):
#         out_normal = self.conv(x)
#
#         if math.fabs(self.theta - 0.0) < 1e-8:
#             return out_normal
#         else:
#             # pdb.set_trace()
#             [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape
#
#             # only CD works on temporal kernel size>1
#             if self.conv.weight.shape[2] > 1:
#                 kernel_diff = self.conv.weight.sum(2).sum(2).sum(2)
#                 kernel_diff = kernel_diff[:, :, None, None, None]
#                 # print(kernel_diff.shape)
#                 # print(kernel_diff)
#                 out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
#                                     padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
#                 return out_normal - self.theta * out_diff, self.conv.weight
#
#             else:
#                 return out_normal
#
# class CDC_TR(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, groups=1, bias=False, theta=0.3):
#
#         super(CDC_TR, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.avgpool = nn.AvgPool3d(kernel_size=(kernel_size, 1, 1), stride=stride, padding=(padding, 0, 0))
#         self.theta = theta
#
#     def forward(self, x):
#         out_normal = self.conv(x)
#         local_avg = self.avgpool(x)
#         if math.fabs(self.theta - 0.0) < 1e-8:
#             return out_normal
#         else:
#             # pdb.set_trace()
#             [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape
#
#             # only CD works on temporal kernel size>1
#             if self.conv.weight.shape[2] > 1:
#                 kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
#                     2).sum(2)
#                 kernel_diff = kernel_diff[:, :, None, None, None]
#                 out_diff = F.conv3d(input=local_avg, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
#                                     padding=0, groups=self.conv.groups)
#                 return out_normal - self.theta * out_diff
#
#             else:
#                 return out_normal


# # tensorflow
# #def conv3D_ST(batch_input, out_channels, kernel_size, padding, activation, stride=1, theta=0.6):
# def conv3D_ST(batch_input, out_channels, kernel_size, padding, activation, stride=1, theta=0):
#     conv = Conv3D(filters=out_channels, kernel_size=kernel_size, padding=padding, activation=activation, data_format='channels_last')
#     out_normal = conv(batch_input)
#     if math.fabs(theta - 0.0) < 1e-8:
#         return out_normal
#     else:
#         conv_weight = list(conv.get_weights())
#         if conv_weight[0].shape[0] > 1:
#             weight = conv_weight[0]
#             kernel_diff = weight.sum(0).sum(0).sum(0)
#             kernel_diff = kernel_diff[None, None, None, :, :]
#             kernel_diff_tensor = tf.convert_to_tensor(kernel_diff)
#             out_diff = tf.nn.conv3d(batch_input, kernel_diff_tensor, [1, stride, stride, stride, 1], padding=padding.upper())

#             return out_normal - theta * out_diff
#         else:
#             return out_normal

            # tensorflow
#def conv3D_ST(batch_input, out_channels, kernel_size, padding, activation, stride=1, theta=0.6):
def conv3D_ST(batch_input, out_channels, kernel_size, padding, activation, stride=1, theta=0.2):
    conv = Conv3D(filters=out_channels, kernel_size=kernel_size, padding=padding, activation=activation, data_format='channels_last')
    out_normal = conv(batch_input)
    if math.fabs(theta - 0.0) < 1e-8:
        return out_normal
    else:
        conv_weight = list(conv.get_weights())
        if conv_weight[0].shape[0] > 1:
            weight = conv_weight[0]
            kernel_diff = weight.sum(0).sum(0).sum(0)
            kernel_diff = kernel_diff[None, None, None, :, :]
            kernel_diff_tensor = tf.convert_to_tensor(kernel_diff)
            out_diff = tf.nn.conv3d(batch_input, kernel_diff_tensor, [1, stride, stride, stride, 1], padding=padding.upper())

            return out_normal - theta * out_diff
        else:
            return out_normal


def conv3D_T(batch_input, out_channels, kernel_size, padding, activation, stride=1, theta=0.2):
    conv = Conv3D(filters=out_channels, kernel_size=kernel_size, padding=padding, activation=activation, data_format='channels_last')
    out_normal = conv(batch_input)
    if math.fabs(theta - 0.0) < 1e-8:
        return out_normal
    else:
        conv_weight = list(conv.get_weights())
        if conv_weight[0].shape[0] > 1:
            weight = conv_weight[0]
            kernel_diff = weight[0, :, :, :, :].sum(0).sum(0) + weight[2, :, :, :, :].sum(0).sum(0)
            kernel_diff = kernel_diff[None, None, None, :, :]
            out_diff = tf.nn.conv3d(batch_input, kernel_diff, [1, stride, stride, stride, 1], padding=padding.upper())

            return out_normal - theta * out_diff
        else:
            return out_normal

def conv3D_TR(batch_input, out_channels, kernel_size, padding, activation, stride, theta=0.6):
    conv = Conv3D(filters=out_channels, kernel_size=kernel_size, padding=padding, activation=activation)
    out_normal = conv(batch_input)
    local_avg = AveragePooling3D(pool_size=(kernel_size, 1, 1), strides=1, padding=padding)(batch_input)

    if math.fabs(theta - 0.0) < 1e-8:
        return out_normal
    else:
        conv_weight = conv.get_weights()
        if conv_weight[0].shape[0] > 1:
            weight = conv_weight[0]
            kernel_diff = weight[0, :, :, :, :].sum(0).sum(0) + weight[2, :, :, :, :].sum(0).sum(0)
            kernel_diff = kernel_diff[None, None, None, :, :]
            out_diff = tf.nn.conv3d(local_avg, kernel_diff, [1, stride, stride, stride, 1], padding=padding.upper())

            return out_normal - theta * out_diff, weight
        else:
            return out_normal


if __name__ == "__main__":
    from tensorflow.python.keras.models import Model
    input_shape = (4, 28, 28, 28, 1)
    # x = tf.ones(input_shape)
    x = tf.keras.Input(input_shape)
    y, weight = conv3D_ST(x, 3, 3, padding='same', activation=None, stride=1)
    model = Model(inputs=x, outputs=y)

    print(y.shape)

    # pytorch
    input_shape1 = (4, 1, 28, 28, 28)
    x1 = torch.ones(input_shape1)
    m1 = CDC_ST(1, 3, kernel_size=3, padding=1, theta=0.6, weight=weight)
    y1, weight1 = m1(x1)
    print(y1.shape)
    y1_transpose = y1.permute(0, 2, 3, 4, 1)




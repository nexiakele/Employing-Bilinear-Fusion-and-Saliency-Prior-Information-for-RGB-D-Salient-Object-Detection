# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:28:14 2018

@author: Dell
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
#from tensorboardX import SummaryWriter
###############################################################################
###############################################################################
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv1x1xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
def conv1x1xbn(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes))
###############################################################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv3x3xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
def conv3x3xbn(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_planes))
          
def convkxk(in_planes, out_planes,k_size=5, stride=1):
    """kxk convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=k_size, stride=stride,
                     padding=k_size//2, bias=False)
def convkxkxbnxrelu(in_planes, out_planes,k_size=5, stride=1):
    """kxk convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=k_size, stride=stride,
                                   padding=k_size//2, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
def convkxkxbn(in_planes, out_planes,k_size=5, stride=1):
    """kxk convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=k_size, stride=stride,
                                   padding=k_size//2, bias=False),
                         nn.BatchNorm2d(out_planes))
class conv3x3_bn_relu_d(nn.Module):
      def __init__(self, in_channel, out_channel, dilated = 1):
        super(conv3x3_bn_relu_d, self).__init__()  
        self.dilated1 = nn.Sequential(
                                nn.Conv2d(in_channel, out_channel,kernel_size=3,stride=1,
                                          padding=dilated, dilation=dilated, bias=False),
                                nn.BatchNorm2d(out_channel),
                                nn.ReLU())
      def forward(self,x):
            out = self.dilated1(x)
            return out

def convkxkxbnxrelu_d(in_planes, out_planes,k_size=5, stride=1,d=1):
    """kxk convolution with padding"""
    p = ((k_size-1)*(d-1) + k_size)//2
    return nn.Sequential(nn.Conv2d(in_planes, 
                                   out_planes, 
                                   kernel_size=k_size, 
                                   stride=stride,
                                   padding=p, 
                                   bias=False,
                                   dilation=d),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())    
        
def deconv3x3(in_c, out_c=None):
        if  out_c ==None:
            out_c = in_c
        return nn.Sequential(
                        nn.ConvTranspose2d(in_c, out_c,kernel_size=3, stride = 2,
                                           padding = 1,output_padding=1, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU())
def deconv2x2(in_c, out_c=None):
        if  out_c ==None:
            out_c = in_c
        return nn.Sequential(
                        nn.ConvTranspose2d(in_c, out_c,kernel_size=2, stride = 2,
                                           padding = 0, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU())
def interpolate(x, scale=2):
        return F.interpolate(x, 
                            scale_factor=scale, 
                            mode='bilinear', 
                            align_corners=False)
        
class split_tensor(nn.Module):
      def __init__(self, blocks):
        super(split_tensor, self).__init__()  
        self.blocks = blocks
      def forward(self,x):
             layers = []
             for i in range(self.blocks):
                   mask = x[:, i:i+1, :, :]
                   layers.append(mask)
             return layers
def split_mask(x, blocks):
 layers = []
 for i in range(blocks):
       mask = x[:, i:i+1, :, :]
       layers.append(mask)
 return layers             
###############################################################################
###############################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,group_number=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.downsample = None

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes, stride),
                                            nn.BatchNorm2d(planes),)
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out

class Res_upsample(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super(Res_upsample, self).__init__() 
        if out_channel == None:
            out_channel = in_channel
        self.conv1 = conv3x3xbnxrelu(in_channel, out_channel)
        self.conv2 = nn.ConvTranspose2d(out_channel, out_channel,kernel_size=2,
                                        stride = 2, padding = 0, bias=False)
        self.bn  = nn.BatchNorm2d(out_channel)
        self.skip = None
        if in_channel != out_channel:
            self.skip = conv1x1xbn(in_channel, out_channel)
        self.relu = nn.ReLU()
    def forward(self,x):
        skip = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        skip = F.interpolate(skip, 
                            scale_factor=2, 
                            mode='bilinear', 
                            align_corners=True)
        if self.skip is not None:
            skip = self.skip(skip)
        out = skip + out
        out = self.relu(out)
        return out  


class basic_deconv_up(nn.Module):
    def __init__(self, in_channel, out_channel=None, scale = 2):
        super(basic_deconv_up, self).__init__() 
        if out_channel is None:
              out_channel = in_channel
        self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(in_channel, out_channel,kernel_size=3, stride = 2,
                                           padding = 1,output_padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(inplace=True))
    def forward(self,x):
        out = self.deconv(x)
        return out
class basic_deconv_up2(nn.Module):
    def __init__(self, in_channel, out_channel=None, scale = 2):
        super(basic_deconv_up2, self).__init__() 
        if out_channel is None:
              out_channel = in_channel
        self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(in_channel, out_channel,kernel_size=2, stride = 2,
                                           padding = 0, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(inplace=True))
    def forward(self,x):
        out = self.deconv(x)
        return out    

def main():
      
      
    x=torch.rand(2,128,128,128) #随便定义一个输入

#    net = basic_group_layer_bn(256, 128)
#    net = basic_subpixel_interpolate_multi_maxpool2(64, 64)
    net = Res_upsample(128,128)
    out = net.forward(x)
    print(out.shape)
#    writer = SummaryWriter(log_dir='./log')
#    writer.add_graph(net, (x))
#    writer.close()  
#    print('done')   
if __name__ == '__main__':
      main()
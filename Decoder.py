# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:18:07 2018

@author: Dell
"""
import torch
from torch import nn
import modules.layers as _layers
import torch.nn.functional as F
################################################################################################
################################################################################################
class Predict(nn.Module):
      def __init__(self,incs, outcs):
            super(Predict, self).__init__() 
            self.body1 = nn.Sequential(_layers.conv3x3(incs[1],outcs),
                                       nn.Sigmoid())
            self.body2 = nn.Sequential(_layers.conv3x3(incs[2],outcs),
                                       nn.Sigmoid())
            self.body3 = nn.Sequential(_layers.conv3x3(incs[3],outcs),
                                       nn.Sigmoid())
            self.body4 = nn.Sequential(_layers.conv3x3(incs[4],outcs),
                                       nn.Sigmoid())
      def forward(self, in1, in2, in3, inc4):
          out4 = self.body4(inc4)
          out3 = self.body3(in3)
          out2 = self.body2(in2)
          out1 = self.body1(in1)
          return out1, out2, out3, out4

##没有 SPIGF 模块， 没有SFP模块， 最简单的decoder
class Decoder0(nn.Module):
      def __init__(self,incs, outcs):
            super(Decoder0, self).__init__() 
            self.body4 = nn.Sequential(_layers.BasicBlock(incs[4],incs[4]),
                                        _layers.BasicBlock(incs[4],incs[4]))
            self.body3 = nn.Sequential(_layers.BasicBlock(incs[3]+incs[4],incs[3]),
                                       _layers.BasicBlock(incs[3],incs[3]),
                                       _layers.BasicBlock(incs[3],incs[3]))
            self.body2 = nn.Sequential(_layers.BasicBlock(incs[2]+incs[3],incs[2]),
                                       _layers.BasicBlock(incs[2],incs[2]))
            self.body1 = nn.Sequential(_layers.BasicBlock(incs[1]+incs[2],incs[1]),
                                       _layers.BasicBlock(incs[1],incs[1]))
            self.body0 = nn.Sequential(_layers.conv3x3xbnxrelu(incs[1],incs[0]),
                                       _layers.conv3x3xbnxrelu(incs[0],incs[0]),
                                       _layers.conv3x3(incs[0],1),
                                       nn.Sigmoid())    
            self.p = Predict(incs, 1)
      def forward(self, in1, in2, in3, inc4):
          out4 = self.body4(inc4)
          out3 = _layers.interpolate(out4)
          out3 = torch.cat((out3, in3),1)
          out3 = self.body3(out3)
          out2 = _layers.interpolate(out3)
          out2 = torch.cat((out2, in2),1)
          out2 = self.body2(out2)
          out1 = _layers.interpolate(out2)
          out1 = torch.cat((out1, in1),1)
          out1 = self.body1(out1)
          out0 = _layers.interpolate(out1)
          out0 = self.body0(out0)
          out1, out2, out3, out4 = self.p(out1, out2, out3, out4)
          return out0, out1, out2, out3, out4 

class ReGuide1(nn.Module):
      def __init__(self,incs, outcs):
            super(ReGuide1, self).__init__() 
            self.conv1 = _layers.conv1x1xbnxrelu(incs[1], 64)
            self.conv2 = _layers.conv1x1xbnxrelu(incs[2], 64)
            self.conv3 = _layers.conv1x1xbnxrelu(incs[3], 64)
            self.conv4 = _layers.conv1x1xbnxrelu(incs[4], 64)
            inc = 64*4
            self.fusion = nn.Sequential(_layers.BasicBlock(inc, outcs[1]),
                                        _layers.BasicBlock(outcs[1], outcs[1]),)
            self.oc1 = _layers.BasicBlock(outcs[1], outcs[2],2)
            self.oc2 = _layers.BasicBlock(outcs[2], outcs[3],2) 
            self.oc3 = _layers.BasicBlock(outcs[3], outcs[4],2)
      def forward(self, in1, in2, in3, in4):
          o1 = self.conv1(in1)
          o2 = self.conv2(in2)
          o3 = self.conv3(in3)
          o4 = self.conv4(in4)
          b,c,h,w = in1.shape
          o2  = F.interpolate(o2, (h,w), mode='bilinear',align_corners=False)
          o3  = F.interpolate(o3, (h,w), mode='bilinear',align_corners=False)
          o4  = F.interpolate(o4, (h,w), mode='bilinear',align_corners=False)
          fuse = torch.cat((o1,o2,o3,o4),1)
          fuse = self.fusion(fuse)
          out1 = self.oc1(fuse)
          out2 = self.oc2(out1)
          out3 = self.oc3(out2)
          return fuse, out1, out2, out3

class Exp3(nn.Module):
  def __init__(self,incs):
      super(Exp3, self).__init__() 
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.fc = nn.Sequential(
                nn.Linear(incs, incs // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(incs // 4, incs, bias=False),
                nn.Sigmoid()
      )
      
      self.softmax=nn.Softmax2d()
  def forward(self, x ):
      b, c, _, _ = x.size()
      y = self.avg_pool(x).view(b, c)
      y = self.fc(y).view(b, c, 1, 1).expand_as(x)      
      fore = x * y
      back = x * (1.0-y)
      map1 = fore.sum(dim=1).unsqueeze(dim=1)
      map2 = back.sum(dim=1).unsqueeze(dim=1)
      maps=torch.cat((map1, map2), 1)
      maps = self.softmax(maps)
#      return fore * maps[:, 0:1, :, :], back*maps[:, 1:2, :, :], maps
      out = fore * maps[:, 0:1, :, :] + back*maps[:, 1:2, :, :]
      return out, maps
  
  def forward_for_visual(self, x ):
      b, c, _, _ = x.size()
      y = self.avg_pool(x).view(b, c)
      y = self.fc(y).view(b, c, 1, 1)    
      yy = y
      y= y.expand_as(x) 
      fore = x * y
      back = x * (1.0-y)
      map1 = fore.sum(dim=1).unsqueeze(dim=1)
      map2 = back.sum(dim=1).unsqueeze(dim=1)
      maps=torch.cat((map1, map2), 1)
      maps = self.softmax(maps)
#      return fore * maps[:, 0:1, :, :], back*maps[:, 1:2, :, :], maps
      out = fore * maps[:, 0:1, :, :] + back*maps[:, 1:2, :, :]
      return fore,back, out, maps, yy
##对 map 进行监督  
class Decoder12(nn.Module):
      def __init__(self,incs, incs2, outcs):
            super(Decoder12, self).__init__() 
            self.guid = ReGuide1(incs,incs)
            self.body4 = nn.Sequential(_layers.BasicBlock(incs[4],outcs[4]),
                                        _layers.BasicBlock(outcs[4],outcs[4]))
            self.body3 = nn.Sequential(_layers.BasicBlock(incs[3]+outcs[4],outcs[3]),
                                       _layers.BasicBlock(outcs[3],outcs[3]),
                                       _layers.BasicBlock(outcs[3],outcs[3]))
            self.body2 = nn.Sequential(_layers.BasicBlock(incs[2]+outcs[3],outcs[2]),
                                       _layers.BasicBlock(outcs[2],outcs[2]))
            self.body1 = nn.Sequential(_layers.BasicBlock(incs[1]+outcs[2],outcs[1]),
                                       _layers.BasicBlock(outcs[1],outcs[1]))
            self.exp1 = Exp3(incs[1])
            self.body0 = nn.Sequential(_layers.conv3x3xbnxrelu(outcs[1],outcs[0]),
                                       _layers.conv3x3xbnxrelu(outcs[0],outcs[0]),
                                       _layers.conv3x3(incs[0],1),
                                       nn.Sigmoid())    
            self.p2 = nn.Sequential(_layers.conv3x3(128,1), nn.Sigmoid())
            self.p = Predict(incs, 1)
      def forward(self, in1, in2, in3, inc4):
          gc1, gc2,gc3,gc4 = self.guid(in1,in2,in3,inc4)
          fuse = inc4 + gc4
          out4 = self.body4(fuse)
          out3 = _layers.interpolate(out4)
          out3 = torch.cat((out3, in3 + gc3),1)
          out3 = self.body3(out3)
          out2 = _layers.interpolate(out3)
          out2 = torch.cat((out2, in2 + gc2 ),1)
          out2 = self.body2(out2)
          out1 = _layers.interpolate(out2)
          out1 = torch.cat((out1, in1 + gc1),1)
          out1 = self.body1(out1)
          out1, maps = self.exp1(out1 )
          out0 = _layers.interpolate(out1)
          out0 = self.body0(out0)
          out1, out2, out3, out4 = self.p(out1, out2, out3, out4)
          p2 = self.p2(gc1)
          return out0,p2, out1, out2, out3, out4, maps
      def forward_for_visual(self, in1, in2, in3, inc4):
          gc1, gc2,gc3,gc4 = self.guid(in1,in2,in3,inc4)
          fuse = inc4 + gc4
          out4 = self.body4(fuse)
          out3 = _layers.interpolate(out4)
          out3 = torch.cat((out3, in3 + gc3),1)
          out3 = self.body3(out3)
          out2 = _layers.interpolate(out3)
          out2 = torch.cat((out2, in2 + gc2 ),1)
          out2 = self.body2(out2)
          out1 = _layers.interpolate(out2)
          out1 = torch.cat((out1, in1 + gc1),1)
          out1 = self.body1(out1)
          fore,back, out1_1, maps, y= self.exp1.forward_for_visual(out1 )
          out0 = _layers.interpolate(out1_1)
          out0 = self.body0(out0)
#          out1, out2, out3, out4 = self.p(out1, out2, out3, out4)
#          p2 = self.p2(gc1)
          return  in1, out2, out1, gc1,  in1 + gc1, out0     #out1,  fore,back, out1_1, maps, y
 
 ########################################################
########################################################
def get_decoder(in_channels, out_channels, decoder_type, inc2=None):
##########################使用第一层的特征#######################################
  if  decoder_type ==1:
   #''' 最基本的Decoder, 传统的解码器''''
    net = Decoder0 (in_channels, out_channels) 
  if  decoder_type ==13:
    ##对 map 进行监督   
    net =  Decoder12 (in_channels,inc2, out_channels)

  return net
      

def main():
    x4=torch.rand(2,64,8,8) #随便定义一个输入  
    x3=torch.rand(2,64,16,16) #随便定义一个输入  
    x2=torch.rand(2,64,32,32) #随便定义一个输入
    x1=torch.rand(2,64,64,64) #随便定义一个输入
    x0=torch.rand(2,64,128,128) #随便定义一个输入

    in_channels = [64,64,64,64,64,64 ]
    out_channels = [64,64,64,64,64,64]
    #####################dense###############################################
    net = get_decoder(in_channels, out_channels,38,[3,6,4,3],1)
    #####################no_dense##############################################
#    net = get_decoder_aux([128,128, 128,128],[128,128, 128,128],  decoder_type)
#    net.forward(x0,x1,x2,x3, x4)
    out = net.forward(x0,x1,x2,x3, x4,x0,x1,x2,x3, x4)
    print(len(out))
    for i in out:
        print(i.shape)
#    writer = SummaryWriter(log_dir='./log')
##    writer.add_graph(net, (x0,x1,x2,x3, x4))
#    writer.add_graph(net, (x1,x2,x3, x4))
#    writer.close()  
#    print('done')   
if __name__ == '__main__':
      main()      
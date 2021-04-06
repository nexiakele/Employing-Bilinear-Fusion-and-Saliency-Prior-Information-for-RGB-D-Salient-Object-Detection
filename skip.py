# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:16:00 2018

@author: Dell
"""
import torch
from torch import nn
#from tensorboardX import SummaryWriter
import modules.layers as _layers
import torch.nn.functional as F
from CompactBilinearPooling_dsybaik import CompactBilinearPooling as CBP
###############################################################################
###############################################################################

class ASPP(nn.Module):
    def __init__(self, inc, outc, d=[1,3,5,7]):
        super(ASPP, self).__init__()
        self.conv1 = _layers.conv3x3_bn_relu_d(inc, outc//4, d[0])
        self.conv2 = _layers.conv3x3_bn_relu_d(inc, outc//4, d[1])
        self.conv3 = _layers.conv3x3_bn_relu_d(inc, outc//4, d[2])
        self.conv4 = _layers.conv3x3_bn_relu_d(inc, outc//4, d[3])
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat((x1, x2, x3, x4), 1)
        return out    
class Biliner_Fusion12(nn.Module):
    def __init__(self, inc, outc,d=[1,2,3,4]):
        super(Biliner_Fusion12, self).__init__()
        print('using bilinear 12 model')
        self.xb = CBP(inc,inc, outc, sum_pool=False)
        
        self.att_map = nn.Sequential(_layers.conv3x3xbnxrelu(outc+inc*2, outc),
                                     _layers.conv3x3xbnxrelu(outc, outc),
                                    nn.Conv2d(outc, 2, kernel_size=3, stride=1, padding=1),
                                    nn.Sigmoid())
        
        self.ff = _layers.BasicBlock(inc*2, outc)
        self.fusion = ASPP(outc*2, outc,[1,2,3,4])
    def forward(self, x, y):
        bf = self.xb(x,y)
        bf =F.normalize(bf, dim=1)
        ########attention
        bfin = torch.cat((bf, x, y),1)
        maps = self.att_map(bfin)
        fxy = torch.cat((x*maps[:,0:1,:,:], y*maps[:,1:2,:,:]),1)
        ########attention
        ff = self.ff(fxy)
        ########
        fusion = torch.cat((ff, bf), 1)
        fusion = self.fusion(fusion)
        return fusion    
    
    def forward_for_visual(self, x, y):
        bf = self.xb(x,y)
        bf =F.normalize(bf, dim=1)
        ########attention
        bfin = torch.cat((bf, x, y),1)
        maps = self.att_map(bfin)
        fxy = torch.cat((x*maps[:,0:1,:,:], y*maps[:,1:2,:,:]),1)
        ########attention
        ff = self.ff(fxy)
        ########
        fusion = torch.cat((ff, bf), 1)
        fusion = self.fusion(fusion)
        return fusion, bf    
class Skip22(nn.Module):
    def __init__(self, inc, outc, fm = Biliner_Fusion12):
        super(Skip22, self).__init__()  
        self.fuse1 = fm(inc[1], outc[1])
        self.fuse2 = fm(inc[2], outc[2])
        self.fuse3 = fm(inc[3], outc[3],[1,3,5,7])
        self.fuse4 = fm(inc[4], outc[4],[1,3,5,7])
    def forward(self,  r1, r2, r3, r4, d1, d2, d3, d4):
        out4 = self.fuse4(r4, d4)
        out3 = self.fuse3(r3, d3)
        out2 = self.fuse2(r2, d2)
        out1 = self.fuse1(r1, d1)
        return  out1, out2, out3, out4
    def forward_for_visual(self,  r1, r2, r3, r4, d1, d2, d3, d4):
#        out4 = self.fuse4(r4, d4)
#        out3 = self.fuse3(r3, d3)
#        out1, bf = self.fuse2.forward_for_visual(r2, d2)
        out1, bf = self.fuse1.forward_for_visual(r1, d1)
        return  out1, bf    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
def get_skip(ic, oc, skip_type=1):
 if  skip_type == 29:
     net = Skip22 (ic, oc, Biliner_Fusion12)     
 return net




###############################################################################
###############################################################################
###############################################################################
###############################################################################            
def main():
    x0=torch.rand(2,64,64,64) #随便定义一个输入
    x1=torch.rand(2,64,32,32)
    x2=torch.rand(2,64,16,16)
    x3=torch.rand(2,64,8,8)
    x4=torch.rand(2,64,4,4)
#    net = Encoder (basic_group_layer_bn,basic_encoder_block_cat_attention ,
#                   in_channels, out_channels,layers)
#    net = skip4 (skip_basic, [64,64,64,64,64],[128,128,128,128,128] )
#    net = get_skip([64,64,64,64,64,64,64],[64,64,64,64,64,64,64] ,1)
#    out = net.forward(x0,x1,x2,x3,x4,x0,x1,x2,x3,x4)
#    print(len(out))
    net = Biliner_Feature4(64,64)
    out = net.forward(x2, x2)
    print(out.shpae)
#    writer = SummaryWriter(log_dir='./log')
#    
#    writer.add_graph(net, (x0,x1,x2,x3,x4))
#    
#    writer.close()  
#    print('done')   
if __name__ == '__main__':
      main()
#    x0=torch.rand(2,64,10,10)
#    conv1 = nn.Conv2d(64, 64, kernel_size=(1,3), stride=1, padding=(0,0))
#    out = conv1(x0)
#    print(out.shape)
            
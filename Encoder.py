# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:14:35 2018

@author: Dell
"""
import torch
from torch import nn
#from tensorboardX import SummaryWriter
from modules import layers as _layer
from torchvision.models import vgg16_bn
import torch.nn.functional as F


############################Encoder############################################
def get_vgglayers(pretrained=False):
    features = list(vgg16_bn(pretrained=pretrained).features)[:43]
    layer1 = nn.Sequential(features[0],features[1],features[2],
                           features[3],features[4],features[5])
    features[7].stride = (2,2)
    layer2 = nn.Sequential(features[7],features[8],features[9],
                           features[10],features[11],features[12])
    features[14].stride = (2,2)
    layer3 = nn.Sequential(features[14],features[15],features[16],
                            features[17],features[18],features[19],
                            features[20],features[21],features[22])  
    features[24].stride = (2,2)
    layer4 = nn.Sequential(features[24],features[25],features[26],
                            features[27],features[28],features[29],
                            features[30],features[31],features[32]) 
    features[34].stride = (2,2)
    layer5 = nn.Sequential(features[34],features[35],features[36],
                            features[37],features[38],features[39],
                            features[40],features[41],features[42])   
    return layer1, layer2, layer3, layer4, layer5

class Vgg16forRGB(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(Vgg16forRGB, self).__init__()
        self.rlayer1,self.rlayer2,self.rlayer3,self.rlayer4,self.rlayer5=get_vgglayers(pretrained)
    def forward(self, x):
        o1 = self.rlayer1(x)
        o2 = self.rlayer2(o1)
        o3 = self.rlayer3(o2)
        o4 = self.rlayer4(o3)
        o5 = self.rlayer5(o4)
        return o1, o2, o3, o4, o5
    
########################depth input    
class encoder_d(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(encoder_d, self).__init__()
        self.RGB_encoder = Vgg16forRGB(pretrained)
        self.T_encoder   = Vgg16forRGB(pretrained)
    def forward(self, rgb, T):
        T = torch.cat((T, T, T), 1)
        r0, r1, r2, r3, r4 = self.RGB_encoder(rgb)
        t0, t1, t2, t3, t4 = self.T_encoder(T)
        return r0, r1, r2, r3, r4, t0, t1, t2, t3, t4       
    
def get_encoder(en_type = 1, trained = False):
    if trained:
        print('pre-trained weight')
    if en_type == 1:
        net = encoder_d(trained)      
    return net

def main():
    x=torch.rand(2,3,32,64) #随便定义一个输入
    y=torch.rand(2,1,32,64) #随便定义一个输入

    net = get_encoder(102)
    out = net.forward(x,y)
    for i in out:
        print(i.shape)
    writer = SummaryWriter(log_dir='./log')
    
    writer.add_graph(net, (x,y))
    
    writer.close()  
    print('done')   
if __name__ == '__main__':
      main()
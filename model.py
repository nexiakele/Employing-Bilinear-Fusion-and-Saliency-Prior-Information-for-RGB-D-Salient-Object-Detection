# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:13:27 2018

@author: Dell
"""

import torch
from torch import nn
#from tensorboardX import SummaryWriter
import numpy as np
import Encoder 
import skip  
import Decoder
###############################################################################
###########################model###############################################
class model1(nn.Module):
    def __init__(self,channels,en_type=0,s_tpye=0,de_type=0, Trained= False):
        super(model1, self).__init__() 
#        #######################################################################
        inc = channels['s_in']
        outc = channels['s_out']
        self.skip     = skip.get_skip(inc,outc,s_tpye)
#        #######################################################################
        in_c = channels['de_in']
        out_c = channels['de_out']
        self.Decoder  = Decoder.get_decoder(in_c,out_c,de_type)
        self._initialize_weights()
        self.Encoder  = Encoder.get_encoder(en_type, Trained) 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x,y): 
        r0,r1,r2,r3,r4,t0,t1,t2,t3,t4 = self.Encoder(x,y)
#        print(r0.shape, r1.shape, r2.shape, r3.shape, r4.shape)
#        print(t0.shape, t1.shape, t2.shape, t3.shape, t4.shape)
        out0, out1, out2, out3=self.skip(r1,r2,r3,r4,t1,t2,t3,t4)
#        print(s0.shape,s1.shape,s2.shape,s3.shape,s4.shape)
        out=self.Decoder(out0, out1, out2, out3)
        return out

class model2(nn.Module):
    def __init__(self,channels,en_type=0,s_tpye=0,de_type=0, Trained= False):
        super(model2, self).__init__() 
#        #######################################################################
        inc = channels['s_in']
        outc = channels['s_out']
        self.skip     = skip.get_skip(inc,outc,s_tpye)
#        #######################################################################
        in_c = channels['de_in']
        in_c2 = channels['de_in2']
        out_c = channels['de_out']
        self.Decoder  = Decoder.get_decoder(in_c,out_c,de_type,in_c2)
        self._initialize_weights()
        self.Encoder  = Encoder.get_encoder(en_type, Trained) 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x,y, is_visual = False): 
        r0,r1,r2,r3,r4,t0,t1,t2,t3,t4 = self.Encoder(x,y)
#        print(r0.shape, r1.shape, r2.shape, r3.shape, r4.shape)
#        print(t0.shape, t1.shape, t2.shape, t3.shape, t4.shape)
        out0, out1, out2, out3=self.skip(r1,r2,r3,r4,t1,t2,t3,t4)
#        print(s0.shape,s1.shape,s2.shape,s3.shape,s4.shape)
        out=self.Decoder(out0, out1, out2, out3)
        return out
    
    
###############################################################################
###############################################################################
###############################################################################    
def get_model(model_tpye = 0, trained = False):      
  channels = {
          's_in' :   [64, 128, 256,  512,  512,  512],
          's_out' :  [64, 128, 256,  512,  512,  512],
          'de_in' :  [64, 128, 256,  512,  512,  512],
          'de_in2' :  [64, 64, 64,    64,  64,  64],
          'de_out' : [64, 128, 256,  512,  512,  512],
          }
  if  model_tpye == 59:
    # 用Compact Bilinear + attention
    net = model2(channels,1, 29, 13, Trained=trained)  
  return net     



###############################################################################
###############################################################################
###############################################################################      
def main():
    x=torch.rand(1,3,32,32) #随便定义一个输入
    y=torch.rand(1,1,32,32) #随便定义一个输入
    net =  get_model(59)        
    out= net.forward(x,y)
    print(len(out))
    for i in out:
        print(i.shape)
        
    ######################################
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
#        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
#        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k/1e6))
#    writer = SummaryWriter(log_dir='log')
#    
#    writer.add_graph(net, (x,y))
#    
#    writer.close()  
#    print('done')


if __name__ == '__main__':
    #test()
    main()
#   aa = ['a','b','c','d', 'dd', 'ff' , 'ee','fea']
#   
#   bb = np.array(aa)
#   np.random.shuffle(bb)
#   bb = bb.tolist()
#   print(bb.pop())
#   print(bb)
#   print(bb.pop())
#   print(bb)
#   print(bb.pop())
#   print(bb)
#   print(bb.pop())
   

    
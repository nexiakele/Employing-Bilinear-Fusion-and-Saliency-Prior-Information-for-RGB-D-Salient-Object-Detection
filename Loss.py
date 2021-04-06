# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:47:56 2018

@author: Dell
"""
import torch
from torch import nn
import torch.nn.functional as F
import modules.lovasz_losses as lo
class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss, self).__init__() 
        self.a = torch.Tensor([[0, 0, 0],
                               [1.0, 0, -1.0],
                               [0, 0, 0]])
        self.a = self.a.view((1,1,3,3)).cuda()
        self.b = torch.Tensor([[0, 1.0, 0],
                                [0, 0, 0],
                                [0, -1.0,0]])
        self.b = self.b.view((1,1,3,3)).cuda()
        self.loss  = nn.SmoothL1Loss()
    def forward(self, logits, targets):
          with torch.no_grad():
                logits_x = F.conv2d(logits, self.a)
                logits_y = F.conv2d(logits, self.b)
                logits_G = torch.sqrt(torch.pow(logits_x,2)+ torch.pow(logits_y,2))
                targets_x = F.conv2d(targets, self.a)
                targets_y = F.conv2d(targets, self.b)
                targets_G = torch.sqrt(torch.pow(targets_x,2)+ torch.pow(targets_y,2))
          loss = self.loss(logits_G, targets_G)
          return loss
            
class fuse_loss(nn.Module):
    def __init__(self):
        super(fuse_loss, self).__init__() 
        self.bceloss = nn.BCELoss()
        self.edge_loss  = edge_loss()
    def forward(self, x, y):
          loss1 = self.bceloss(x,y)
          loss2 = self.edge_loss(x, y)
          return loss1 + loss2
class multi_scale_loss(nn.Module):
    def __init__(self):
        super(multi_scale_loss, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        main_loss = self.loss1(out[0], gt[0])
        au1  = self.loss1(out[1], gt[1])*0.5
        au2  = self.loss2(out[2], gt[2])*0.5
        au3  = self.loss2(out[3], gt[3])*0.5
        record= [main_loss.item(),au1.item(),au2.item(),au3.item()]
        totall_loss = main_loss+au1+au2+au3
        return  totall_loss, record   
class multi_scale_loss2(nn.Module):
    def __init__(self):
        super(multi_scale_loss2, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        main_loss = self.loss1(out[0], gt[0])
        au1  = self.loss1(out[1], gt[1])*0.5
        au2  = self.loss1(out[2], gt[2])*0.5
        au3  = self.loss2(out[3], gt[3])*0.5
        au4  = self.loss2(out[4], gt[4])*0.5
        record= [main_loss.item(),au1.item(),au2.item(),au3.item(),au4.item()]
        totall_loss = main_loss+au1+au2+au3
        return  totall_loss, record
class multi_scale_loss3(nn.Module):
    def __init__(self):
        super(multi_scale_loss3, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        main_loss = self.loss1(out[0], gt[0])
        au1  = self.loss1(out[1], gt[0])*0.5
        au2  = self.loss1(out[2], gt[1])*0.5
        au3  = self.loss2(out[3], gt[2])*0.5
        au4  = self.loss2(out[4], gt[3])*0.5
        au5  = self.loss2(out[5], gt[4])*0.5
        record= [main_loss.item(),au1.item(),au2.item(),au3.item(),au4.item(),au5.item()]
        totall_loss = main_loss+au1+au2+au3
        return  totall_loss, record
    
class multi_scale_loss4(nn.Module):
    def __init__(self):
        super(multi_scale_loss4, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        main_loss = self.loss1(out[0], gt[0])
        au1  = self.loss1(out[1], gt[1])
        au2  = self.loss2(out[2], gt[2])
        record= [main_loss.item(),au1.item(),au2.item()]
        totall_loss = main_loss+au1+au2
        return  totall_loss, record   
    
class multi_scale_loss5(nn.Module):
    def __init__(self):
        super(multi_scale_loss5, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        tloss = self.loss1(out[0], gt[0])
        rau1  = self.loss1(out[1], gt[1])*0.5
        rau2  = self.loss2(out[2], gt[2])*0.5
        rau3  = self.loss2(out[3], gt[3])*0.5
        rau4  = self.loss2(out[4], gt[4])*0.5
        dau1  = self.loss1(out[5], gt[1])*0.5
        dau2  = self.loss2(out[6], gt[2])*0.5
        dau3  = self.loss2(out[7], gt[3])*0.5
        dau4  = self.loss2(out[8], gt[4])*0.5
        record= [tloss.item(),rau1.item(),rau2.item(),rau3.item(),rau4.item(),
                 dau1.item(),dau2.item(),dau3.item(),dau4.item()]
        totall_loss = tloss+rau1+rau2+rau3+rau4+dau1+dau2+dau3+dau4
        return  totall_loss, record  
    
class base1(nn.Module):
    def __init__(self):
        super(base1, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        au1  = self.loss1(out[0], gt[1])
        au2  = self.loss1(out[1], gt[2])
        au3  = self.loss2(out[2], gt[3])
        au4  = self.loss2(out[3], gt[4])
        totall_loss = au1+au2+au3+au4
        return  totall_loss
class multi_scale_loss6(nn.Module):
    def __init__(self):
        super(multi_scale_loss6, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2 = base1()
    def forward(self,out, gt):
        main_loss = self.loss1(out[0], gt[0])
        au1 = self.loss2(out[1:5],gt)*0.5
        au2 = self.loss2(out[5:9],gt)*0.5
        au3 = self.loss2(out[9:],gt)*0.5
        totall_loss = main_loss+au1+au2+au3
        return  totall_loss, [main_loss.item(), au1.item(), au2.item(), au3.item()]
    
class base2(nn.Module):
    def __init__(self):
        super(base2, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        au1  = self.loss1(out[0], gt[1])
        au2  = self.loss1(out[1], gt[2])
        au3  = self.loss2(out[2], gt[3])
        au4  = self.loss2(out[3], gt[4])
        au5  = self.loss2(out[4], gt[5])
        totall_loss = au1+au2+au3+au4+au5
        return  totall_loss
class multi_scale_loss7(nn.Module):
    def __init__(self):
        super(multi_scale_loss7, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2 = base2()
    def forward(self,out, gt):
        main_loss = self.loss1(out[0], gt[0])
        au1 = self.loss2(out[1:6],gt)*0.5
        au2 = self.loss2(out[6:11],gt)*0.5
        au3 = self.loss2(out[11:],gt)*0.5
        totall_loss = main_loss+au1+au2+au3
        return  totall_loss, [main_loss.item(), au1.item(), au2.item(), au3.item()]  
    
class multi_loss(nn.Module):
    def __init__(self):
        super(multi_loss, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                gr = F.interpolate(gt, (h,w))
            if i<2:
                loss = self.loss1(o, gr)
            else:
                loss = self.loss2(o, gr)
            if i!=0:    
                loss =loss*0.5
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r

class multi_loss3(nn.Module):
    def __init__(self):
        super(multi_loss3, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        b,c,hg,wg = gt.shape
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            if h != hg or w != wg:
                with torch.no_grad():
                    o = F.interpolate(o, (hg,wg))
            if i<2:
                loss = self.loss1(o, gt)
            else:
                loss = self.loss2(o, gt)
            all_loss+=loss/(2**i)
            loss_r.append(loss.item())
        return  all_loss, loss_r

    
def get_hard_sample(pred, gt):
    with torch.no_grad():
         loss = F.binary_cross_entropy(pred, gt, reduction='none')
    temp = 0.0
    value = -1
    for i, lo in enumerate(loss):
        lo_sum = lo.sum()
        if temp < lo_sum:
            temp = lo_sum
            value = i
    return value

class multi_loss2(nn.Module):
    def __init__(self):
        super(multi_loss2, self).__init__() 
        self.loss1 =   fuse_loss()
#        self.loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        hard = get_hard_sample(out[0], gt)
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                gr = F.interpolate(gt, (h,w))
            loss = self.loss1(o, gr)    
            if i!=0:    
                loss =loss*0.5
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r, hard
    
class multi_loss4(nn.Module):
    def __init__(self):
        super(multi_loss4, self).__init__() 
        self.loss1 =   fuse_loss()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        hard = get_hard_sample(out[0], gt)
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                gr = F.interpolate(gt, (h,w))
            loss = self.loss1(o, gr)
            if i!=0:    
                loss =loss*0.5
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r, hard
    
class multi_loss5(nn.Module):
    def __init__(self):
        super(multi_loss5, self).__init__() 
        self.loss1 =   fuse_loss()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                gr = F.interpolate(gt, (h,w))
            loss = self.loss1(o, gr)
            if i!=0:    
                loss =loss*0.5
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r

class fuse_loss2(nn.Module):
    def __init__(self):
        super(fuse_loss2, self).__init__() 
        self.bceloss = nn.BCELoss()
    def forward(self, x, y):
          loss1 = self.bceloss(x,y)
          loss2 = lo.lovasz_softmax(x, y, classes=[1], ignore=255)
#          print(loss1.item(), loss2.item())
          return loss1 + loss2
class multi_loss6(nn.Module):
    def __init__(self):
        super(multi_loss6, self).__init__() 
        self.loss1 =   fuse_loss2()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                gr = F.interpolate(gt, (h,w))
            loss = self.loss1(o, gr)
            if i!=0:    
                loss =loss*0.5
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r
    
class fuse_loss3(nn.Module):
    def __init__(self):
        super(fuse_loss3, self).__init__() 
        self.bceloss = nn.BCELoss()
        self.edge_loss  = edge_loss()
    def forward(self, x, y):
          loss1 = self.bceloss(x,y)
          loss2 = lo.lovasz_softmax(x, y, classes=[1], ignore=255)
          loss3 = self.edge_loss(x,y)
#          print(loss1.item(), loss2.item())
          return loss1 + loss2+loss3
class multi_loss7(nn.Module):
    def __init__(self):
        super(multi_loss7, self).__init__() 
        self.loss1 =   fuse_loss3()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                gr = F.interpolate(gt, (h,w))
            loss = self.loss1(o, gr)
            if i!=0:    
                loss =loss*0.5
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r
    
class multi_loss8(nn.Module):
    def __init__(self):
        super(multi_loss8, self).__init__() 
        self.loss1 =   fuse_loss2()
        self.loss2 = nn.SmoothL1Loss()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                    gr = F.interpolate(gt, (h,w))
            if c == 1:
                loss = self.loss1(o, gr)
                if i!=0:    
                    loss =loss*0.5
            else: 
                grs = torch.cat((gr, 1-gr),1)
                loss = self.loss2(o , grs)
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r

class multi_loss9(nn.Module):
    def __init__(self):
        super(multi_loss9, self).__init__() 
        self.loss1 =   fuse_loss()
        self.loss2 = nn.SmoothL1Loss()
    def forward(self,out, gt):
        all_loss = 0.0
        loss_r  = []
        for i, o in enumerate(out):
            b, c, h, w = o.shape
            with torch.no_grad():
                    gr = F.interpolate(gt, (h,w))
            if c == 1:
                loss = self.loss1(o, gr)
                if i!=0:    
                    loss =loss*0.5
            else: 
                grs = torch.cat((gr, 1-gr),1)
                loss = self.loss2(o , grs)
            all_loss+=loss
            loss_r.append(loss.item())
        return  all_loss, loss_r

    
##############################################################################
def get_loss(loss_type = 0):
      if loss_type == 1:  
            loss = multi_scale_loss()  
      elif loss_type == 2:  
            loss = multi_scale_loss2() 
      elif loss_type == 3:  
            loss = multi_scale_loss3()     
      elif loss_type == 4:  
            loss = multi_scale_loss4()    
      elif loss_type == 5:  
            loss = multi_scale_loss5()  
      elif loss_type == 6:  
            loss = multi_scale_loss6()  
      elif loss_type == 7:  
            loss = multi_scale_loss7()    
      elif loss_type == 8:  
            loss = multi_loss()  
      elif loss_type == 9:  
            loss = multi_loss2()             
      elif loss_type == 10:  
            loss = multi_loss3()  
      elif loss_type == 11:  
            loss = multi_loss4() 
      elif loss_type == 12:  
            loss = multi_loss5() 
      elif loss_type == 13:  
            loss = multi_loss6() 
      elif loss_type == 14:  
            loss = multi_loss7()     
      elif loss_type == 15:  
            loss = multi_loss8() 
      elif loss_type == 16:  
            loss = multi_loss9() 
      return loss
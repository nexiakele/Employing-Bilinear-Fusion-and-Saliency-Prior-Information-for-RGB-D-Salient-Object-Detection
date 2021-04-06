# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:01:05 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:46:16 2018

@author: Dell
"""
#############################################
import time
import numpy as np
#############################################
import torch
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import matplotlib.image  as mpimg
#############################################
import tools.Tools as tool
import tools.litte_dataset_load as data_loader
from tools.metrics import Eval_tool
#############################################
from model import get_model
import Loss as ls
import Config
import os
import torch.nn.functional as F
#############################################
def save_d(img_tensor, dir_path):
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()
    out = out[0]
    out =  out[0]
    out[out > 1] = 1
    out[out<-1] = -1
    mpimg.imsave(dir_path,out, cmap='gray')
def val_iter(net, val_data_loader, device, dir_path, eval_tool):
    net.eval()
    with torch.no_grad():    
    #######验证过程
        for i_batch, sample_batched in enumerate(val_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].to(device)
            thermal = sample_batched['thermal'].to(device)
            gt = sample_batched['label'].to(device)
            name = sample_batched['name']
            h  = sample_batched['height']
            w  = sample_batched['width']
            out  = net(rgb, thermal)
            out0 = out[0]
#            out0 = F.interpolate(out0, (h, w), mode='bilinear')
            save_d(out0, dir_path +'/' + name[0] + '.png')
            eval_tool.run_eval(out0, gt)
            ############每个batch的结果统计和输出##########
            if i_batch % 100 == 0 and i_batch > 0:
                print('-->step:' , i_batch, 'done!')
    ##################每个epoch的损失统计和打印#############################
    mae, maxF, Sm = eval_tool.get_score()
    print('mae:', mae, 'max F measure:', maxF, 'S measure:' , Sm)
    return  mae, maxF, Sm
             
def val_iter1(net, val_data_loader, device, dir_path, eval_tool):
    net.eval()
    with torch.no_grad():    
    #######验证过程
        for i_batch, sample_batched in enumerate(val_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].to(device)
            thermal = sample_batched['thermal'].to(device)
            gt = sample_batched['label'].to(device)
            name = sample_batched['name']
            out  = net(rgb, thermal)
            out0 = out
            save_d(out0, dir_path +'/' + name[0] + '.png')
            eval_tool.run_eval(out0, gt)
            ############每个batch的结果统计和输出##########
            if i_batch % 200 == 0 and i_batch > 0:
                print('-->step:' , i_batch, 'done!')
    ##################每个epoch的损失统计和打印#############################
    mae, maxF, Sm = eval_tool.get_score()
    print('mae:', mae, 'max F measure:', maxF, 'S measure:' , Sm)
    return  mae, maxF, Sm
          
def train(args = Config.Config(), val_fun=val_iter, datsets='SSD'):
    args.name= 'infence' 
###############################################################################
#####################读取数据###################################################
    Date_File = "/media/hpc/data/work/dataset/RGB-D_saliency/"
    image_h, image_w = data_loader.get_Parameter()
####################读取测试数据###################################################
    val_dataset = data_loader.data_loader(Date_File,'test', subset=datsets,
                                    transform=transforms.Compose([
                                       data_loader.scaleNorm(image_w, image_h, (1, 1.2), False),
                                       data_loader.ToTensor(),
                                       ]))
    val_data_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=2)
###############################################################################
################################准备设备########################################
    if  args.is_cuda and torch.cuda.is_available():
          device = torch.device("cuda", args.device)
          torch.cuda.set_device(args.device)
    else:
          device = torch.device("cpu")
###############################################################################
################################加载模型########################################
    net = get_model(args.model_type)
    net.to(device)
    
    print('训练模型为: ', args.model_type, 'GPU:' ,  device, '图像大小', image_h, image_w)
################################加载优化器######################################
    tool.load_ckpt(net, None, args.last_ckpt, device)
    time_dir = tool.make_infence_dir(args)
    eval_tool = Eval_tool()
    eval_tool.reset()
    begin_time = time.time()
    mae, maxF, Sm = val_fun(net, val_data_loader, device, time_dir, eval_tool)
    totall_time =  time.time()-begin_time
    avg_time = totall_time / len(val_data_loader)          
    print(avg_time)
def get_infence(args, dataset = 1,model_tpye = 0):      
  if model_tpye == 1:
    train(args,val_iter, dataset)
  if model_tpye == 2:
    train(args, val_iter1, dataset) 
if __name__ == '__main__':
    train()
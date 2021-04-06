# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:21:35 2018

@author: Dell
"""
import numpy as np
from torch import nn
import torch
import os
from torch.nn import init
import matplotlib.pyplot as plt
import time
from skimage import io
############################################################
#####################备份与恢复##############################
#########备份########
def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch):
    # usually this happens only on the start of a epoch
    state = {
        'global_step': global_step,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch)
    path = ckpt_dir + '/' + ckpt_model_filename
    time.sleep(5)
    torch.save(state, path)
    #创建识别文件
    check_path = ckpt_dir + '/0-0-0-0-0-0-0-0-0.txt'
    if not os.path.exists(check_path):
          with open(check_path, 'w') as f:
                f.write('check')
    print('===> {:>2} has been successfully saved'.format(path))
#########恢复########
def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("===> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file,map_location='cuda:0')
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("===> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("===> no checkpoint found at '{}'".format(model_file))
        os._exit(0)
############################################################
#################创建文件目录#################################
def make_dir(args):
    #创建类型目录， name =  aux or base
    father_dir =  './result/' + str(args.dataset)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    father_dir =  father_dir + '/' + str(args.name)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建网络类型目录
    father_dir =   father_dir + '/model_tpye_'+ str(args.model_type)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建时间目录
    dirs = os.listdir( father_dir )
    trian_times = max(len(dirs), 1)
    time_dir = father_dir + '/' + str(trian_times)
    if os.path.exists(time_dir+'/ckpt/0-0-0-0-0-0-0-0-0.txt'):
          trian_times = len(dirs) + 1
    time_dir = father_dir + '/' + str(trian_times)
    if not os.path.exists(time_dir) :
        os.mkdir(time_dir)
    summary_dir = time_dir + '/summary_' + str(args.model_type)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    ckpt_dir =  time_dir + '/ckpt'
    if not os.path.exists(ckpt_dir):
              os.mkdir(ckpt_dir)
    with open(time_dir + '/remarks.txt' , 'w') as f:
          f.write(args.get_string())
    return summary_dir, ckpt_dir, time_dir
############################################################
def make_infence_dir(args):
    #创建类型目录， name =  aux or base
    father_dir =  './result/' + str(args.dataset)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    father_dir =  father_dir + '/' + str(args.name)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建网络类型目录
    father_dir =   father_dir + '/model_tpye_'+ str(args.model_type)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建时间目录
    father_dir =  father_dir + '/' + str(args.train_time)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    time_dir = father_dir + '/model_type_'+ str(args.model_type)+ '_epoch_'+ str(args.epochs)
    if not os.path.exists(time_dir) :
        os.mkdir(time_dir)
    return time_dir
##############得到ckpt目录###########################
def get_dir_through_ckpt(ckpt_path,args):
      ckpt_dir = os.path.dirname(ckpt_path)
      time_dir = os.path.dirname(ckpt_dir)
      summary_dir =  time_dir + '/summary_' + str(args.model_type)
      return summary_dir, ckpt_dir, time_dir 

############################################################
###################网络权重初始化#############################
#######################################
# weight init
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
              init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)    
#######################################
def xavier(param):
    init.xavier_uniform_(param)
#######################################
def Gassion(param):
    init.normal_(param, mean=0, std=1) 
############################################################
############################################################
def is_cpkt(epoch, args):
      if epoch ==0: 
            return False
      if epoch + 1 == args.epochs:
            return True
      if epoch < args.start_ckpt_epoch:
            return epoch % args.ckpt_decay1 == 0
      else:
            return epoch % args.ckpt_decay2 == 0
def onehot_mask(mask, num_cls=21):
    """
    :param mask: label of a image. tensor shape:[h, w]
    :param num_cls: number of class. int
    :return: onehot encoding mask. tensor shape:[num_cls, h, w]
    """
    b, h, w = mask.shape
    mask_onehot = torch.zeros_like(mask).unsqueeze(1).expand(b, num_cls, h, w)
    mask_onehot = mask_onehot.scatter_(1, mask.long().unsqueeze(1), 1)
    return mask_onehot
############################################################
############################################################
if __name__ == '__main__':
     string = '/media/hpc/data/work/work/work/result/nju2000/au/model_tpye_14/2019-01-02-09-09/ckpt/ckpt_epoch_249.00.pth'
#    print(string.split('/')[-3])[[1,0,0],[1,0,0],[0,0,0]],[[1,0,0],[1,0,0],[1,0,0]]
    
#      img = np.array([[[[1,0,0],[0,0,0],[0,0,0]]],
#                      [[[1,0,0],[1,0,0],[0,0,0]]],
#                      [[[1,0,0],[1,0,0],[1,0,0]]],
#                      [[[1,1,1],[1,1,1],[1,1,1]]]])
#      img = torch.from_numpy(img).float()
##      print(img[3,:,:,:].sum())
##      print(img.sum(dim=0))
##      print(img.sum(dim=1))
##      print(img.sum(dim=2))
##      print(img.sum(dim=3))

     print(setting)

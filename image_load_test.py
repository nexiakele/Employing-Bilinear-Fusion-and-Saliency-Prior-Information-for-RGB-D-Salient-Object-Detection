# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:22:49 2018

@author: Dell
"""
import torchvision.transforms as transforms
import tools.litte_dataset_load as data_loader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torch
import cv2
def depthp(depth):
      maxd = depth.max()
      mind  = depth.min()
      depth = -(depth-maxd) / (maxd-mind)
      return depth
def show(img_tensor, mode = 'rgb'):
    out = img_tensor.detach().numpy()
    if mode  == 'rgb':
          for i in out:
                plt.imshow(i.transpose((1, 2, 0)))
                plt.show()
    elif mode  == 'depth':
          for i in out:
                plt.imshow(i[0], cmap='gray')
                plt.show()

    else:
          print('wrong mode!')
          
def save(img_tensor, mode, dir_path,name):
    out = img_tensor.detach().numpy()
    if mode  == 'rgb':
          for i,n in zip(out, name):
                io.imsave(dir_path+n+'.jpg',i.transpose((1, 2, 0)))
    elif mode  == 'depth':
          for i,n in zip(out, name):
                io.imsave(dir_path+n+'.png',i[0])
    else:
          print('wrong mode!')

def save_all(sample):
      dir_name = {
            'rgb' : './save_test/rgb/',
            'depth': './save_test/depth/',
            'gt' : './save_test/gt/',
            'gt0' : './save_test/gt0/',
            'gt1' : './save_test/gt1/',
            'gt2' : './save_test/gt2/',
            'gt3' : './save_test/gt3/',
            'gt4' : './save_test/gt4/',
            }
      
      rgb = sample['image']
      depth = sample['thermal']
      gt = sample['label']
      gt2 = sample['label2']
      gt4 = sample['label4']
#      gt16 = sample['label16']
#      gt32 = sample['label32']
      name = sample['name']
      rgb = rgb.clamp(min=-1,max=1)
#      depth = depthp(depth)
      save(rgb,'rgb',dir_name['rgb'], name )
      save(depth,'rgb',dir_name['depth'], name )
      save(gt,'depth',dir_name['gt'], name )
      save(gt2,'depth',dir_name['gt0'], name )
      save(gt4,'depth',dir_name['gt1'], name )    
#      save(gt8,'label',dir_name['gt2'], name )    
#      save(gt16,'label',dir_name['gt3'], name )   
#      save(gt32,'label',dir_name['gt4'], name )  
def test():
      flag =   True # False True
      if flag:
          Date_File = 'E:/数据库/多模态显著性/RGB-T/'
#          Date_File = 'E:/数据库/多模态显著性/NPRL/NPRL/'
#          Date_File = 'E:/数据库/多模态显著性/nju2000/nju2000/'
          Date_File = 'E:/数据库/多模态显著性/rgbd_saliency_datasets/'
      else:    
          Date_File = "/media/hpc/data/work/dataset/RGB-T/"
          Date_File = "/media/hpc/data/work/dataset/nju2000/"
      h, w = data_loader.get_Parameter()    
      train_dataset = data_loader.data_loader(Date_File,'test', 'LFSD',
                                    transform=transforms.Compose([
                                           data_loader.scaleNorm(w, h, (1, 1.2), True),
                                           data_loader.RandomCrop(w, h),
                                           data_loader.RandomRotate(10),
                                           data_loader.RandomFlip(),
                                           data_loader.RandomGaussianBlur(),
                                           data_loader.RandomAdjust(),
                                           data_loader.ToTensor(),
#                                       data_loader.RandomRotate(10),     
#                                       data_loader.scaleNorm(w, h, (1, 1.4), True),
#                                       data_loader.RandomFlip(),
#                                       data_loader.RandomCrop(w, h),
#                                       data_loader.RandomGaussianBlur(),
#                                       data_loader.RandomAdjust(),
#                                       data_loader.ToTensor(),
                                       ]))
      train_data_loader = DataLoader(train_dataset, 1,shuffle=True, num_workers=1)
      record = []

      for i_batch, sample_batched in enumerate(train_data_loader):
            rgb = sample_batched['image']
            thermal = sample_batched['thermal']
            gt = sample_batched['label']
            gt2 = sample_batched['label2']
            gt4 = sample_batched['label4']
            gt8 = sample_batched['label8']
            gt16 = sample_batched['label16']
            if flag:
                xx = dct.dct_2d(rgb[0,0,:,:])
                show(xx, mode='depth')
#                show(rgb, mode='rgb')
#                show(thermal, mode='depth')
#                show(gt, mode='depth')
##                print(rgb.shape, thermal.shape, gt.shape)
#                show(gt2, mode='depth')
#                show(gt4, mode='depth')
#                show(gt8, mode='depth')
#                show(gt16, mode='depth')
#                print(rgb.max(), rgb.min(), rgb.mean())
#                print(thermal.max(), thermal.min(), thermal.mean())
#                print(gt.max(), gt.min(), gt.mean())
                if i_batch > 20:
                   break
            else:
                print(thermal.max(), thermal.min(), thermal.max())
                save_all(sample_batched)

#      print('---end---')
      record = np.array(record)
      print(record.min())

if __name__ == '__main__':    
     test()

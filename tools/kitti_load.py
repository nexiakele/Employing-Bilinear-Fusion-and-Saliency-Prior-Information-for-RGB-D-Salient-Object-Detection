# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:36:21 2018

@author: Dell
"""


import torch
from skimage import io as sk_io
from skimage import color as sk_color
from skimage import transform as sk_transform
import numpy as np
import matplotlib
import random
###############################################################################
def get_Parameter(is_weight=False):
      image_h = 320
      image_w = 1216   
      max_disp = 100.0
      return image_h, image_w, max_disp
class data_loader():
    def __init__(self, data_path,  model= 'train' , transform=None):
        self.data_path = data_path
        if model == 'train':
              self.path_file  = data_path+'train.txt'
        elif model == 'val':
              self.path_file = data_path+'val.txt'
        elif model == 'test':
              self.path_file = data_path+'test.txt'
        else:
              print('model not exist!')
              
        self.path = np.loadtxt(self.path_file, dtype=str)
        self.transform = transform
    def __len__(self):
        return len(self.path)
    def __getitem__(self, idx):
        #选择一副图像
        p = self.path[idx]
        #图像路径和名称
        rgb_path = self.data_path +  p[0]
        depth_path = self.data_path + p[1]
        mask_path = self.data_path + p[2]
        name = p[0].split('/')[-1]
        #读取图像
        rgb = sk_io.imread(rgb_path)
        depth =sk_io.imread(depth_path,as_gray=True) * 1.0  / 256.
        gt = sk_io.imread(mask_path,as_gray=True) * 1.0 / 256.
        #处理数据
        if len(rgb.shape) == 2:
              rgb = sk_color.gray2rgb(rgb)
        sample = {'image': rgb, 'depth': depth, 'label': gt}
        if self.transform:
            sample = self.transform(sample)
        sample.update({'name' : name[0:-4]})
        return sample

class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __init__(self, image_h, image_w ):
          self.image_h = image_h
          self.image_w = image_w
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = sk_transform.resize(image, 
                                    (self.image_h, self.image_w), 
                                    order=1,
                                    mode='reflect', 
                                    preserve_range=True,
                                    anti_aliasing=False)
        # Nearest-neighbor
        depth = sk_transform.resize(depth, 
                                    (self.image_h, self.image_w), 
                                    order=0,
                                    mode='reflect', 
                                    preserve_range=True,
                                    anti_aliasing=False)
        label = sk_transform.resize(label, 
                                    (self.image_h, self.image_w), 
                                    order=0,
                                    mode='reflect', 
                                    preserve_range=True,
                                    anti_aliasing=False)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = sk_transform.resize(image, 
                                    (target_height, target_width),
                                    order=1, mode='reflect', 
                                    preserve_range=True,
                                    anti_aliasing=False)
        # Nearest-neighbor
        depth = sk_transform.resize(depth, 
                                    (target_height, target_width),
                                    order=0, 
                                    mode='reflect', 
                                    preserve_range=True,
                                    anti_aliasing=False)
        label = sk_transform.resize(label, 
                                    (target_height, target_width),
                                     order=0, 
                                     mode='reflect', 
                                     preserve_range=True,
                                     anti_aliasing=False)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, image_h, image_w):
        self.image_h = image_h
        self.image_w = image_w
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.image_h)
        j = random.randint(0, w - self.image_w)

        return {'image': image[i:i + self.image_h, j:j + self.image_w, :],
                'depth': depth[i:i + self.image_h, j:j + self.image_w],
                'label': label[i:i + self.image_h, j:j + self.image_w]}

class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()
        if random.random() > 0.75:
            image = np.flipud(image).copy()
            depth = np.flipud(depth).copy()
            label = np.flipud(label).copy()  
        return {'image': image, 'depth': depth, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, max_disp = 100.):
        self.max_disp = max_disp
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        mask = np.where(depth > 0.0, 1.0, 0.0)
        depth = np.expand_dims(depth, 0).astype(np.float) 
        mask = np.expand_dims(mask, 0).astype(np.float) 
        label = np.expand_dims(label, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'mask':  torch.from_numpy(mask).float(),
                'label': torch.from_numpy(label).float(),
                }

###############################################################################
###############################################################################

def test2():
      Date_File = 'E:/dataset/sun_rgbd/'

#      show_mask2(sample['rgb'],sample['gt'],37)
#      show(rgb/255)
#      hsv = RandomHSV((0.85, 1.25),(0.8, 1.2),(20, 30),1)
#      sample = hsv.__call__(sample)
#      gray = RandomGray()
#      sample = gray.__call__(sample)
      
##      norm = per_image_standardization()
##      
##      sample = norm.__call__(sample)
#      rgb = sample['rgb']
#      show(rgb/255)
##      to = ToTensor()
##      sample = to.__call__(sample)
##      zero = zero_one()
##      sample = zero.__call__(sample)
##      rgb = sample['rgb']
##      show_rgb(rgb)
##      print(rgb.shape)
if __name__ == '__main__':
#    test2()
    aa = np.array([[1,23,45,1],[2,3,5,1]])
    args = np.where(aa==1)
    aa[args]=0
    print(aa)
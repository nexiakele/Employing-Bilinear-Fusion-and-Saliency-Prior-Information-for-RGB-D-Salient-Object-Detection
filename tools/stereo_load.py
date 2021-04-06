# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:36:21 2018

@author: Dell
"""


import torch
import numpy as np
import matplotlib
import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
###############################################################################
def get_Parameter(is_weight=False):
      image_h = 224
      image_w = 224   
      return image_h, image_w
class data_loader():
    def __init__(self, data_path,  model= 'test' , subset=1, transform=None):
        self.data_path = data_path
        if  model == 'test':
              self.path_file = data_path+'test.txt'
        else:
              print('model not exist!')
        print(self.path_file)      
        self.path = np.loadtxt(self.path_file, dtype=str)
        self.transform = transform
    def __len__(self):
        return len(self.path)
    def __getitem__(self, idx):
        #选择一副图像
        p = self.path[idx]
        #图像路径和名称
        rgb_path = self.data_path +  p[0]
        thermal_path = self.data_path + p[1]
        mask_path = self.data_path + p[2]
        name = p[0].split('/')[-1]
        #读取图像
        rgb = Image.open(rgb_path).convert('RGB')
        thermal = Image.open(thermal_path).convert('L')
        gt = Image.open(mask_path).convert('L')
        #保存图像尺寸
        w , h = rgb.size
        #处理数据
        sample = {'image': rgb, 'thermal': thermal, 'label': gt}
        if self.transform:
            sample = self.transform(sample)
        sample.update({'name' : name[0:-4],
                       'width' : w,
                       'height': h})
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

        return {'image': img_new, 'thermal': sample['thermal'], 'label': sample['label']}


class scaleNorm(object):
    def __init__(self, image_w, image_h, scale, is_scale=True):
        self.image_h = image_h
        self.image_w = image_w
        self.scale_low = min(scale)
        self.scale_high = max(scale)
        self.is_scale  =is_scale
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        
        if self.is_scale:
            target_scale = random.uniform(self.scale_low, self.scale_high)
            target_height = int(round(target_scale * self.image_h))
            target_width = int(round(target_scale * self.image_w ))
        else:
            target_height = self.image_h
            target_width = self.image_w
        # Bi-linear
        image = image.resize((target_width, target_height), Image.BILINEAR)
        thermal = thermal.resize((target_width, target_height), Image.BILINEAR)
        # Nearest-neighbor
        label= label.resize((target_width, target_height), Image.NEAREST)
        return {'image': image, 'thermal': thermal, 'label': label}

class RandomCrop(object):
    def __init__(self, image_w, image_h):
        self.image_h = image_h
        self.image_w = image_w
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        w , h = image.size
        i = random.randint(0, w - self.image_w)
        j = random.randint(0, h - self.image_h)
        image  = image.crop((i, j, i + self.image_w, j + self.image_h))
        thermal = thermal.crop((i, j, i + self.image_w, j + self.image_h))
        label = label.crop((i, j, i + self.image_w, j + self.image_h))
        return {'image': image, 'thermal': thermal, 'label': label}

class RandomFlip(object):
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.75:
            image   = image.transpose(Image.FLIP_TOP_BOTTOM)
            thermal = thermal.transpose(Image.FLIP_TOP_BOTTOM)
            label   = label.transpose(Image.FLIP_TOP_BOTTOM)
        return {'image': image, 'thermal': thermal, 'label': label}

class RandomRotate(object):
    def __init__(self, degree=25):
        self.degree = degree
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        static = random.random()
        if  static < 0.2:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            image   = image.rotate(rotate_degree, Image.BILINEAR)
            thermal = thermal.rotate(rotate_degree, Image.BILINEAR)
            label = label.rotate(rotate_degree, Image.NEAREST)
        elif static < 0.4:
            image   = image.rotate(90, Image.BILINEAR)
            thermal = thermal.rotate(90, Image.BILINEAR)
            label = label.rotate(90, Image.NEAREST) 
        elif static < 0.6:
            image   = image.rotate(-90, Image.BILINEAR)
            thermal = thermal.rotate(-90, Image.BILINEAR)
            label = label.rotate(-90, Image.NEAREST)   
        return {'image': image, 'thermal': thermal, 'label': label}
#class RandomRotate2(object):
#    def __init__(self, degree=25):
#        self.degree = degree
#
#    def __call__(self, sample):
#        image, thermal, label = sample['image'], sample['thermal'], sample['label']
#        if random.random() < 0.25:
#            w, h = image.size
#            rotate_degree = random.uniform(-1*self.degree, self.degree)
#            image   = image.rotate(rotate_degree, Image.BILINEAR, True)
#            thermal = thermal.rotate(rotate_degree, Image.BILINEAR, True)
#            label = label.rotate(rotate_degree, Image.NEAREST, True)
#            image = self.centercrop(image, w,h)
#            thermal = self.centercrop(thermal, w,h)
#            label = self.centercrop(label, w,h)
#        return {'image': image, 'thermal': thermal, 'label': label}
#    def centercrop(self, image, tw, th):
#        w, h = image.size
#        wc = (w-tw)
#        hc = (h-th)
#        image  = image.crop((wc, hc, w-wc, h-hc))
#        return image

class RandomGaussianBlur(object):
    def __call__(self, sample):
        image, thermal = sample['image'], sample['thermal']
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        if random.random() < 0.2:
            thermal = thermal.filter(ImageFilter.GaussianBlur(radius=random.random()))    
        sample['image'] = image 
        sample['thermal'] = thermal 
        return sample

class RandomAdjust(object):
    def __call__(self, sample):
        image, thermal = sample['image'], sample['thermal']
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        image = self.adjust_saturation(image)
        image = self.adjust_hue(image)
        thermal = self.adjust_brightness(thermal)
        thermal = self.adjust_contrast(thermal)
        sample['image'] = image 
        sample['thermal'] = thermal 
        return sample
    def adjust_brightness(self,img):
        if random.random() < 0.25:
            brightness_factor = random.uniform(0.7,1.3)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        return img        
    def adjust_contrast(self,img):
        if random.random() < 0.25:
            contrast_factor = random.uniform(0.7,1.3)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        return img   
    def adjust_saturation(self,img):
        if random.random() < 0.25:
            saturation_factor = random.uniform(0.7,1.3)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation_factor)
        return img  
    def adjust_hue(self,img):
        if random.random() < 0.25:
            input_mode = img.mode
            if input_mode in {'L', '1', 'I', 'F'}:
                return img
            hue_factor = random.uniform(-0.2,0.2)
            h, s, v = img.convert('HSV').split()
            np_h = np.array(h, dtype=np.uint8)
            # uint8 addition take cares of rotation across boundaries
            with np.errstate(over='ignore'):
                np_h += np.uint8(hue_factor * 255)
            h = Image.fromarray(np_h, 'L')
            img = Image.merge('HSV', (h, s, v)).convert(input_mode)
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, max_disp = 100.):
        self.max_disp = max_disp
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        w, h= label.size
        label2 = label.resize((w//2, h//2), Image.NEAREST)
        label4 = label.resize((w//4, h//4), Image.NEAREST)
        label8 = label.resize((w//8, h//8), Image.NEAREST)
        label16 = label.resize((w//16, h//16), Image.NEAREST)
        
        image = np.array(image).astype(np.float32).transpose((2, 0, 1)) / 255.0
        thermal = 1.0 - np.expand_dims(np.array(thermal), 0).astype(np.float32)/ 255.0
        label   = np.expand_dims(np.array(label),  0).astype(np.float32) / 255.0
        label2  = np.expand_dims(np.array(label2), 0).astype(np.float32) / 255.0
        label4  = np.expand_dims(np.array(label4), 0).astype(np.float32)/ 255.0
        label8  = np.expand_dims(np.array(label8), 0).astype(np.float32)/ 255.0
        label16  = np.expand_dims(np.array(label16), 0).astype(np.float32)/ 255.0
        
        image   = torch.from_numpy(image).float()
        thermal = torch.from_numpy(thermal).float()
        label   = torch.from_numpy(label).float()
        label2  = torch.from_numpy(label2).float()
        label4  = torch.from_numpy(label4).float()
        label8  = torch.from_numpy(label8).float()
        label16 = torch.from_numpy(label16).float()
        return {'image': image,
                'thermal': thermal,
                'label': label,
                'label2': label2,
                'label4': label4,
                'label8': label8,
                'label16': label16,
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
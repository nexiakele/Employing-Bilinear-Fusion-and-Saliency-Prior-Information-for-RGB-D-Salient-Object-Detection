# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:22:57 2019

@author: Dell
"""
import numpy as np
class label_tools(object):
    def __init__(self,n_classes):
          self.n_classes = n_classes
          self.cmap = self.color_map(self.n_classes+1)
    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0
        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])
        cmap = cmap / 255.0 if normalized else cmap
        return cmap

    def color_label(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l, 0]
            g[temp == l] = self.cmap[l, 1]
            b[temp == l] = self.cmap[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def label_to_input(self, labels):
            if len(labels.shape) == 3:
                  batch , im_h, im_w = labels.shape
                  new_labels = np.zeros((batch, self.n_classes, im_h, im_w))
                  for i in range(batch):
                        label = labels[i]
                        for cl in range(self.n_classes):
                              mask = label == cl
                              new_labels[i,cl, mask] = 1
            return new_labels
    def show_label(self, labels, is_tensor = False):
          if is_tensor :
                labels = labels.detach().cpu().numpy()
          for label in labels:
                label = self.color_label(label)
                plt.imshow(label)
                plt.show()
    def save_label(self, labels, names, path, is_tensor = False):
          if is_tensor :
                labels = labels.detach().cpu().numpy()
          for label, name in zip(labels, names):
                label_path = path + name + '.jpg'
                label = self.color_label(label)
                io.imsave(label_path, label)
                print('success save ', name )
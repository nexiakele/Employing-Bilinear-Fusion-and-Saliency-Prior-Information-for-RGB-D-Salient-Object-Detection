# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
import os
import time
import argparse
from train_nju import get_trianer
from Config import Config
from infence_nju import get_infence
from train_train import get_trianer1 
argement = argparse.ArgumentParser(description='运行参数')
argement.add_argument('--run-type', default=0, type=int, metavar='N',help='run-type default (0)')
argement.add_argument('--device', default=3, type=int, metavar='N',help='run-type default (0)')
run_args = argement.parse_args()
def get_infence_list(device, model_type, dataset, epochs, trian_time=1):
    infence_list = []
    for epoch in epochs:
        #[model_type,device,train_times, ckpt_epoch, dataset_name]
        infence_list.append([model_type, device, trian_time, epoch, dataset])
    return infence_list
if __name__ == '__main__':
    if  run_args.run_type == 1:        
        d = run_args.device
        conf = Config()
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[59, 15, d, 0.002, 200, 0.8, 19, 'train', 1, 75]
        conf.set_train_conf(setting)
        get_trianer1(conf, 5)  
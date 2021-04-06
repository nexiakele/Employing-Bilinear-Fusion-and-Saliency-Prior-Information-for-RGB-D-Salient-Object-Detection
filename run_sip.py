# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
import os
import time
import argparse
from infence_sip import get_infence
from Config import Config
argement = argparse.ArgumentParser(description='运行参数')
argement.add_argument('--run-type', default=0, type=int, metavar='N',help='run-type default (0)')
run_args = argement.parse_args()
def get_infence_list(device, model_type, dataset, dataset2, epochs, trian_time=1):
    infence_list = []
    for epoch in epochs:
        #[model_type,device,train_times, ckpt_epoch, dataset_name]
        infence_list.append([model_type, device, trian_time, epoch, dataset, dataset2])
    return infence_list
if __name__ == '__main__':
    if  run_args.run_type == 0 :
        d=1
        conf = Config()
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        epochs = [i for i in range(150, 200,2)]
        settings =  get_infence_list(d, 49, 'SIP', 'nju1', epochs, 1)
        for setting in settings:            
            conf.set_infence2(setting)
            get_infence(conf, 2, 1)              

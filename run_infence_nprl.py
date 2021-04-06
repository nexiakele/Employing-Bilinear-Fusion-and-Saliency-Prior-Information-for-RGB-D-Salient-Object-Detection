# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
#import os
#import time
import argparse
from infence_nlpr import get_infence
from Config import Config
argement = argparse.ArgumentParser(description='运行参数')
argement.add_argument('--run-type', default=0, type=int, metavar='N',help='run-type default (0)')
run_args = argement.parse_args()
def get_infence_list(device, model_type, dataset, epochs, trian_time=1):
    infence_list = []
    for epoch in epochs:
        #[model_type,device,train_times, ckpt_epoch, dataset_name]
        infence_list.append([model_type, device, trian_time, epoch, dataset])
    return infence_list
if __name__ == '__main__':

    if  run_args.run_type == 7 :
        d = 1
        conf = Config()
        epochs = [i for i in range(150, 199, 2)]
#        
#        settings =  get_infence_list(d, 42, 'nplr1', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 1, 1)  
#            
#        settings =  get_infence_list(d, 42, 'nplr2', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 2, 1)  
            
            
        settings =  get_infence_list(d, 59, 'nplr1', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 1, 1)  
            
        settings =  get_infence_list(d, 59, 'nplr2', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 2, 1)  
            
        settings =  get_infence_list(d, 59, 'nplr3', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 3, 1)  
            
#        settings =  get_infence_list(d, 51, 'nplr2', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 2, 1)  
#            
#        settings =  get_infence_list(d, 51, 'nplr1', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 1, 1)  
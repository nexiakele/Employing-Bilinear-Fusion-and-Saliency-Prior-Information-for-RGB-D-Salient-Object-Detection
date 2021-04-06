# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
import os
import time
import argparse
from infence_nju import get_infence
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
           
    if  run_args.run_type == 0 :  
        d = 0  
        conf = Config()
        epochs = [i for i in range(150, 199,2)]
        settings =  get_infence_list(d, 42, 'nju1', epochs, 2)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 1, 1)
    if  run_args.run_type == 1 :  
        d = 2  
        conf = Config()
        epochs = [i for i in range(150, 199,2)]
        settings =  get_infence_list(d, 42, 'nju2', epochs, 2)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 2, 1)     
    if  run_args.run_type == 2 :  
        d = 3  
        conf = Config()
        epochs = [i for i in range(150, 199,2)]
        settings =  get_infence_list(d, 42, 'nju3', epochs, 2)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 3, 1) 
    if  run_args.run_type == 3 :  
        d = 1 
        conf = Config()
        epochs = [i for i in range(150, 199,2)]
        settings =  get_infence_list(d, 42, 'nju4', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 4, 1) 
    if  run_args.run_type == 4 :  
        d = 0  
        conf = Config()
        epochs = [i for i in range(150, 199,2)]
        settings =  get_infence_list(d, 48, 'nju4', epochs, 2)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 4, 1) 
            
    if  run_args.run_type == 5 :  
        d = 1  
        conf = Config()
        epochs = [i for i in range(150, 199,2)]
        settings =  get_infence_list(d, 51, 'nju1', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 1, 1)  
#        conf = Config()
#        epochs = [i for i in range(150, 199,2)]
#        settings =  get_infence_list(d, 51, 'nju2', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 2, 1)  
            
    if  run_args.run_type == 6 :  
        d = 0  
        conf = Config()
        epochs = [i for i in range(15, 76,15)]
        settings =  get_infence_list(d, 52, 'nju1', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 1, 1)  
#        conf = Config()
#        epochs = [i for i in range(150, 199,2)]
#        settings =  get_infence_list(d, 51, 'nju4', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 4, 1) 
#        conf = Config()
#        epochs = [i for i in range(150, 199,2)]
#        settings =  get_infence_list(d, 42, 'nju2', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 2, 1)  
#
#        conf = Config()
#        epochs = [i for i in range(150, 199,2)]
#        settings =  get_infence_list(d, 42, 'nju3', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 3, 1)  
#
#        conf = Config()
#        epochs = [i for i in range(150, 199,2)]
#        settings =  get_infence_list(d, 42, 'nju4', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 4, 1)      
            
    if  run_args.run_type == 7 :  
        d = 2  
        conf = Config()
        epochs = [i for i in range(150, 199,5)]
        settings =  get_infence_list(d, 61, 'nju1', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 1, 1)  
            
    if  run_args.run_type == 8 :  
        d = 0  
        conf = Config()
        epochs = [i for i in range(5, 50,5)]
        settings =  get_infence_list(d, 43, 'nju1', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 1, 1)  
            
#        conf = Config()
#        epochs = [i for i in range(5, 50,5)]
#        settings =  get_infence_list(d, 44, 'nju1', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 1, 1)  
            
    if  run_args.run_type == 333 :  
        d = 3  
        conf = Config()
        epochs = [5]
        settings =  get_infence_list(d, 43, 'nju1', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 1, 3)  
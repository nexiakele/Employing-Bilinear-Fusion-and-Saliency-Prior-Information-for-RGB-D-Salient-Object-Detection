# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
import os
import time
import argparse
from train_nlpr import get_trianer
from Config import Config
from infence_nlpr import get_infence
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


    if  run_args.run_type == 1 :
        d = 0
        
        conf = Config()
        conf.batch_size=4
        conf.val_freq = 5
        conf.checkpoint=150
        conf.ckpt_decay1=25
        conf.ckpt_decay2=2
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[42, 15, d, 0.002, 200, 0.8, 18, 'nplr1', 1, 20]
        conf.set_train_conf(setting)
        get_trianer(conf, 1) 
    if  run_args.run_type == 2 :
        d = 2        
        
        conf = Config()
        conf.batch_size=4
        conf.val_freq = 5
        conf.checkpoint=150
        conf.ckpt_decay1=25
        conf.ckpt_decay2=2
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[42, 15, d, 0.002, 200, 0.8, 18, 'nplr2', 1, 20]
        conf.set_train_conf(setting)
        get_trianer(conf, 2) 
        
#    if  run_args.run_type == 3 :
#        d = 3        
        conf = Config()
        conf.batch_size=4
        conf.val_freq = 5
        conf.checkpoint=150
        conf.ckpt_decay1=25
        conf.ckpt_decay2=2
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[42, 15, d, 0.002, 200, 0.8, 18, 'nplr3', 1, 20]
        conf.set_train_conf(setting)
        get_trianer(conf, 3) 
        
        
    if  run_args.run_type == 3 :
        d = 1       
        conf = Config()
        conf.batch_size=4
        conf.val_freq = 5
        conf.checkpoint=150
        conf.ckpt_decay1=25
        conf.ckpt_decay2=2
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[51, 15, d, 0.002, 200, 0.8, 20, 'nplr3', 1, 20]
        conf.set_train_conf(setting)
        get_trianer(conf, 3)    
        
        conf = Config()
        conf.batch_size=4
        conf.val_freq = 5
        conf.checkpoint=150
        conf.ckpt_decay1=25
        conf.ckpt_decay2=2
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[51, 15, d, 0.002, 200, 0.8, 20, 'nplr2', 1, 20]
        conf.set_train_conf(setting)
        get_trianer(conf, 2) 
        
        
        
#        conf = Config()
#        conf.batch_size=4
#        conf.val_freq = 5
#        conf.checkpoint=150
#        conf.ckpt_decay1=25
#        conf.ckpt_decay2=2
#        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
#        setting=[51, 15, d, 0.002, 200, 0.8, 18, 'nplr1', 1, 20]
#        conf.set_train_conf(setting)
#        get_trianer(conf, 1) 
        
        conf = Config()
        epochs = [i for i in range(150, 199, 2)]
        
#        settings =  get_infence_list(d, 49, 'nplr1', epochs, 1)
#        for setting in settings:            
#            conf.set_infence(setting)
#            get_infence(conf, 1, 1)  
            
        settings =  get_infence_list(d, 49, 'nplr2', epochs, 2)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 2, 1)  
            
            
        settings =  get_infence_list(d, 49, 'nplr3', epochs, 1)
        for setting in settings:            
            conf.set_infence(setting)
            get_infence(conf, 3, 1)  
            
        
    if  run_args.run_type == 4 :
        d = 1       
        
#        conf = Config()
#        conf.batch_size=4
#        conf.val_freq = 5
#        conf.checkpoint=150
#        conf.ckpt_decay1=25
#        conf.ckpt_decay2=2
#        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
#        setting=[59, 15, d, 0.002, 200, 0.8, 20, 'nplr1', 1, 160]
#        conf.set_ckpt_train_conf(setting)
#        get_trianer(conf, 1)    
        
        
        conf = Config()
        conf.batch_size=4
        conf.val_freq = 5
        conf.checkpoint=150
        conf.ckpt_decay1=25
        conf.ckpt_decay2=2
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[59, 15, d, 0.002, 200, 0.8, 20, 'nplr2', 1, 20]
        conf.set_train_conf(setting)
        get_trianer(conf, 2)   

        conf = Config()
        conf.batch_size=4
        conf.val_freq = 5
        conf.checkpoint=150
        conf.ckpt_decay1=25
        conf.ckpt_decay2=2
        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        setting=[59, 15, d, 0.002, 200, 0.8, 20, 'nplr3', 1, 20]
        conf.set_train_conf(setting)
        get_trianer(conf, 3)         

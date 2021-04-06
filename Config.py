# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:06:29 2018

@author: Dell
"""
class Config(object):
    def __init__(self):
        ###################
        self.name = 'trian' 
        ###################
        self.model_type = 0 
        self.loss_type = 0 
        self.is_cuda = True
        self.device = 0
        ###################
        self.dataset = 'sun_rgbd'
        ###################
        self.lr=0.0001 
        self.weight_decay=0
        self.momentum = 0.9
        self.lr_decay_rate=0.8 
        self.lr_decay_epoch=30
        self.lr_start_decay_epoch=0
        ############################
        self.epochs=1000
        self.batch_size=4
        self.start_epoch=0
        ############################
        self.last_ckpt='' 
        self.checkpoint=True 
        self.ckpt_dir='./ckpt' 
        self.start_ckpt_epoch=150
        self.ckpt_decay1=15
        self.ckpt_decay2=2 
        self.summary_dir='./summary' 
        self.summary_epoch=1
        self.retrain_with_opt = False
        self.train_time = 1
        ############################
        self.val_freq = 5
    def set_train_conf(self, conf):
        #[model_type, loss_type,     device,         lr, 
        # epoch,      lr_decay_rate, lr_decay_epoch, dataset_name,
        #]
        self.model_type = conf[0]
        self.loss_type  = conf[1]
        self.device     = conf[2]
        self.lr         = conf[3]
        self.epochs     = conf[4]
        self.lr_decay_rate = conf[5]
        self.lr_decay_epoch = conf[6]
        self.dataset = conf[7]
        
    def set_ckpt_train_conf(self, conf):
        #[model_type, loss_type,     device,         lr, 
        # epoch,      lr_decay_rate, lr_decay_epoch, dataset_name,
        #  train_times, ckpt_epoch]
        self.model_type = conf[0]
        self.loss_type  = conf[1]
        self.device     = conf[2]
        self.lr         = conf[3]
        self.epochs     = conf[4]
        self.lr_decay_rate = conf[5]
        self.lr_decay_epoch = conf[6]
        self.last_ckpt = './result/'
        self.last_ckpt = self.last_ckpt + str(conf[7]) + '/trian/model_tpye_'
        self.last_ckpt = self.last_ckpt + str(self.model_type)+'/'+str(conf[8])
        self.last_ckpt = self.last_ckpt + '/ckpt/ckpt_epoch_'+str(conf[9])+'.00.pth'
        
    def set_infence(self, conf):
        #[model_type,device,train_times, ckpt_epoch, dataset_name]
        self.model_type = conf[0]
        self.device  = conf[1]
        self.train_time  = conf[2]
        self.epochs     = conf[3]
        self.dataset =    conf[4]
        self.set_ckpt_string()
    def set_infence2(self, conf):
        #[model_type,device,train_times, ckpt_epoch, dataset_name]
        self.model_type = conf[0]
        self.device  = conf[1]
        self.train_time  = conf[2]
        self.epochs     = conf[3]
        self.dataset =    conf[4]
        self.set_ckpt_string(conf[5])
    def get_string(self):
        string = ''
        for k, v in self.__dict__.items():
            string += k + ':' + str(v) + '\r\n'
        return string
    def set_ckpt_string(self, dataset=None):
        if dataset == None:
            dataset = self.dataset
        self.last_ckpt = './result/' + dataset + '/trian/model_tpye_'
        self.last_ckpt = self.last_ckpt + str(self.model_type)+'/'+ str(self.train_time)
        self.last_ckpt = self.last_ckpt + '/ckpt/ckpt_epoch_'+str(self.epochs)+'.00.pth'            
if __name__ == '__main__':
    conf = Config()
    string = conf.get_string()
    print(string)
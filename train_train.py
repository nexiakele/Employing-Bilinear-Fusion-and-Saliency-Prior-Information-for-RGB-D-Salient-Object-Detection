# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:46:16 2018

@author: Dell
"""
#############################################
import time
import numpy as np
#############################################
import torch
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
#############################################
import tools.Tools as tool
import tools.SOD_loader as data_loader
from tools.metrics import Eval_tool
#############################################
from model import  get_model
import Loss as ls
import Config
#############################################
def train_iter(net, train_data_loader, device, optimizer, criterion):
    train_loss = 0.0
    net.train()
    for i_batch, sample_batched in enumerate(train_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].to(device)
            thermal = sample_batched['thermal'].to(device)
            gt = sample_batched['label'].to(device)
            gt2 = sample_batched['label2'].to(device)
            gt4 = sample_batched['label4'].to(device)
            gt8 = sample_batched['label8'].to(device)
            gt16 = sample_batched['label16'].to(device)
            gt32 = sample_batched['label32'].to(device)
            ##################获取结果##################
            optimizer.zero_grad()
            out = net(rgb, thermal)

            ##################计算损失###################
            loss, l_re = criterion(out, (gt, gt2, gt4, gt8, gt16, gt32))
            ###################反向传播##################
            loss.backward()
            optimizer.step()
            ############每个batch的结果统计和输出##########
            train_loss += loss.item()
            if i_batch % 50 == 0 and i_batch > 0:
                lstr = '-'
                for i in l_re:
                    lstr = lstr + ' ' + str(i)
                lstr += '-'    
                print('step:',i_batch, 
                      'lr:', optimizer.param_groups[0]['lr'],
                      'loss:', train_loss / i_batch,
                      lstr)
                
    return train_loss / i_batch

def train_iter2(net, train_data_loader, device, optimizer, criterion):
    train_loss = 0.0
    net.train()
    for i_batch, sample_batched in enumerate(train_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].to(device)
            thermal = sample_batched['thermal'].to(device)
            gt = sample_batched['label'].to(device)
            gt2 = sample_batched['label2'].to(device)
            gt4 = sample_batched['label4'].to(device)
            gt8 = sample_batched['label8'].to(device)
            gt16 = sample_batched['label16'].to(device)
            gt32 = sample_batched['label32'].to(device)
            ##################获取结果##################
            optimizer.zero_grad()
            out = net(rgb, thermal)

            ##################计算损失###################
            loss, l_re = criterion(out, (gt, gt2, gt4, gt8, gt16, gt32))
            ###################反向传播##################
            loss.backward()
            optimizer.step()
            ############每个batch的结果统计和输出##########
            train_loss += loss.item()
            if i_batch % 100 == 0 and i_batch > 0:
                lstr = '--'
                for i in l_re:
                    lstr = lstr + ' ' + str(i)
                lstr += '--'    
                print('step:' , i_batch, 
                      'lr:', optimizer.param_groups[0]['lr'],
                      'loss:', train_loss / i_batch,
                      lstr)
                
    return train_loss / i_batch

def val_iter(net, val_data_loader, device, criterion, eval_tool):
    net.eval()
    val_loss = 0.0
    with torch.no_grad():    
    #######验证过程
        for i_batch, sample_batched in enumerate(val_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].to(device)
            thermal = sample_batched['thermal'].to(device)
            gt = sample_batched['label'].to(device)
            out  = net(rgb, thermal)
            out0 = out[0]
            loss = criterion(out0, gt)  
            
            eval_tool.run_eval(out0, gt)
            
            val_loss += loss.item()
            ############每个batch的结果统计和输出##########
            if i_batch % 100 == 0 and i_batch > 0:
                print('-->step:' , i_batch, 'done!','loss:',val_loss / i_batch)
    ##################每个epoch的损失统计和打印#############################
    avg_val_loss = val_loss/len(val_data_loader)
    mae, maxF, Sm = eval_tool.get_score()
    print('val_loss:', avg_val_loss, 'mae:', mae, 'max F measure:', maxF, 'S measure:' , Sm)
    return avg_val_loss, mae, maxF, Sm
             
def train_iter3(net, train_data_loader, device, optimizer, criterion):
    train_loss = 0.0
    net.train()
    for i_batch, sample_batched in enumerate(train_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].to(device)
            thermal = sample_batched['thermal'].to(device)
            gt = sample_batched['label'].to(device)
            ##################获取结果##################
            optimizer.zero_grad()
            out = net(rgb, thermal)

            ##################计算损失###################
            loss, l_re = criterion(out, gt)
            ###################反向传播##################
            loss.backward()
            optimizer.step()
            ############每个batch的结果统计和输出##########
            train_loss += loss.item()
            if i_batch % 100 == 0 and i_batch > 0:
                lstr = '--'
                for i in l_re:
                    lstr = lstr + ' ' + str(i)
                lstr += '--'    
                print('step:' , i_batch, 
                      'lr:', optimizer.param_groups[0]['lr'],
                      'loss:', train_loss / i_batch,
                      lstr)
                
    return train_loss / i_batch



def ckpt_load(args, net, optimizer, device):
    if args.last_ckpt is not '':
          print('=>6: 载入训练好的模型的参数')
          global_step, args.start_epoch = tool.load_ckpt(net, optimizer, args.last_ckpt, device)
          print('第',args.start_epoch,'加载完毕', '总步数为：', global_step)
          args.start_epoch+=1
          print('第',args.start_epoch,'开始训练')
          summary_dir , ckpt_dir, time_dir = tool.get_dir_through_ckpt(args.last_ckpt, args)
          ############记录器
          train_report     = np.load(summary_dir + '/train_report.npy').tolist()
          val_record = np.load(summary_dir + '/val_record.npy').tolist()
    else:
          #不使用预训练的模型，则创建新目录
          print('=>6: 建立新的模型，创建目录文件')
          summary_dir,ckpt_dir,time_dir=tool.make_dir(args)
          print('     ==>文件名:', summary_dir,ckpt_dir,time_dir)
          train_report = []
          val_record   = []  
    return train_report, val_record, summary_dir, ckpt_dir, time_dir                
def train(args = Config.Config(), train_fun=train_iter, val_fun=val_iter, datsets=1):
###############################################################################
#####################读取数据###################################################
#    Date_File = "/media/hpc/data/work/dataset/nju2000/"
##    Date_File = "/home/ly/disk1/work/dataset/nju2000/"
#    Date_File = "/data/HNC/dataset/RGBD_Saliency/nju2000/"
    Date_File = "/data/HNC/dataset/RGBD_SOD/"
    image_h, image_w = data_loader.get_Parameter()
    print('=>1: 输入图像参数', image_h, image_w)
#####################读取训练数据###################################################
    train_dataset = data_loader.data_loader(Date_File,'train', subset=datsets,
                               transform=transforms.Compose([
                                           data_loader.scaleNorm(image_w, image_h, (1, 1.2), True),
                                           data_loader.RandomRotate(10),
                                           data_loader.RandomCrop(image_w, image_h),
                                           data_loader.RandomFlip(),
                                           data_loader.RandomGaussianBlur(),
                                           data_loader.RandomAdjust(),
                                           data_loader.ToTensor(),
                                           ]))
    train_data_loader = DataLoader(train_dataset, args.batch_size,shuffle=True, num_workers=4)
#####################读取测试数据###################################################
    val_dataset = data_loader.data_loader(Date_File,'test',subset=2,
                                    transform=transforms.Compose([
                                       data_loader.scaleNorm(image_w, image_h, (1, 1.2), False),
                                       data_loader.ToTensor(),
                                       ]))
    val_data_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=2)
###############################################################################
################################准备设备########################################
    if  args.is_cuda and torch.cuda.is_available():
          device = torch.device("cuda", args.device)
          torch.cuda.set_device(args.device)
    else:
          device = torch.device("cpu")
    print('=>2: 使用的GPU为: ',  device)
###############################################################################
################################加载模型########################################
    net = get_model(args.model_type, True)
    net.to(device)
    print('=>3: 训练模型为: ', args.model_type)
################################加载损失函数####################################
    print('=>4: 损失函数为： ', args.loss_type)
    criterion = ls.get_loss(args.loss_type)
    criterion.to(device)
    L1_loss = torch.nn.L1Loss()
    L1_loss.to(device)
################################加载优化器######################################
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum, 
                                weight_decay=args.weight_decay, nesterov = True)
###############################################################################
    scheduler=StepLR(optimizer,step_size=args.lr_decay_epoch,gamma=args.lr_decay_rate) 
    print('=>5: 优化器的参数:' ,optimizer)
###############################################################################
###############################################################################
###############################是否使用预训练模型################################
    train_report, val_record, summary_dir, ckpt_dir, time_dir = ckpt_load(args, net, optimizer, device)
##############################开始训练##########################################
    print('==================>开始训练<==================')
    eval_tool = Eval_tool()
    for epoch in range(int(args.start_epoch), args.epochs):
        scheduler.step(epoch)
        #####################训练##########################
        print('Epoch:',epoch)
        begin_time = time.time()
        train_loss = train_fun(net, train_data_loader, device, optimizer, criterion)
        totall_time =  time.time()-begin_time
        print('run time', totall_time, 'train_loss:', train_loss)
        train_report.append(train_loss)
        #####################测试##########################  
        ##################################################'
        ##################################################
        if epoch % args.val_freq ==0:
            begin_time = time.time()
            eval_tool.reset()
            print('<-------val at Epoch:', epoch, '-------------->')
            val_loss, mae, maxF, Sm = val_fun(net, val_data_loader, device, L1_loss, eval_tool)
            totall_time =  time.time()-begin_time
            print('<-------run time:', totall_time,'---------->')
            val_record.append([val_loss, mae, maxF, Sm])
            np.save(summary_dir + '/val_record', val_record)  
            ##################################################
            ##################################################
            ##################################################
        np.save(summary_dir + '/train_report', train_report) 
        if tool.is_cpkt(epoch, args):
              tool.save_ckpt(ckpt_dir, net, optimizer, 1, epoch)
              




def get_trianer1(args, model_tpye = 0):      
  if model_tpye == 1:
    train(args, train_iter, val_iter)
  if model_tpye == 2:
    train(args, train_iter, val_iter, datsets=2) 
  if model_tpye == 3:
    train(args, train_iter, val_iter, datsets=3)  
  if model_tpye == 4:
    train(args, train_iter, val_iter, datsets=4)  
  if model_tpye == 5:
    train(args, train_iter3, val_iter, datsets=1)  
  if model_tpye == 6:
    train(args, train_iter3, val_iter, datsets=2)  
  if model_tpye == 7:
    train(args, train_iter3, val_iter, datsets=3)     
  if model_tpye == 8:
    train(args, train_iter3, val_iter, datsets=4)     
if __name__ == '__main__':
    train()
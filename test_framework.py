import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time

import scipy 
import scipy.misc


BATCHSIZE_PER_CARD = 1

class TTAFramework():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def multi_scaled_imgs(self,imgs, flip_size,rgb_gray,ch,cw,scales): 
        #INPUT 
        #    imgs = [8,3,1024,1024]
        #return 
        #    imgs = [3,8,3,sh,sw]
        #     
        len_scales = len(scales) 

        ms_mask = np.zeros((flip_size, ch,cw))
        for scale in range(0,len_scales):
            scaling = nn.Upsample(scale_factor=scales[scale], mode='nearest') 
            scaled_imgs = scaling(imgs) #8, 3, 1024,1024 
            out = self.net.forward(scaled_imgs).squeeze().cpu().data.numpy() 
            #downsampling 
            scaled_size = [ch,cw] 

            for fs in range(flip_size): 
                scaled_mask = scipy.misc.imresize(out[fs],scaled_size,interp='bilinear', mode=None) 
                scaled_mask = np.divide(scaled_mask, 255) 
                ms_mask[fs] = ms_mask[fs] + scaled_mask 

        return ms_mask

    def multi_scale_logits(self,imgs, flip_size=8,rgb_gray=3, ch=1024,cw=1024,scales=[0.75,1.0,1.25]):
        #imgs : flip_size * rgb_gray * ch * cw = 8*3*1024*1024 
        ms_mask = self.multi_scaled_imgs(imgs, flip_size,rgb_gray, ch,cw,scales) #[scales,flipsize,3,sh,sw]|3,8,3,sh,sw  
        return ms_mask 

    def test_one_img_from_path(self, path, scales=[1.0], evalmode = True):
        if evalmode:
            self.net.eval()
        if len(scales) > 1: 
            BATCHSIZE_PER_CARD = 1 
        else: 
            BATCHSIZE_PER_CARD = 4
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path,scales)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path,scales)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path,scales)

    def test_one_img_from_path_4(self, path,scales):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        flip_size=2;rgb_gray=3;ch=1024; cw=1024;
        maska = self.multi_scale_logits(img1,flip_size,rgb_gray,ch,cw,scales) #mask : 2,3,1024,1024 -> 2,1024,1024
        maskb = self.multi_scale_logits(img2,flip_size,rgb_gray,ch,cw,scales) 
        maskc = self.multi_scale_logits(img3,flip_size,rgb_gray,ch,cw,scales) 
        maskd = self.multi_scale_logits(img4,flip_size,rgb_gray,ch,cw,scales) 
        
        #maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        #maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        #maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        #maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, path,scales):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        
        img6 = V(torch.Tensor(img6).cuda())
        
        flip_size=4;rgb_gray=3;ch=1024; cw=1024;
        maska = self.multi_scale_logits(img5,flip_size,rgb_gray,ch,cw,scales) #mask : 4,3,1024,1024 -> 4,1024,1024
        maskb = self.multi_scale_logits(img6,flip_size,rgb_gray,ch,cw,scales) #mask : 4,3,1024,1024 -> 4,1024,1024 
        #maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        #maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path,scales):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1] # geometric ensemble 
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1] # geometric ensemble 
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        #img5(8,3,1024,1024) -> Multi-Scaled(MS,8,3,1024,1024) -> solver(MS,8,1024,1024) -> RevScaled(8,1024,1024) 
        #scales = [0.8125,0.875,0.9375, 1, 1.0625,1.125,1.1875]        
        flip_size=8;rgb_gray=3;ch=1024; cw=1024;
        mask = self.multi_scale_logits(img5,flip_size,rgb_gray,ch,cw,scales) #mask : 8,1024,1024 
        
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def load(self, path):
        
        p_dict = torch.load(path) 
        model_dict = self.net.state_dict() 
        
        p_dict = {k: v for k, v in p_dict.items() if k in model_dict}
        model_dict.update(p_dict) 
        self.net.load_state_dict(model_dict) 
        
        #self.net.load_state_dict(torch.load(path))
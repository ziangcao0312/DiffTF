# Standard NeRF Blender dataset loader
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
import os
import imageio.v2 as imageio
from tqdm import tqdm
import cv2
import json
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
from os import path
import ipdb
import re

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
    

class OurtestddDDataset(Dataset):
    """
    NeRF dataset loader
    """

    # focal: float
    # c2w: torch.Tensor  # (n_images, 4, 4)
    # gt: torch.Tensor  # (n_images, h, w, 3)
    # h: int
    # w: int
    # n_images: int
    # split: str

    def __init__(
        self,
        config,
        root,
    ):
        super(OurtestddDDataset,self).__init__()
 

        self.base_path=root
        self.config=config

        class_names=os.listdir(root)
        class_names.sort()
        class_num=len(class_names)

        allc2w=[]
        allimgpath=[]
        alldpath=[]
        allk=[]
        label=[]
        label_f=0
        class_label=0
        allclasslabel=[]
        cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #class_names=class_names[:1]
        
            #ipdb.set_trace()
        instance_num=len(os.listdir(root))
        #instance_num=1  #4gpus 6batch

        instance_num=1

        instance_name=os.listdir(root)
        #instance_name.sort(key=lambda l: int(re.findall('\d+', l)[0]))  

        instance_name=instance_name[:instance_num]
        self.H,self.W=512,512
        oric2w=[]
        focal = 525
        #ipdb.set_trace()
        for each_instance in instance_name:
            path=os.path.join(root,each_instance)
            img_path = os.path.join(path, 'rgb')
            pos_path = os.path.join(path, 'pose')

            num=len(os.listdir(pos_path))
            pos_name=os.listdir(pos_path)
            pos_name.sort(key=lambda l: int(re.findall('\d+', l)[0]))  

            img_name=[f for f in os.listdir(img_path) if f.endswith('.png')]
            #ipdb.set_trace()
            img_name.sort(key=lambda l: int(re.findall('\d+', l)[0]))  
            #img_name=img_name[:int(num*args.train_test)]  
            #ipdb.set_trace()
            #ipdb.set_trace()

            
            #oric2w=[]
            if config.testvideo==1:
                for i in range(round(num*(config.train_test))):
                    i=i*int(1/config.train_test)
                    c2w=np.loadtxt(os.path.join(pos_path,pos_name[i])).reshape(4,4)
                    #c2w[:,1:3]*=-1
                    c2w[:3, 3] *=1.8

                    allc2w.append(c2w)
                    allimgpath.append(os.path.join(img_path,img_name[i]))
                    
                    allk.append(np.array([[focal,0,self.W * 0.5],\
                                            [0,focal,self.H * 0.5],\
                                                [0,0,1]]))
                    label.append(label_f)
            else:
                for i in range(10):
                    i=i*int(10)+100
                    c2w=np.loadtxt(os.path.join(pos_path,pos_name[i])).reshape(4,4)
                    #c2w[:,1:3]*=-1
                    oric2w.append(c2w)
                    c2w[:3, 3] *=1.8
                    #ipdb.set_trace()

                    allc2w.append(c2w)
                    allimgpath.append(os.path.join(img_path,img_name[i]))
                    
                    allk.append(np.array([[focal,0,self.W * 0.5],\
                                            [0,focal,self.H * 0.5],\
                                                [0,0,1]]))
                    label.append(label_f)
            
            allclasslabel.append(class_label)
            label_f+=1
        #ipdb.set_trace()
        self.allc2w=np.stack(allc2w)
        self.allimgpath=allimgpath
        config.num_instance=label_f

        self.allk=np.stack(allk)
        self.label=np.stack(label)
        

                


        #ipdb.set_trace()  
        self.num=len(allimgpath)



    def __len__(self):
        return self.num

        
    def __getitem__(self, index) :
        target=torch.Tensor(imageio.imread(self.allimgpath[index]))/255.

        if self.config.white_bkgd and target.shape[-1]==4:
            target = target[...,:3]*target[...,-1:] + (1.-target[...,-1:])
        else:
            target = target[...,:3]
        
        pose=torch.Tensor(self.allc2w[index,:3,:4])
        K=torch.Tensor(self.allk[index])
        H=self.H
        W=self.W
        label=torch.Tensor([self.label[index]]).long()
        near=torch.Tensor([0.2])#np.array(int(depth.min()))-1
        far=near+2
        
        #ipdb.set_trace()  
        
        rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)

        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1,2]).long() # (N_rand, 2)
        #ipdb.set_trace()
        rays_o = rays_o[coords[:, 0], coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[coords[:, 0], coords[:, 1]]  # (N_rand, 3)
        #ipdb.set_trace()
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[coords[:, 0], coords[:, 1]] # (N_rand, 3)imageio.imwrite('a.png',np.array(target[select_coords[:, 1], select_coords[:, 0]]).reshape(200,200,4))

        #ipdb.set_trace()

        depth_s=None
        if self.config.using_depth:
            depth=torch.Tensor(imageio.imread(self.alldpath[index]))
            depth_s= depth[select_coords[:, 0], select_coords[:, 1]]
            
            #ipdb.set_trace()


            return {
                'batch_rays': batch_rays,
                'label': label,
                'target_s': target_s,
                'depth_s': depth_s,
                'near': near,
                'far': far,
                'hwk': [H,W,K]
            }
        else:
                return {
                'batch_rays': batch_rays,
                'label': label,
                'target_s': target_s,
                'near': near,
                'far': far,
                'hwk': [H,W,K]
            }


if __name__=='__main__':
    from parser_config import config_parser

    parser = config_parser()
    newargs = parser.parse_args()
    path='/nvme/caoziang/3D_generation/data/test'
    dataset = OurDDataset(newargs,path)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,  pin_memory=True)
    
    for i in range(5):

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for index, data in tqdmDataLoader:
                ipdb.set_trace()  
                print(i)
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


class OurDDataset(Dataset):
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
        super(OurDDataset,self).__init__()
 

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
        
        instance_num=len(os.listdir(root))
        

        instance_name=os.listdir(root)
        #we use 1500 objects to train the shared decoder.
        instance_name=np.array(instance_name)[np.random.choice(len(instance_name),size=[1500],replace=False)]
        
        self.H,self.W=512,512
        
        focal = 525
        for each_instance in instance_name:
            if os.path.isdir(os.path.join(root,each_instance)):
                path=os.path.join(root,each_instance)
                img_path = os.path.join(path, 'rgb')
                pos_path = os.path.join(path, 'pose')

                num=len(os.listdir(pos_path))
                pos_name=os.listdir(pos_path)
                pos_name.sort(key=lambda l: int(re.findall('\d+', l)[0]))  

                img_name=[f for f in os.listdir(img_path) if f.endswith('.png')]
                img_name.sort(key=lambda l: int(re.findall('\d+', l)[0]))  

                for i in range(round(num*config.train_test)):
                    c2w=np.loadtxt(os.path.join(pos_path,pos_name[i])).reshape(4,4)
                    c2w[:3, 3] *=1.8

                    allc2w.append(c2w)
                    allimgpath.append(os.path.join(img_path,img_name[i]))
                    
                    allk.append(np.array([[focal,0,self.W * 0.5],\
                                            [0,focal,self.H * 0.5],\
                                                [0,0,1]]))
                    label.append(label_f)
                
                allclasslabel.append(class_label)
                label_f+=1
                print(label_f)
        self.allc2w=np.stack(allc2w)
        self.allimgpath=allimgpath
        config.num_instance=label_f

        self.allk=np.stack(allk)
        self.label=np.stack(label)
        self.config=config

    

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
        near=torch.Tensor([0.2])
        far=near+2
        

        
        rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)


        if target[:10,:10,:].sum()<1e-4:
            mask=(target.mean(-1)>1e-4)
        else:
            mask=(target.mean(-1)<0.9999)
        if np.random.random() > self.config.pos_rate:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)[mask]  
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

       
        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        if coords.shape[0]>=self.config.N_rand:
            select_inds = np.random.choice(coords.shape[0], size=[self.config.N_rand], replace=False)  # (N_rand,)
        else:
            select_inds = np.random.choice(coords.shape[0], size=[self.config.N_rand], replace=True)

        select_coords = coords[select_inds].long() # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]] # (N_rand, 3)


        depth_s=None
        if self.config.using_depth:
            depth=torch.Tensor(imageio.imread(self.alldpath[index]))
            depth_s= depth[select_coords[:, 0], select_coords[:, 1]]
            

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

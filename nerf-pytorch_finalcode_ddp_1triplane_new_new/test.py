import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from configparse import config_parser
import matplotlib.pyplot as plt
import ipdb
from run_nerf_helpers import *
from torch.utils.data import DataLoader
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn

parser = config_parser()
args = parser.parse_args()

np.random.seed(0)



def test(args):

    with torch.no_grad():
        if args.dataset=='omni':
            from omni_dataset import OurDDataset,OurtestDDataset
        elif args.dataset=='shapenet':
            from shapenet_dataset import OurDDataset,OurtestDDataset
        else:
            print('error dataset')
    
        testdata=OurtestDDataset(args,args.datadir)
        K=None
        
        
        test_loader = DataLoader(testdata, batch_size=1, shuffle=False,pin_memory=False, generator=torch.Generator(device='cuda'))
        

        savebasedir = args.basedir
        expname = args.expname

        testsavedir = os.path.join(savebasedir, expname, 'test')
        os.makedirs(testsavedir, exist_ok=True)


        # Create nerf model
        _, render_kwargs_test, start, optimizer,warmUpScheduler = create_nerf(args)
        
        render_kwargs_test['network_fn'].eval().cuda()


        rgbss=[]
        


        for i, data in enumerate(test_loader):
            batch_rays=data['batch_rays'].cuda()
            label=data['label'].cuda().squeeze(-1)
            target_s=data['target_s'].cuda()
            if args.using_depth:
                depth_s=1/data['depth_s'][...,0].cuda()
            near=data['near'].cuda()
            far=data['far'].cuda()
            #ipdb.set_trace()


            #rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)

            #ipdb.set_trace()
            rgbs, _,_ = render_path1(batch_rays, args.chunk, render_kwargs_test, gt_imgs=target_s, label=label,savedir=os.path.join(testsavedir,str(i)+'.png'),savedir1=os.path.join(testsavedir,str(i)+'_gt.png'),savedir2=os.path.join(testsavedir,str(i)+'_acc.png'),savedir3=os.path.join(testsavedir,str(i)+'_accgt.png'),savedir4=os.path.join(testsavedir,str(i)+'triplane.png'),near=near,far=far)
            
            print(i)
            rgbss.append(rgbs)

        if args.testvideo:
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), np.stack(rgbss), fps=30, quality=8)

    

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    test(args)


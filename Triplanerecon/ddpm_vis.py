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

parser = config_parser()
args = parser.parse_args()

def testddpm(args):
    if args.dataset=='omni':
        from omni_dataset_ddpm import OurtestddDDataset
    elif args.dataset=='shapenet':
        from shapenet_dataset_ddpm import OurtestddDDataset
    useidx=[]
    with torch.no_grad():
        triplanes=np.load(args.triplanepath)['triplane']

        if len(useidx)==0:
            namelist=np.arange(len(triplanes))
            gpunum=args.num_gpu
            gpuidx=args.idx
            if len(namelist)//gpunum*(gpuidx+1)<=len(namelist): 
                namelist=namelist[len(namelist)//gpunum*gpuidx:len(namelist)//gpunum*(gpuidx+1)]
            elif len(namelist)//gpunum*(gpuidx+1)>len(namelist) and len(namelist)//gpunum*(gpuidx)<=len(namelist):
                namelist=namelist[len(namelist)//gpunum*gpuidx:]
            else:
                return
        else:
            namelist=useidx
        for num in namelist:
            
            idxx=num
            triplane=torch.Tensor(triplanes).cuda()[idxx:idxx+1]
        
            testdata=OurtestddDDataset(args,args.datadir)
            K=None
            
            
            test_loader = DataLoader(testdata, batch_size=1, shuffle=False,pin_memory=False, generator=torch.Generator(device='cuda'))
            

            savebasedir = args.basedir
            expname = args.expname

            testsavedir = os.path.join(savebasedir, expname, 'test',str(idxx))
            os.makedirs(testsavedir, exist_ok=True)

            _, render_kwargs_test, start, optimizer,warmUpScheduler = create_nerf(args)
            gt_triplane=render_kwargs_test['network_fn'].tri_planes.clone()
            render_kwargs_test['network_fn'].eval().cuda()

            triplane=triplane.view(gt_triplane.shape)


            rgbss=[]
            render_kwargs_test['network_fn'].tri_planes.copy_(triplane)
            if args.mesh:
                if not os.path.exists(os.path.join(testsavedir+'.ply')) or not os.path.exists(os.path.join(testsavedir+'.obj')):
                    generate_mesh(args,os.path.join(testsavedir+'.ply'),render_kwargs_test['network_query_fn'],render_kwargs_test['network_fn'])

                    continue
                else:
                    continue

            if args.testvideo and os.path.exists(os.path.join(testsavedir+ 'video.mp4')):
                print('skip',testsavedir)
                continue

            for i, data in enumerate(test_loader):
                batch_rays=data['batch_rays'].cuda()
                label=data['label'].cuda().squeeze(-1)
                if args.using_depth:
                    depth_s=1/data['depth_s'][...,0].cuda()
                near=data['near'].cuda()
                far=data['far'].cuda()
                
                render_kwargs_test['mode']='test'
                rgbs, _,_ = render_path2(batch_rays, args.chunk, render_kwargs_test, label=label,savedir=os.path.join(testsavedir,str(i)+'be.png'),savedir1=os.path.join(testsavedir,str(i)+'ddpm.png'),savedir2=os.path.join(testsavedir,str(i)+'ddpm_acc.png'),savedir3=os.path.join(testsavedir,str(i)+'triplane.png'),near=near,far=far)
                print(i)
                rgbss.append(rgbs)
                if args.testvideo!=1 and i>9:
                    break

            if args.testvideo:
                imageio.mimwrite(os.path.join(testsavedir+ 'video.mp4'), np.stack(rgbss), fps=30, quality=8)
            


    

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    testddpm(args)

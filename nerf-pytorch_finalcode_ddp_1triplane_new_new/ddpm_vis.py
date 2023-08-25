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

    with torch.no_grad():
        #ipdb.set_trace()
        triplanes=np.load(args.triplanepath)['triplane']#np.load(args.triplanepath)['triplane']#.squeeze()/4#torch.load(args.triplanepath)#np.load(args.triplanepath)['triplane']
        #ipdb.set_trace()
        #triplanes=triplanes[::10]
        #triplanes=np.random.randn(1,18,256,256)
        for num in range(len(triplanes)):
            # if num<=3000:
            #     continue
            idxx=num
            # if os.path.exists(os.path.join(args.basedir, args.expname, 'test',str(idxx)+'video.mp4')):
            #     print(idxx)
            #     continue

            
            #ipdb.set_trace()
            #labelss=torch.Tensor(triplane['label']).cuda()
            triplane=torch.Tensor(triplanes).cuda()[idxx:idxx+1].repeat(1,1,1,1)
            #ipdb.set_trace()
        
            testdata=OurtestddDDataset(args,args.datadir)
            K=None
            
            
            test_loader = DataLoader(testdata, batch_size=1, shuffle=False,pin_memory=False, generator=torch.Generator(device='cuda'))
            

            savebasedir = args.basedir
            expname = args.expname

            testsavedir = os.path.join(savebasedir, expname, 'test',str(idxx))
            os.makedirs(testsavedir, exist_ok=True)


            # Create nerf model
            _, render_kwargs_test, start, optimizer,warmUpScheduler = create_nerf(args)
            gt_triplane=render_kwargs_test['network_fn'].tri_planes.clone()
            render_kwargs_test['network_fn'].eval().cuda()

            triplane=triplane.view(gt_triplane.shape)


            rgbss=[]
            render_kwargs_test['network_fn'].tri_planes.copy_(triplane)
            if args.mesh:
                #ipdb.set_trace()
                if not os.path.exists(os.path.join(testsavedir+'.ply')) or not os.path.exists(os.path.join(testsavedir+'.obj')):
                    #try:
                    generate_mesh(args,os.path.join(testsavedir+'.ply'),render_kwargs_test['network_query_fn'],render_kwargs_test['network_fn'])
                    #except:
                        #print(testsavedir)
                    continue
                else:
                    continue

            if args.testvideo and os.path.exists(os.path.join(testsavedir+ 'video.mp4')):
                print('skip',testsavedir)
                continue

            for i, data in enumerate(test_loader):
                batch_rays=data['batch_rays'].cuda()
                label=data['label'].cuda().squeeze(-1)
                #target_s=data['target_s'].cuda()
                if args.using_depth:
                    depth_s=1/data['depth_s'][...,0].cuda()
                near=data['near'].cuda()
                far=data['far'].cuda()
                #ipdb.set_trace()


                #rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)

                
                # render_kwargs_test['network_fn'].tri_planes.copy_(gt_triplane)
                # rgbs, _,_ = render_path1(batch_rays, args.chunk, render_kwargs_test, gt_imgs=target_s, label=label,savedir=os.path.join(testsavedir,str(i)+'.png'),savedir1=os.path.join(testsavedir,str(i)+'_gt.png'),near=near,far=far)
                
                # new=gt_triplane.clone()
                # new[torch.where(new>4)]=4
                # render_kwargs_test['network_fn'].tri_planes.copy_(new)
                # rgbs, _,_ = render_path1(batch_rays, args.chunk, render_kwargs_test, gt_imgs=target_s, label=label,savedir=os.path.join(testsavedir,str(i)+'sss.png'),savedir1=os.path.join(testsavedir,str(i)+'_gt.png'),near=near,far=far)
                
                # noise=torch.randn(gt_triplane.shape)
                # render_kwargs_test['network_fn'].tri_planes.copy_(gt_triplane+noise/noise.max()/2)
                # rgbs, _,_ = render_path1(batch_rays, args.chunk, render_kwargs_test, gt_imgs=target_s, label=label,savedir=os.path.join(testsavedir,str(i)+'noise.png'),savedir1=os.path.join(testsavedir,str(i)+'_gt.png'),near=near,far=far)
                #ipdb.set_trace()
                
                rgbs, _,_ = render_path2(batch_rays, args.chunk, render_kwargs_test, label=label,savedir=os.path.join(testsavedir,str(i)+'be.png'),savedir1=os.path.join(testsavedir,str(i)+'ddpm.png'),savedir2=os.path.join(testsavedir,str(i)+'ddpm_acc.png'),savedir3=os.path.join(testsavedir,str(i)+'triplane.png'),near=near,far=far)
                print(i)
                #rgbss.append(np.uint8(rgbs.cpu().numpy()[0]*255))
                rgbss.append(rgbs)
                #ipdb.set_trace()
                if args.testvideo!=1 and i>9:
                    break

            if args.testvideo:
                imageio.mimwrite(os.path.join(testsavedir+ 'video.mp4'), np.stack(rgbss), fps=30, quality=8)
            


    

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    testddpm(args)

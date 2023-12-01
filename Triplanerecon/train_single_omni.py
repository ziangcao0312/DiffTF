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
from omni_dataset_single import OurDDataseteach,OurtestDDataseteach
from torch.utils.data import DataLoader
import logging
import re



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
parser = config_parser()
args = parser.parse_args()




def train_single(args):

    class_names=os.listdir(args.datadir)
    class_names.sort()
    class_num=len(class_names)

    allpath=[]
    allk=[]
    for class_name in class_names:
        if not os.path.isdir(os.path.join(args.datadir,class_name)):
                continue
        all_instances=os.path.join(args.datadir,class_name)
        instance_name=[]
        for name in os.listdir(all_instances):
            if os.path.isdir(os.path.join(args.datadir,class_name,name)):
                instance_name.append(name)
        instance_name.sort(key=lambda l: int(re.findall('\d+', l)[0]))  

        for each_instance in instance_name:
            aapath=os.path.join(all_instances,each_instance)
            allpath.append(aapath)
            

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())    


    
        

    
    logger = get_logger(os.path.join(basedir,expname,str(args.idx)+'exp.log'))

    logger.info('start training!')
    n=args.idx
    eachrange=len(allpath)//args.num_gpu+1
    if eachrange*(n+1)<=len(allpath): 
        allpath=allpath[eachrange*n:eachrange*(n+1)]
    elif eachrange*(n+1)>len(allpath) and eachrange*n<=len(allpath):
        allpath=allpath[eachrange*n:]
    else:
        return

    for index in range(len(allpath)):
        datadir=allpath[index]
        try:
            render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
        
            
            for modules in [render_kwargs_train['network_fn'].pts_linears,\
                            render_kwargs_train['network_fn'].views_linears,\
                                render_kwargs_train['network_fn'].feature_linear,\
                                    render_kwargs_train['network_fn'].alpha_linear,\
                                        render_kwargs_train['network_fn'].rgb_linear,\
                                ]:
                            for param in modules.parameters():
                                param.requires_grad = False


            if os. path.exists(os.path.join(basedir, expname,datadir.split('/')[-1],str(args.N_iters-1).zfill(6)+'.tar')):
                logger.info('Saved checkpoints at'+ os.path.join(basedir, expname,datadir.split('/')[-1],str(args.N_iters-1).zfill(6)+'.tar'))
                continue
            traindata=OurDDataseteach(args,datadir)
            
            training_loader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker, generator=torch.Generator(device='cuda'))
            iter_per_epoch = len(training_loader)
            global_step = 0
            os.makedirs(os.path.join(basedir, expname,datadir.split('/')[-1]), exist_ok=True)
            


            while global_step < args.N_iters: 
                epoch = global_step // iter_per_epoch
                
                time0 = time.time()


                for i, data in enumerate(training_loader):
                    batch_rays=data['batch_rays'].cuda()
                    label=data['label'].cuda().squeeze(-1)
                    target_s=data['target_s'].cuda()
                    if args.using_depth:
                        depth_s=1/data['depth_s'][...,0].cuda()
                    near=data['near'].cuda()
                    far=data['far'].cuda()
                    

                    rgb, disp, acc, extras = render(chunk=args.chunk, rays=batch_rays,near=near,far=far,label=label,
                                                        verbose=i < 10, retraw=True,
                                                        **render_kwargs_train)

                    optimizer.zero_grad()
                    mask_img=torch.zeros_like(acc).to(device=acc.device)
                    if args.white_bkgd:
                        mask=(target_s.mean(-1)<0.9999)
                    else:
                        mask=(target_s.mean(-1)>1e-4)
                    mask_img[mask]=1
                    loss = img2mse(rgb,target_s)
                    trans = extras['raw'][...,-1]
                    
                    triplane=render_kwargs_train['network_fn'].tri_planes[label]
                    norm=torch.abs(triplane).max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    triplane=triplane/norm
                
                    tvloss_y = torch.abs(triplane[:, :,:-1] - triplane[:, :, 1:]).mean()
                    tvloss_x = torch.abs(triplane[:, :, :,:-1] - triplane[:, :, :, 1:]).mean()
                    tvloss = tvloss_y + tvloss_x
                    loss += tvloss * 5e-1
                
                    l1_loss = (triplane ** 2).mean()
                    loss += l1_loss * 1e-1
                    psnr = mse2psnr((img2mse(rgb[mask], target_s[mask])))
                    psnr_all = mse2psnr((img2mse(rgb, target_s)))
                    
                    if 'rgb0' in extras:
                        img_loss0 = img2mse(extras['rgb0'],target_s)
                        loss = loss + img_loss0

                    loss.backward()
                    optimizer.step()

                

                    # NOTE: IMPORTANT!
                    ###   update learning rate   ###
                    decay_rate = 0.1
                    decay_steps = args.lrate_decay * 1000
                    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                    for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lrate
                        
                    ################################

                    dt = time.time()-time0
                    

                    # Rest is logging
                    if global_step%(args.i_weights)==0 and global_step!=0:
                        path = os.path.join(basedir, expname,datadir.split('/')[-1], '{:06d}.tar'.format(global_step))
                        torch.save({
                            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        }, path)
                        logger.info('Saved checkpoints at'+ path)

                    if global_step%args.i_print==0:
                        logger.info('Epoch:[{}/{}]\t loss={:f}\t psnr={:f}\t psnr_all={:f}\t name={}\t lr={:.5f}'.format(global_step , args.N_iters, float(loss.item()), float(psnr.item()) ,float(psnr_all.item()),datadir.split('/')[-1],float(new_lrate)))
                    global_step += 1
        except:
            logger.info('error',datadir.split('/')[-1])

def test_single(args):

    with torch.no_grad():
        class_names=os.listdir(args.datadir)
        class_names.sort()
        class_num=len(class_names)

        allpath=[]
        allk=[]
        for class_name in class_names:
            if not os.path.isdir(os.path.join(args.datadir,class_name)):
                continue
            all_instances=os.path.join(args.datadir,class_name)
            instance_name=[]
            for name in os.listdir(all_instances):
                if os.path.isdir(os.path.join(args.datadir,class_name,name)):
                    instance_name.append(name)
            instance_name.sort(key=lambda l: int(re.findall('\d+', l)[0]))  

            for each_instance in instance_name:
                aapath=os.path.join(all_instances,each_instance)
                allpath.append(aapath)
        n=args.idx
        eachrange=len(allpath)//args.num_gpu+1
        if eachrange*(n+1)<=len(allpath): 
            allpath=allpath[eachrange*n:eachrange*(n+1)]
        elif eachrange*(n+1)>len(allpath) and eachrange*n<=len(allpath):
            allpath=allpath[eachrange*n:]
        else:
            return

        for index in range(len(allpath)):

            datadir=allpath[index]
    
            testdata=OurtestDDataseteach(args,datadir)
            K=None
            
            
            test_loader = DataLoader(testdata, batch_size=1, shuffle=False,pin_memory=False, generator=torch.Generator(device='cuda'))
            

            
            args.num_instance=1

            testsavedir = os.path.join(args.basedir, args.expname,datadir.split('/')[-1], 'test')
            os.makedirs(testsavedir, exist_ok=True)

           
            _, render_kwargs_test, start, optimizer,warmUpScheduler = create_nerf(args,datadir.split('/')[-1])
            
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
                
                rgbs, _,_ = render_path1(batch_rays, args.chunk, render_kwargs_test, gt_imgs=target_s, label=label,savedir=os.path.join(testsavedir,str(i)+'.png'),savedir1=os.path.join(testsavedir,str(i)+'_gt.png'),savedir2=os.path.join(testsavedir,str(i)+'_acc.png'),savedir3=os.path.join(testsavedir,str(i)+'_accgt.png'),savedir4=os.path.join(testsavedir,str(i)+'triplane.png'),near=near,far=far)
                
                print(i)
                rgbss.append(rgbs)

            if args.testvideo:
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), np.stack(rgbss), fps=30, quality=8)





if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if args.state=='train_single':
        train_single(args)
    elif args.state=='test_single':
        test_single(args)
    else:
        print('error')

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
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
dist.init_process_group(backend='nccl',rank=WORLD_RANK, world_size=WORLD_SIZE)
np.random.seed(0)




def train(args):
    if args.dataset=='omni':
        from omni_dataset import OurDDataset,OurtestDDataset
    elif args.dataset=='shapenet':
        from shapenet_dataset import OurDDataset,OurtestDDataset
    elif args.dataset=='shapenet_alldecoder':
        from shapenet_dataset_all import OurDDataset,OurtestDDataset
    else:
        print('error dataset')
    traindata=OurDDataset(args,args.datadir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(traindata)

    training_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size,num_workers=args.num_worker, sampler=train_sampler,generator=torch.Generator(device='cuda'))

    # Create log dir and copy the config file
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

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start


    iter_per_epoch = len(training_loader)
    logger = get_logger(os.path.join(basedir,expname,str(dist.get_rank())+'exp.log'))

    logger.info('start training!')


    while global_step < args.N_iters: 
        epoch = global_step // iter_per_epoch
        training_loader.sampler.set_epoch(epoch)


        time0 = time.time()

        #ipdb.set_trace()
        for data in training_loader:
        #     a=data
        #     print(a)
            #ipdb.set_trace()
            batch_rays=data['batch_rays'].cuda()
            label=data['label'].cuda().squeeze(-1)
            target_s=data['target_s'].cuda()
            if args.using_depth:
                depth_s=1/data['depth_s'][...,0].cuda()
            near=data['near'].cuda()
            far=data['far'].cuda()
            #ipdb.set_trace()

            #ipdb.set_trace()

            rgb, disp, acc, extras = render(chunk=args.chunk, rays=batch_rays,near=near,far=far,label=label, retraw=True,
                                                **render_kwargs_train)
            
            optimizer.zero_grad()
            mask_img=torch.zeros_like(acc).to(device=acc.device)
            if args.white_bkgd:
                mask=(target_s.mean(-1)<0.9999)
            else:
                mask=(target_s.mean(-1)>1e-4)
            mask_img[mask]=1
            #ipdb.set_trace()
            #target_s=target_s*2-1
            img_loss = img2mse(rgb,target_s)+img2mse(acc,mask_img)#(img2mse(rgb[mask], target_s[mask])+img2mse(rgb[~mask], target_s[~mask]))
            trans = extras['raw'][...,-1]
            # def charloss(rgb,target_s):
            #         diff = torch.add(rgb, -target_s)
            #         error = torch.sqrt(diff * diff + 1e-6)
            #         return torch.mean(error)
            #ipdb.set_trace()
            
            loss = img_loss#+(torch.sqrt((render_kwargs_train['network_fn'].module.tri_planes[label])**2).sum())*1e-8#+loss1#+loss1#+tvloss(render_kwargs_train['network_fn'].tri_planes)*1e-3
            
            triplane=render_kwargs_train['network_fn'].module.tri_planes[label]
            
            tvloss_y = torch.abs(triplane[:, :,:-1] - triplane[:, :, 1:]).mean()
            tvloss_x = torch.abs(triplane[:, :, :,:-1] - triplane[:, :, :, 1:]).mean()
            tvloss = tvloss_y + tvloss_x
            loss += tvloss * 1e-4
            #loss_dict['train/tvloss'] = tvloss * 5e-4
        
            l1_loss = (triplane ** 2).mean()
            loss += l1_loss * 5e-5
            #loss_dict['train/l1loss'] = l1_loss * 2e-4
            #ipdb.set_trace()
            if torch.isnan(loss):
                ipdb.set_trace()
            psnr = mse2psnr(img2mse(rgb,target_s))
            #print(render_kwargs_train['network_fn'].pts_linears[0].weight[0])
            #print(render_kwargs_train['network_fn'].tri_planes[0])

            if 'rgb0' in extras:
                img_loss0 = (img2mse(extras['rgb0'],target_s)+img2mse(extras['acc0'],mask_img))#img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                #psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()

            #new=render_kwargs_train['network_fn'].tri_planes.clone()
            #print(((old-new)**2).sum())

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))

            threshold=0
            new_lrate2 = args.lrate * (decay_rate ** ((global_step+threshold) / decay_steps))
            #ipdb.set_trace()
            for param_group in optimizer.param_groups:
                #ipdb.set_trace()
                if len(param_group['params'][0])==args.num_instance and param_group['params'][0].shape[1]==args.triplanechannel:
                    param_group['lr'] = new_lrate*10
                    
                else:
                    if global_step>=threshold:
                        param_group['lr'] = new_lrate2
                    else:
                        param_group['lr'] =0
            ################################

            dt = time.time()-time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####

            # Rest is logging
            if global_step%args.i_weights==0 and dist.get_rank() == 0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(global_step))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

            if global_step%args.i_print==0:
                logger.info('Epoch:[{}/{}]\t loss={:f}\t psnr={:f}'.format(global_step , args.N_iters, float(loss.item()), float(psnr.item()) ))
            # if global_step%args.i_print==0:
                tqdm.write(f"[TRAIN] Iter: {global_step} Loss: {loss.item()}  PSNR: {psnr.item()}")

            global_step += 1

    

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train(args)


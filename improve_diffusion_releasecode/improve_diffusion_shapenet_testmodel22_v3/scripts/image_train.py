"""
Train a diffusion model on images.
"""

import sys, os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current + "/../")

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.RenderPeople_datasets import load_triplane_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    if dist.get_rank() == 0:
        writer = None#SummaryWriter(os.path.join(args.log_dir, "runs"))
    else:
        writer = None

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    #model.load_state_dict(torch.load('/nvme/caoziang/3D_generation/Checkpoint_all/ddpm_pos_uncondition431_ori_5e_14_newmodelv2/ema_0.9999_010000.pt'))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    data = load_triplane_data(
        data_dir=args.datasetdir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        writer=writer,
    ).run_loop()
    if dist.get_rank() == 0:
        writer.close()


def create_argparser():
    datasetdir='/mnt/petrelfs/caoziang/3D_generation/Checkpoint_all/shapenet_triplane_new_new_chair/triplane'
    data_dir="/mnt/petrelfs/caoziang/3D_generation/Checkpoint_all/"
    expname='diffusion_shapenet_testmodel22_chair'
    defaults = dict(
        data_name="omni",
        clip_denoised=False,
        data_dir=data_dir,
        datasetdir=datasetdir,
        log_dir=data_dir+expname,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=1e-4,
        lr_anneal_steps=0,
        batch_size=4, #18 channel use batchsize 8
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

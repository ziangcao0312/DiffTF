"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import ipdb
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_testdefaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_testdefaults().keys())
    )
    #ipdb.set_trace()
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    #model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()
    #ipdb.set_trace()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    if args.class_cond:
        allclasses = th.arange(0, 216, device=dist_util.dev())
    ckpt = th.load(args.gt_path)
    tri_plane = ckpt['network_fn_state_dict']['tri_planes']

    noise1=th.randn(1,18,256,256).to(device=dist_util.dev())

    noise2=th.randn(1,18,256,256).to(device=dist_util.dev())
    # for i in range(args.num_samples):
    #     ratio=i/(args.num_samples-1)
    #     print(ratio)
    # ipdb.set_trace()
    
    for i in range(args.num_samples):
        # ratio=i/(args.num_samples-1)
        # noise=th.lerp(noise1, noise2, ratio)
        # noise=noise/noise.std()

        model_kwargs = {}
        if args.class_cond:
           # ipdb.set_trace()
            classes = allclasses[170:171]#[np.random.randint(216,size=args.batch_size)]#.repeat(args.batch_size)
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 18, args.image_size, args.image_size),
            #noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,progress=True
        )
        #ipdb.set_trace()
        #tri_plane1 = tri_plane[:1]
        # try:
        #     weight=th.load(args.weightpath)
        # except:
        #     weight=[5.32]#th.abs(tri_plane1).max()
        #sample = ((sample ) * 9)#th.atanh(sample.clip(-0.999,0.999))*3        #((sample ) * 9)
        #np.savez_compressed(os.path.join(args.save_path,'a.npz'), triplane=sample.cpu().numpy())

        
        
        #ipdb.set_trace() 
        #mse=(th.abs(tri_plane1-sample)).mean()
        #print(mse)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    #arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(args.save_path, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        #ipdb.set_trace()
        if args.class_cond:
            np.savez_compressed(out_path, triplane=arr, label=label_arr)
        else:
            np.savez_compressed(out_path, triplane=arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=626,
        batch_size=1,
        use_ddim=False,
        model_path="/mnt/petrelfs/caoziang/3D_generation/Checkpoint_all/diffusion_shapenet_testmodel27_omni/ema_0.9999_350000.pt",
        gt_path='/mnt/petrelfs/caoziang/3D_generation/Checkpoint_all/shapenet_triplane_new_new/003000.tar',
        save_path='/mnt/petrelfs/caoziang/3D_generation/Checkpoint_all/diffusion_shapenet_testmodel27_omni',
    )
    defaults.update(model_and_diffusion_testdefaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

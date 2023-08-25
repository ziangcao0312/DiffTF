"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys, os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current + "/../")

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import imageio

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    if "ema" in args.model_path:
        suffix = f"ckpt_{args.model_path.split('/')[-1].split('_')[2].split('.')[0]}_ema"
    else:
        suffix = f"ckpt_{args.model_path.split('/')[-1].split('.')[0].split('model')[1]}"

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample#.permute(0, 2, 3, 1)
        # sample = sample.contiguous()

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
    arr = arr[: args.num_samples]

    # reverse triplane
    # x_new = arr.reshape(-1)
    # x_new_min, x_new_max = -0.6611096, 0.6611096 # -0.6611096 0.6611096
    # x_reverse = (x_new + 1) / 2 * (x_new_max - x_new_min) + x_new_min
    # x_reverse_new = np.zeros_like(x_new)
    # for i in range(x_reverse.shape[0]):
    #     if x_reverse[i] > 0:
    #         x_reverse_new[i] = 100 ** (x_reverse[i]) - 1  #math.log(x[i]+1, 100)
    #     elif x_reverse[i] < 0:
    #         x_reverse_new[i] = -100 ** (-x_reverse[i]) + 1 #-math.log(-(x[i]-1), 100)
    # tri_plane_new_reverse = x_reverse_new.reshape(*arr.shape) / 20
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.hist(tri_plane_new_reverse.reshape(-1), bins=100)
    # plt.savefig(f'rednderpeople_seq_000000_triplane_256x256x96_norm_diff_steps_1000_hist_0.png')
    # plt.close()
    # plt.hist(x_new, bins=100)
    # plt.savefig(f'rednderpeople_seq_000000_triplane_256x256x96_norm_diff_steps_1000_hist_scale_log_0.png')

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{suffix}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

        # for i in range(5):
        #     imageio.imwrite(os.path.join(args.log_dir, f'gen_{i}.png'), arr[i])

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        log_dir="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

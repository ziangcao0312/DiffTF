from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os, imageio, time, copy, math, json
from random import sample
from mpi4py import MPI
from improved_diffusion import dist_util, logger

import torch

def load_triplane_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = TriplaneDataset(
        image_size,
        data_dir,
        classes=class_cond,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False, persistent_workers=False
        )
    while True:
        yield from loader
class TriplaneDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_dir,
        classes=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.classes=classes
        self.resolution = resolution
        allname=os.listdir(data_dir)
        allname.sort()  
        idx=0


        label=[]
        path=[]

        for name in allname:
            label.append(idx)
            path.append(os.path.join(data_dir,name))

        self.path=path
        self.class_cond=torch.Tensor(label)

       
    def __len__(self):
        return self.class_cond.shape[0]

    def __getitem__(self, idx):

        tri_planes=torch.load(self.path[idx],map_location='cpu')
        norm=torch.abs(tri_planes).max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        tri_planes=(tri_planes/norm)[0]

        out_dict={}
        if self.classes:
            
            out_dict["y"] = np.array(self.class_cond[idx], dtype=np.int64)

        return tri_planes, out_dict

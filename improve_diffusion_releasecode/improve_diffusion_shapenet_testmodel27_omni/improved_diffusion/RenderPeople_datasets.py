from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os, imageio, cv2, time, copy, math, json
from random import sample
from cv2 import Rodrigues as rodrigues
import ipdb
import blobfile as bf
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
        #ipdb.set_trace()
        allname=os.listdir(data_dir)
        allname.sort()  
        idx=0
        related={}


        label=[]
        path=[]
        indexfor={}

        for name in allname:
            # usedclass=np.loadtxt('/mnt/petrelfs/caoziang/3D_generation/data/selcet100.txt',dtype='str')
            # if name[:-8] not in usedclass:
            #     continue
            tripath=os.path.join(data_dir,name)
            for triname in os.listdir(tripath):
                path.append(os.path.join(tripath,triname))
                label.append(idx)
            indexfor[name]=idx
            idx+=1
        import json
        json_str = json.dumps(indexfor)
        with open('class_index.json', 'w') as json_file:
            json_file.write(json_str)
 

    

       
        #ipdb.set_trace()

        self.path=path
        self.class_cond=torch.Tensor(label)

        #print('Reloading from', ckpt_path)
        #ckpt = torch.load(ckpt_path, map_location='cpu')


        #ipdb.set_trace()
        # class_cond= torch.Tensor(\
        #     [0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 20, 21, 22, 22, 22, 22, 22, 22, 23, 23, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 28, 28, 29, 30, 30, 30, 31, 32, 32, 32, 33, 33, 34, 35, 36, 37, 37, 37, 38, 38, 38, 39, 39, 40, 40, 40, 41, 41, 41, 41, 42, 43, 43, 43, 44, 44, 45, 45, 45, 45, 46, 46, 47, 48, 49, 50, 50, 50, 51, 51, 51, 52, 52, 53, 53, 53, 53, 54, 54, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 62, 63, 63, 63, 64, 65, 65, 66, 67, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 73, 73, 73, 73, 74, 74, 75, 75, 75, 75, 75, 76, 77, 77, 77, 77, 77, 78, 78, 78, 79, 80, 81, 81, 82, 83, 83, 84, 84, 84, 85, 85, 85, 86, 87, 88, 88, 88, 89, 89, 90, 90, 90, 91, 91, 92, 93, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 96, 97, 98, 99, 99, 100, 100, 101, 101, 101, 101, 102, 102, 103, 104, 104, 104, 104, 105, 105, 105, 105, 106, 107, 107, 107, 108, 108, 109, 110, 110, 110, 110, 111, 111, 111, 112, 112, 112, 112, 113, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 117, 118, 118, 118, 118, 118, 119, 119, 119, 120, 121, 121, 121, 122, 122, 122, 122, 123, 123, 123, 123, 124, 125, 125, 125, 126, 127, 127, 128, 128, 128, 129, 130, 130, 131, 131, 132, 132, 133, 133, 134, 135, 135, 135, 136, 136, 137, 137, 137, 137, 138, 139, 140, 141, 141, 141, 142, 142, 142, 142, 143, 144, 144, 144, 145, 146, 147, 148, 148, 148, 148, 149, 149, 150, 150, 150, 151, 152, 153, 154, 154, 155, 156, 156, 156, 157, 157, 158, 158, 159, 160, 161, 161, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 166, 167, 167, 167, 167, 167, 167, 168, 168, 168, 168, 169, 169, 169, 169, 170, 170, 170, 171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 173, 173, 174, 175, 175, 176, 177, 177, 177, 178, 178, 178, 178, 178, 179, 179, 180, 181, 181, 181, 181, 182, 182, 183, 183, 183, 184, 184]
        #     )

        # tri_plane=tri_plane.repeat(16,1,1,1)
        # class_cond=class_cond.repeat(16)

        #self.local_images = tri_plane[shard:][::num_shards]
        #self.class_cond= label[shard:][::num_shards]

    def __len__(self):
        # return len(self.local_images)
        return self.class_cond.shape[0]

    def __getitem__(self, idx):

        tri_planes=torch.load(self.path[idx],map_location='cpu')
        norm=torch.abs(tri_planes).max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #norm=torch.norm(tri_planes, dim=1,keepdim = True)#torch.abs(tri_planes).max(2)[0].max(2)[0].unsqueeze(-1).unsqueeze(-1)

        tri_planes=(tri_planes/norm)[0]
        #tri_planes=torch.tanh(tri_planes)[0]#.clip(-1,1)[0]

        out_dict={}
        if self.classes:
            
            out_dict["y"] = np.array(self.class_cond[idx], dtype=np.int64)

        return tri_planes, out_dict

# class TriplaneDataset(Dataset):
#     def __init__(
#         self,
#         resolution,
#         data_dir,
#         classes=None,
#         shard=0,
#         num_shards=1,
#     ):
#         super().__init__()
#         self.classes=classes
#         self.resolution = resolution
#         #ipdb.set_trace()
#         ckpt_path = os.path.join('/nvme/caoziang/3D_generation/Checkpoint_all/aa.npz')
#         print('Reloading from', ckpt_path)
#         #ckpt = torch.load(ckpt_path, map_location='cpu')
#         tri_plane = torch.Tensor(np.load(ckpt_path)['triplane']).to(dist_util.dev())
#         print(tri_plane.shape)
#         #ipdb.set_trace()
#         norm=[]
#         for i in range(len(tri_plane)):
#             norm.append(torch.abs(tri_plane)[i].max())
#         self.norm=torch.stack(norm).unsqueeze(1).unsqueeze(1).unsqueeze(1)
#         tri_plane=tri_plane/9
#         tri_plane=tri_plane.clip(-1,1)
#         #tri_plane=tri_plane/4#/self.norm

#         #ipdb.set_trace()
#         class_cond= torch.Tensor(\
#             [0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 20, 21, 22, 22, 22, 22, 22, 22, 23, 23, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 28, 28, 29, 30, 30, 30, 31, 32, 32, 32, 33, 33, 34, 35, 36, 37, 37, 37, 38, 38, 38, 39, 39, 40, 40, 40, 41, 41, 41, 41, 42, 43, 43, 43, 44, 44, 45, 45, 45, 45, 46, 46, 47, 48, 49, 50, 50, 50, 51, 51, 51, 52, 52, 53, 53, 53, 53, 54, 54, 54, 55, 55, 56, 57, 57, 58, 59, 59, 60, 61, 62, 63, 63, 63, 64, 65, 65, 66, 67, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 73, 73, 73, 73, 74, 74, 75, 75, 75, 75, 75, 76, 77, 77, 77, 77, 77, 78, 78, 78, 79, 80, 81, 81, 82, 83, 83, 84, 84, 84, 85, 85, 85, 86, 87, 88, 88, 88, 89, 89, 90, 90, 90, 91, 91, 92, 93, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 96, 97, 98, 99, 99, 100, 100, 101, 101, 101, 101, 102, 102, 103, 104, 104, 104, 104, 105, 105, 105, 105, 106, 107, 107, 107, 108, 108, 109, 110, 110, 110, 110, 111, 111, 111, 112, 112, 112, 112, 113, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 117, 118, 118, 118, 118, 118, 119, 119, 119, 120, 121, 121, 121, 122, 122, 122, 122, 123, 123, 123, 123, 124, 125, 125, 125, 126, 127, 127, 128, 128, 128, 129, 130, 130, 131, 131, 132, 132, 133, 133, 134, 135, 135, 135, 136, 136, 137, 137, 137, 137, 138, 139, 140, 141, 141, 141, 142, 142, 142, 142, 143, 144, 144, 144, 145, 146, 147, 148, 148, 148, 148, 149, 149, 150, 150, 150, 151, 152, 153, 154, 154, 155, 156, 156, 156, 157, 157, 158, 158, 159, 160, 161, 161, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 166, 167, 167, 167, 167, 167, 167, 168, 168, 168, 168, 169, 169, 169, 169, 170, 170, 170, 171, 171, 171, 171, 171, 171, 171, 172, 172, 172, 172, 172, 172, 173, 173, 174, 175, 175, 176, 177, 177, 177, 178, 178, 178, 178, 178, 179, 179, 180, 181, 181, 181, 181, 182, 182, 183, 183, 183, 184, 184]
#             )

#         # tri_plane=tri_plane.repeat(16,1,1,1)
#         # class_cond=class_cond.repeat(16)

#         self.local_images = tri_plane[shard:][::num_shards]
#         self.class_cond= class_cond[shard:][::num_shards]

#     def __len__(self):
#         # return len(self.local_images)
#         return self.local_images.shape[0]

#     def __getitem__(self, idx):
#         out_dict={}
#         if self.classes:
            
#             out_dict["y"] = np.array(self.class_cond[idx], dtype=np.int64)

#         return self.local_images[idx], out_dict


# ################################################################################


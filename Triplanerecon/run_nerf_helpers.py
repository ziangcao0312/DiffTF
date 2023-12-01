import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm, trange
from configparse import config_parser
import matplotlib.pyplot as plt
import ipdb
from run_nerf_helpers import *
from torch.utils.data import DataLoader
import logging
import plyfile
import meshio
import imageio
import random
import time
import skimage.measure
# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

DEBUG = False
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )


    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 2] = (voxel_grid_origin[0] + verts[:, 0] - verts[:, 0].mean())
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1] - verts[:, 1].mean()
    mesh_points[:, 0] = voxel_grid_origin[2] + verts[:, 2] - verts[:, 2].mean()

    # apply additional offset and scale
    scale=mesh_points.max()-mesh_points.min()  #6*mesh_points.std()
    mesh_points = mesh_points / scale

    
    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")

def convert_sdf_samples_to_ply_omni(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 2] = (voxel_grid_origin[0] + verts[:, 0] - verts[:, 0].mean())
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1] - verts[:, 1].mean()
    mesh_points[:, 0] = voxel_grid_origin[2] + verts[:, 2] - verts[:, 2].mean()

    # apply additional offset and scale
    scale=mesh_points.max()-mesh_points.min()  #6*mesh_points.std()
    mesh_points = mesh_points / scale

   
    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn,label, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [inputs.shape[0],-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        if viewdirs.shape!=inputs.shape:
            input_dirs = viewdirs[:,:,None].expand(inputs.shape)
        else:
            input_dirs=viewdirs
        input_dirs_flat = torch.reshape(input_dirs, [inputs.shape[0],-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)

    input_all=torch.cat([inputs_flat,embedded_dirs],-1)
    outputs_flat =  fn(input_all,label)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat,label, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[1], chunk):
        ret = render_rays(rays_flat[:,i:i+chunk],label=label, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(chunk=1024*32, rays=None, c2w=None, ndc=True,label=None,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    rays_o, rays_d = rays[:,0,...], rays[:,1,...]

    
    viewdirs = rays_d

    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [rays_d.shape[0],-1,3]).float()

    sh = rays_d.shape 

    # Create ray batch
    rays_o = torch.reshape(rays_o, [sh[0],-1,3]).float()
    rays_d = torch.reshape(rays_d, [sh[0],-1,3]).float()

    near, far = near[:,None,:] * torch.ones_like(rays_d[...,:1]), far[:,None,:] * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    # Render and reshape
    all_ret = batchify_rays(rays, label,chunk,**kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[2:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def generate_mesh(args,path,network_query_fn,network_fn,mesh_mode='.obj'):
           
    max_batch=1000000
    device=network_fn.tri_planes.device
    samples, voxel_origin, voxel_size = create_samples(N=args.shape_res, voxel_origin=[0, 0, 0], cube_length=args.box_warp * 1)#.reshape(1, -1, 3)
    samples = samples.to(device=device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                
                sigma = network_query_fn(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], torch.zeros(1).long().to(device=device), network_fn)[...,3:] 
                
                
                sigmas[:, head:head+max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)
    sigmas = sigmas.reshape((args.shape_res, args.shape_res, args.shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * args.shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    
    
    if args.dataset=='shapenet':
        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, path, level=10)
    else:
        convert_sdf_samples_to_ply_omni(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, path, level=10)
    mesh=meshio.read(path)

    mesh.write(path[:-4]+args.mesh_mode)


def render_path1(batch_rays, chunk, render_kwargs, gt_imgs=None, savedir=None,savedir1=None,savedir2=None,savedir3=None,savedir4=None,near=None,far=None,label=None):



    rgbs = []
    disps = []

    t = time.time()
    
    rgbs, disps, acc, _ = render(chunk=chunk, rays=batch_rays,near=near,far=far, label=label,**render_kwargs)
    reso=int(rgbs.shape[-2]**0.5)
    rgbs=rgbs.view(-1,reso,reso,3)
    disps=disps.view(-1,reso,reso,1)
    acc=acc.view(-1,reso,reso,1)
    gt_imgs=gt_imgs.view(-1,reso,reso,3)
    if render_kwargs['white_bkgd']:
        mask=(gt_imgs.mean(-1)<0.9999)
    else:
        mask=(gt_imgs.mean(-1)>1e-4)
    mask_img=torch.zeros_like(acc)
    mask_img[mask]=1
    triplane=(render_kwargs['network_fn'].tri_planes)
    norm=torch.abs(triplane).max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    triplane=triplane/norm
    b,c,w,h=triplane.shape
    imageio.imwrite(savedir4,np.uint8(to_rgb_triplane(triplane.reshape(b,3,c//3,w,h).permute(0,2,3,1,4).reshape(b,c//3,w,h*3))[0]))
    if savedir is not None:
        for i in range(len(rgbs)):

            rgb8 =to8b(rgbs[i].cpu().numpy())
            filename = os.path.join(savedir)
            imageio.imwrite(savedir, rgb8)
            imageio.imwrite(savedir1, np.uint8(gt_imgs[i].cpu().numpy()*255))
            imageio.imwrite(savedir2,np.uint8(acc[0].cpu().numpy()*255).repeat(3,axis=-1))
            imageio.imwrite(savedir3,np.uint8(mask_img[0].cpu().numpy()*255).repeat(3,axis=-1))
            print('psnr:' ,mse2psnr(img2mse(torch.Tensor(rgb8/255).to(device=gt_imgs.device)[mask[i]],(gt_imgs[i])[mask[i]])))
            print('psnr_all:' ,mse2psnr(img2mse(torch.Tensor(rgb8/255).to(device=gt_imgs.device),(gt_imgs[i]))))
    return rgbs, disps,acc
def to_rgb_triplane(plane):
        x = plane.float()
       
        colorize_triplane = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = torch.nn.functional.conv2d(x, weight=colorize_triplane)
        x = ((x - x.min()) / (x.max() - x.min()) * 255.).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        return x
def render_path2(batch_rays, chunk, render_kwargs, savedir=None,savedir1=None,savedir2=None,savedir3=None,near=None,far=None,label=None):



    rgbs = []
    disps = []

    t = time.time()

    rgbs, disps, acc, _ = render(chunk=chunk, rays=batch_rays,near=near,far=far, label=label,**render_kwargs)
    reso=int(rgbs.shape[-2]**0.5)
    rgbs=rgbs.view(-1,reso,reso,3)
    disps=disps.view(-1,reso,reso,1)
    acc=acc.view(-1,reso,reso,1)
    acc_mask=torch.ones_like(acc)

    acc_mask[torch.where(acc<0.8)]=0
    
    triplane=(render_kwargs['network_fn'].tri_planes)
    norm=torch.abs(triplane).max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    b,c,w,h=triplane.shape
    triplane=triplane/norm
    if savedir3 is not None:
        imageio.imwrite(savedir3,np.uint8(to_rgb_triplane(triplane.reshape(b,3,c//3,w,h).permute(0,2,3,1,4).reshape(b,c//3,w,h*3))[0]))
    
    for i in range(len(rgbs)):

        rgb8 =to8b(rgbs[i].cpu().numpy())
        
        if savedir is not None:
            imageio.imwrite(savedir, rgb8)
        if savedir1 is not None:
            if render_kwargs['white_bkgd']:
                imageio.imwrite(savedir1, np.uint8(rgb8*acc_mask[0].cpu().numpy()+(1-acc_mask[0].cpu().numpy())*255))
            else:
                imageio.imwrite(savedir1, np.uint8(rgb8*acc_mask[0].cpu().numpy()))
        if savedir2 is not None:
            imageio.imwrite(savedir2,np.uint8(acc[0].cpu().numpy()*255).repeat(3,axis=-1))
    if render_kwargs['white_bkgd']:
        rgbs=np.uint8(rgb8*acc_mask[0].cpu().numpy()+(1-acc_mask[0].cpu().numpy())*255)
    else:
        rgbs=np.uint8(rgb8*acc_mask[0].cpu().numpy())


    return rgbs, disps,acc



def raw2outputs(mode,raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw-1)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1) 

    rgb = torch.sigmoid(raw[...,:3])*(1 + 2*0.001) - 0.001  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    if mode!='train':
        alpha[torch.where((alpha>0)&(alpha<0.01))]=0
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0],alpha.shape[1], 1)), 1.-alpha + 1e-10], -1), -1)[:,:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])


    return rgb_map, disp_map, acc_map, weights, depth_map



def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out
def sample_stratified(ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False, det=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                if det:
                    depths_coarse += 0.5 * depth_delta[..., None]
                else:
                    depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                if det:
                    depths_coarse += 0.5 * depth_delta
                else:
                    depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse
def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    #ipdb.set_trace()
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)


    bb_min = [-1*(box_side_length/2), -1*(box_side_length/2), -1*(box_side_length/2)]
    bb_max = [1*(box_side_length/2), 1*(box_side_length/2), 1*(box_side_length/2)]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)
def unify_samples(depths1, colors1, densities1, depths2, colors2, densities2):
    all_depths = torch.cat([depths1, depths2], dim = -2)
    all_colors = torch.cat([colors1, colors2], dim = -2)
    all_densities = torch.cat([densities1, densities2], dim = -2)

    _, indices = torch.sort(all_depths, dim=-2)
    all_depths = torch.gather(all_depths, -2, indices)
    all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
    all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

    return all_depths, all_colors, all_densities
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                mode='train',
                raw_noise_std=0.,
                label=None,
                verbose=False,
                pytest=False,
                dataset='omni',
                box_warp=4.0):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    B,N_rays,_ = ray_batch.shape
    
    rays_o, rays_d = ray_batch[:,:,0:3], ray_batch[:,:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,:,-3:] 
    bounds = torch.reshape(ray_batch[...,6:8], [B,-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)

    z_vals = near * (1.-t_vals) + far * (t_vals)
    

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    if dataset=='shapenet':
        ray_start, ray_end = get_ray_limits_box(rays_o, rays_d, box_side_length=box_warp)
        is_ray_valid = ray_end > ray_start
        if torch.any(is_ray_valid).item():
            ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
            ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        z_vals = sample_stratified(rays_o, ray_start, ray_end, N_samples,det=(perturb==0.))[...,0]

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, label,network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(mode,raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples=torch.zeros(z_vals_mid.shape[0],z_vals_mid.shape[1],N_importance)
        for idd in range(z_vals_mid.shape[0]):
            z_samples[idd] = sample_pdf(z_vals_mid[idd], weights[idd][...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        
        z_samples = z_samples.detach()

        
        z_vals, idx = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts_new = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None] # [N_rays, N_samples + N_importance, 3]


        raw_new = network_query_fn(pts_new, viewdirs, label, network_fn)
        raw=torch.cat([raw, raw_new], -2)
        raw=torch.gather(raw, -2, idx.unsqueeze(-1).expand(-1, -1, -1, raw.shape[-1]))

        
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(mode,raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['sigmaformesh']=raw[...,:3]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret
def create_nerf(args,name=None):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    if args.state!='testddpm':
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=args.triplanechannel, size=args.triplanesize,output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,num_instance=args.num_instance,box_warp=args.box_warp)
    else:
        model = ddpmNeRF(D=args.netdepth, W=args.netwidth,
            input_ch=args.triplanechannel, size=args.triplanesize,output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,num_instance=args.num_instance,box_warp=args.box_warp)
    grad_vars = list(model.parameters())
    if args.state=='train_single' :
        trainable_params=[]
        trainable_params+=[{'params':model.tri_planes,
        'lr': args.lrate}]
    else:
        trainable_params=[]
        trainable_params+=[{'params':model.tri_planes,
        'lr': args.lrate*10}]
        trainable_params+=[{'params':model.pts_linears.parameters(),
        'lr': args.lrate}]
        trainable_params+=[{'params':model.views_linears.parameters(),
        'lr': args.lrate}]
        trainable_params+=[{'params':model.feature_linear.parameters(),
        'lr': args.lrate}]
        trainable_params+=[{'params':model.alpha_linear.parameters(),
        'lr': args.lrate}]
        trainable_params+=[{'params':model.rgb_linear.parameters(),
        'lr': args.lrate}]
        

    model_fine = None


    network_query_fn = lambda inputs, viewdirs, label,network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,label=label,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(trainable_params, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.load_weight==1 or args.state=='test':
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            model.load_state_dict(ckpt['network_fn_state_dict'])
    elif args.state=='train_single' :
        
        ckpt = torch.load(args.decoderdir)

        ckpt['network_fn_state_dict']['tri_planes']=torch.randn(1,args.triplanechannel,args.triplanesize,args.triplanesize)
        model.load_state_dict(ckpt['network_fn_state_dict'])
    elif args.state=='test_single' :
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname,name, f) for f in sorted(os.listdir(os.path.join(basedir, expname,name))) if 'max' not in f and 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            model.load_state_dict(ckpt['network_fn_state_dict'])
    elif args.state=='testddpm':
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)


            model.load_state_dict(ckpt['network_fn_state_dict'])
    else:
        print('error state')
            
    if args.ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = model.to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank).to(local_rank)
    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'N_importance' : args.N_importance,
        'dataset': args.dataset,
        'box_warp': args.box_warp

    }

    
    print('Not ndc!')
    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
class Triplane(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        
        self.plane_axis=self.generate_planes()
    def generate_planes(self):
        """
        Defines planes by the three vectors that form the "axes" of the
        plane. Should work with arbitrary number of planes and planes of
        arbitrary orientation.
        """
        return torch.tensor([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]],
                                [[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]],
                                [[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0]]], dtype=torch.float32)

    def project_onto_planes(self,planes, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
        inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3).to(device=coordinates.device)
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]

    def sample_from_planes(self,plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N*n_planes, C, H, W)

        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
        coordinates = self.project_onto_planes(plane_axes, coordinates).unsqueeze(1)
        
        output_features = torch.nn.functional.grid_sample(plane_features, coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)  # xy,xz,zy
        
        return output_features
        

    def forward(self, planes, sample_coordinates,box=1):


        return self.sample_from_planes(self.plane_axis, planes, sample_coordinates, padding_mode='zeros', box_warp=box)

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts
def exists(val):
    return val is not None
def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, size=256,input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,num_instance=1,box_warp=4):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch//3
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.hidden_dim=W

        self.triplane=Triplane()
        
        self.tri_planes = nn.Parameter(torch.randn(num_instance, input_ch, size, size))
       
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])

        self.softplus=nn.Softplus()

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W, 3)
        self.box_warp=box_warp
        
                
        

    def forward(self, x,label):
        

        input_pts, input_views = torch.split(x, [int(x.shape[-1]-self.input_ch_views), self.input_ch_views], dim=-1)
        B,N,M=input_views.shape

       
        triplane=self.tri_planes[label]
        norm=torch.abs(triplane).max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        triplane=triplane/norm
        
        sample_triplane=(triplane).view(B,3,self.tri_planes.shape[-3]//3,self.tri_planes.shape[-2],self.tri_planes.shape[-1])
        
        input_pts=(self.triplane(sample_triplane,input_pts,self.box_warp)).permute(0,2,1,3).reshape(B*N,self.tri_planes.shape[-3])
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views.view(B*N,M)], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb.view(B,N,3), alpha.view(B,N,1)], -1)
        
        return outputs 


class ddpmNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, size=256,input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,num_instance=1,box_warp=4):
        """ 
        """
        super(ddpmNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch//3
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.hidden_dim=W

        self.triplane=Triplane()
        
        self.tri_planes = nn.Parameter(torch.randn(num_instance, input_ch, size, size))
        
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W)])

        self.softplus=nn.Softplus()

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W, 3)
        self.box_warp=box_warp
       
                
        

    def forward(self, x,label):
        

        input_pts, input_views = torch.split(x, [int(x.shape[-1]-self.input_ch_views), self.input_ch_views], dim=-1)
        B,N,M=input_views.shape

        triplane=self.tri_planes[label]
        
        sample_triplane=(triplane).view(B,3,self.tri_planes.shape[-3]//3,self.tri_planes.shape[-2],self.tri_planes.shape[-1])
        input_pts=(self.triplane(sample_triplane,input_pts,self.box_warp)).permute(0,2,1,3).reshape(B*N,self.tri_planes.shape[-3])
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views.view(B*N,M)], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb.view(B,N,3), alpha.view(B,N,1)], -1)
        
        return outputs 
# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

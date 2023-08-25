cd /nvme/caoziang/3D_generation/nerf-pytorch_finalcode
export CUDA_VISIBLE_DEVICES=1
conda activate plenoxel
python run_nerf.py --config ./configs/train.txt


srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python -m torch.distributed.launch --nproc_per_node 1 train.py --config ./configs/shapenet/train.txt

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python -m torch.distributed.launch --nproc_per_node 1 train.py --config ./configs/omni/train.txt

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python test.py --config ./configs/shapenet/test.txt

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python test.py --config ./configs/omni/test.txt

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_omni.py --config ./configs/omni/train_single.txt --idx 1

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_shapenet.py --config ./configs/shapenet/train_single.txt

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_omni.py --config ./configs/omni/test_single.txt

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_shapenet.py --config ./configs/shapenet/test_single.txt

source ~/.bashrc
cd /mnt/petrelfs/caoziang/3D_generation/nerf-pytorch_finalcode_ddp_1triplane_new
conda activate diffusion
srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_shapenet_all.py --config ./configs/shapenet/train_single_all.txt --idx 


nohup srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:8 -n8 --ntasks-per-node=8 --job-name=diffusion --kill-on-bad-exit=1 python -m torch.distributed.launch --nproc_per_node 8 run_nerf_ddp.py --config ./configs/train.txt &

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 8 run_nerf_ddp.py --config ./configs/train.txt

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 --pty bash -i 

srun -p 3dobject_aigc --mpi=pmi2 --gres=gpu:8 -n8 --ntasks-per-node=8 --job-name=diffusion --kill-on-bad-exit=1 --pty bash -i 

srun -p super_priority --mpi=pmi2 --job-name=diffusion --kill-on-bad-exit=1  --pty bash -i 



srun -p priority --mpi=pmi2 --gres=gpu:4 -n4 --ntasks-per-node=4 --job-name=diffusion --kill-on-bad-exit=1 python -m torch.distributed.launch --nproc_per_node 4 run_nerf_ddp.py --config ./configs/train.txt

srun -p priority --mpi=pmi2 --gres=gpu:2 -n2 --ntasks-per-node=2 --job-name=diffusion --kill-on-bad-exit=1 python run_nerf.py --config ./configs/train.txt

srun -p priority --mpi=pmi2 --gres=gpu:$gpu_num -n$gpu_num --mem-per-gpu=32G --ntasks-per-node=$gpu_num --job-name=improved_diffusion --kill-on-bad-exit=1


cd /mnt/petrelfs/caoziang/3D_generation/nerf-pytorch_finalcode_ddp_1triplane_new_new
srun -p 3dobject_aigc_scale --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_shapenet.py --config ./configs/shapenet/train_single.txt --idx 

srun -p 3dobject_aigc_light --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_shapenet.py --config ./configs/shapenet/train_single.txt --idx 


cd /mnt/petrelfs/caoziang/3D_generation/nerf-pytorch_finalcode_ddp_1triplane_new_new

srun -p 3dobject_aigc_scale --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=diffusion --kill-on-bad-exit=1 python train_single_shapenet.py --config ./configs/omni/train_single.txt --idx 
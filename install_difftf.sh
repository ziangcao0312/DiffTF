conda create -n difftf python=3.10
conda activate difftf

pip install conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install mpi4py
pip install opencv-python imageio tqdm blobfile==2.0.2 einops scikit-image logging configargparse ipdb matplotlib kaolin h5py
pip install meshio
pip install plyfile

cd ./3dDiffusion
python setup.py install 
cd ..

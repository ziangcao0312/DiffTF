GPU_NUM=$1
for ((allindex=0;allindex<${GPU_NUM};allindex++));  
do
    tmux new-session -d -s "alltriplane_fitting_"${allindex} "echo ${allindex}; conda activate difftf; python ./Triplanerecon/train_single_shapenet.py --config ./Triplanerecon/configs/shapenet/train_single.txt --num_gpu ${GPU_NUM} --idx ${allindex} --datadir ./dataset/ShapeNet/renders_car --basedir ./Checkpoint --expname shapenet_triplane --decoderdir ./Checkpoint/shapenet_sharedecoder/300000.tar"

done
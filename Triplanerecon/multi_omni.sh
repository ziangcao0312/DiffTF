GPU_NUM=$1
for ((allindex=0;allindex<${GPU_NUM};allindex++));  
do
    tmux new-session -d -s "alltriplane_fitting_"${allindex} "echo ${allindex}; conda activate difftf; python ./Triplanerecon/train_single_omni.py --config ./Triplanerecon/configs/omni/train_single.txt --num_gpu ${GPU_NUM} --idx ${allindex} --datadir ./dataset/Omniobject3D/renders --basedir ./Checkpoint --expname omni_triplane --decoderdir ./Checkpoint/omni_sharedecoder/300000.tar"

done
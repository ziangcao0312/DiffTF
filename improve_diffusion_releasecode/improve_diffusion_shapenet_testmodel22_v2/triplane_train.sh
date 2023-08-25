
ID=$1
diff_steps=$2
gpu_num=$3
image_size=$4
in_channels=$5
num_channels=$6 # 128
batch_size=$7 # 128


JOB_NAME="renderpeople_"

EXPS="/nvme/caoziang/3D_generation/Checkpoint_all"
if ! [ -d "$EXPS/$JOB_NAME" ]; then
   mkdir -p $EXPS/$JOB_NAME
fi

DATA_FLAGS="--data_name renderpeople"

MODEL_FLAGS="--image_size $image_size --num_channels 192 --num_res_blocks 2 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--in_channels $in_channels --out_channels $in_channels --diffusion_steps $diff_steps --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 5e-5 --batch_size $batch_size"

mpiexec -n $gpu_num python scripts/image_train.py  --log_dir $EXPS/$JOB_NAME $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS 



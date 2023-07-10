#!/usr/bin/env bash

set -u
set -e

exp_name=`basename ${0%.*}`
exp_name=${exp_name: 4}
current_file=`realpath $0`

##'0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='6'
nproc_per_node=`expr ${#CUDA_VISIBLE_DEVICES} + 1`
nproc_per_node=`expr $nproc_per_node / 2`
port=6668

data_type='FFHQ'

batch=8
size=512
size_d=512  # discriminator size
dataset_imgsize=1024

n_mlp=8

path='/data1/chenlong/0517/man2/smooth/'
ckpt='/data/yinlulu/glint_pro/stylegan3_glint/pretrained_models/stylegan2-ffhq-512x512.pt',

ckpt_save_dir='./checkpoints'
mkdir -p $ckpt_save_dir

cp $current_file $ckpt_save_dir
cp config.py $ckpt_save_dir
#cp param_pairs.py $ckpt_save_dir

r1=10  #10
path_regularize=2  # 2
d_reg_every=16  # 16
g_reg_every=4  # 4
lr=0.002  # 2e-3
augment_p=0  # 0

python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$port train.py \
    --arch stylegan2 \
    --iter 990000 \
    --batch $batch \
    --n_sample 1 \
    --size $size \
    --size_d $size_d \
    --dataset_imgsize $dataset_imgsize \
    --r1 $r1 \
    --path_regularize $path_regularize \
    --path_batch_shrink 2 \
    --d_reg_every $d_reg_every \
    --g_reg_every $g_reg_every \
    --mixing 0.9 \
    --ckpt $ckpt \
    --checkpoints_dir $ckpt_save_dir \
    --lr $lr \
    --channel_multiplier 2 \
    --n_mlp=$n_mlp \
    --augment_p $augment_p \
    --ada_target 0.6 \
    --ada_length 500000 \
    --ada_every 256 \
    --path $path \

#--wandb \
#--augment \

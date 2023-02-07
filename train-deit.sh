#!/usr/bin/env bash

# Train DeiT models, consisting of these configurations:
# 1. deit_tiny_patch16_224
# 2. deit_small_patch16_224
# 3. deit_base_patch16_224
# 4. deit_medium_patch16_LS

data_dir=${1:-/nas/common_data/ILSVRC2012}
model=${2:-deit_tiny_patch16_224}
output_dir=${3:-./output}
mkdir -p $output_dir

/usr/bin/env python3 -m torch.distributed.launch \
    --nproc_per_node=1 --use_env \
    main.py \
    --model $model \
    --batch-size 256 \
    --epoch 1 \
    --output_dir $output_dir \
    --data-path $data_dir

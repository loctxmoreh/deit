#!/usr/bin/env bash

# Train CaiT models, consisting of these configurations:
# cait_S24_224, cait_XXS24_224, cait_XXS36_224

data_dir=${1:-/nas/common_data/ILSVRC2012}
model=${2:-cait_XXS24_224}
output_dir=${3:-./output}

mkdir -p $output_dir

/usr/bin/env python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    main.py \
    --model $model \
    --batch-size 32 \
    --epoch 1 \
    --output_dir $output_dir \
    --data-path $data_dir

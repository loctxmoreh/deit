#!/usr/bin/env bash

# Train PatchConvNet models, consisting of these configurations:
# S60, S120, B60, B120, L60, L120, S60_multi

data_dir=${1:-/nas/common_data/ILSVRC2012}
model=${2:-S60}
output_dir=${3:-./output}
mkdir -p $output_dir

/usr/bin/env python3 -m torch.distributed.launch \
    --nproc_per_node=1 --use_env \
    main.py \
    --model $model \
    --batch-size 32 \
    --epoch 1 \
    --output_dir $output_dir \
    --data-path $data_dir

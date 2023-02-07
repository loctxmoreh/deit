#!/usr/bin/env bash

# Train ResMLP models, consisting of these configurations:
# resmlp_12, resmlp_24, resmlp_36, resmlpB_24

data_dir=${1:-/nas/common_data/ILSVRC2012}
model=${2:-resmlp_12}
output_dir=${3:-./output}
mkdir -p $output_dir

/usr/bin/env python3 -m torch.distributed.launch \
    --nproc_per_node=1 --use_env \
    main.py \
    --model $model \
    --batch-size 32 \
    --epoch 1 \
    --drop 0.0 --drop-path 0.0 \
    --input-size 224 \
    --output_dir $output_dir \
    --data-path $data_dir

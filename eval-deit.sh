#!/usr/bin/env bash

# Download & evaluate all pretrained checkpoints of DeiT provided by original authors

set -e

data_dir=${1:-/nas/common_data/ILSVRC2012}
[[ ! -d $data_dir ]] && echo "Data dir ${data_dir} not found" && exit 1

models=(
    deit_tiny_patch16_224
    deit_small_patch16_224
    deit_base_patch16_224
    deit_tiny_distilled_patch16_224
    deit_small_distilled_patch16_224
    deit_base_distilled_patch16_224
)

declare -A urls=(
    ["deit_tiny_patch16_224"]="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
    ["deit_small_patch16_224"]="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
    ["deit_base_patch16_224"]="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
    ["deit_tiny_distilled_patch16_224"]="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth"
    ["deit_small_distilled_patch16_224"]="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    ["deit_base_distilled_patch16_224"]="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth"
)

for model in ${models[@]}; do
    /usr/bin/env python3 main.py --eval \
        --model $model \
        --resume ${urls[$model]} \
        --batch-size 256 \
        --data-path $data_dir
done

# clean up downloaded checkpoints
rm $HOME/.cache/torch/hub/checkpoints/*.pth

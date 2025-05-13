#!/bin/bash

GPU=0
dataset=cifar80no
noise_type=symmetric
closeset_ratio=0.8
topk=3

while getopts g:d:n:c:k: flag
do
    case "${flag}" in
        g) GPU=${OPTARG};;
        d) dataset=${OPTARG};;
        n) noise_type=${OPTARG};;
        c) closeset_ratio=${OPTARG};;
        k) topk=${OPTARG};;
    esac
done

python our.py --config config/cifar100.cfg --synthetic-data ${dataset} --noise-type ${noise_type} --closeset-ratio ${closeset_ratio} \
                --gpu ${GPU} --net CNN --batch-size 128 --lr 0.05 --warmup-lr 0.05 --opt sgd --warmup-lr-scale 1 \
                --warmup-epochs 50 --lr-decay cosine:50,5e-5,300 --epochs 300 \
                --activation relu --low-dim 128\
                --temperature 0.1 --T 0.5 --alpha 1.0 --n-neighbors 200 \
                --lambda-bcl 0.3 --lambda-con 0.5 --top-k ${topk} --alpha-id 0.9 --alpha-ood 0.1\
                --log DRS --save-model
#!/bin/bash

DEVICE=0  # Set DEVICE to 0 or 1

docker-compose run --rm gpu python exec/train.py \
    experiments/1xxx_train/1102_vtnt_bert_224-448-denoise-5.yml \
    device=$DEVICE
docker-compose run --rm gpu python exec/train.py \
    experiments/1xxx_train/1105_vtnt_bert_448-denoise-5 \
    device=$DEVICE
docker-compose run --rm gpu python exec/train.py \
    experiments/1xxx_train/1109_vtnt_bert_512-1024-denoise-5.yml \
    device=$DEVICE
docker-compose run --rm gpu python exec/train.py \
    experiments/1xxx_train/1113_swin_large_bert_384 \
    device=$DEVICE
docker-compose run --rm gpu python exec/train.py \
    experiments/1xxx_train/1124_swin_large_bert_384_pil_pseudo.yml \
    device=$DEVICE
docker-compose run --rm gpu python exec/train.py \
    experiments/1xxx_train/1126_swin_large_bert_384_pil_pseudo_no-denoise.yml \
    device=$DEVICE
docker-compose run --rm gpu python exec/train.py \
    experiments/1xxx_train/1127_vtnt_bert_512-1024_pseudo_no-denoise.yml \
    device=$DEVICE

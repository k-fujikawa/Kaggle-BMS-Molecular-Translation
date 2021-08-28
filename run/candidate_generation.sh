#!/bin/bash

DEVICE=0  # Set DEVICE to 0 or 1

docker-compose run --rm gpu python exec/infer.py \
    output/1109_vtnt_bert_512-1024-denoise-5 \
    device=$DEVICE num_beams=1 dataset=test
docker-compose run --rm gpu python exec/infer.py \
    output/1109_vtnt_bert_512-1024-denoise-5 \
    device=$DEVICE num_beams=4 dataset=test

docker-compose run --rm gpu python exec/infer.py \
    output/1113_swin_large_bert_384 \
    device=$DEVICE num_beams=1 dataset=test
docker-compose run --rm gpu python exec/infer.py \
    output/1113_swin_large_bert_384 \
    device=$DEVICE num_beams=4 dataset=test

docker-compose run --rm gpu python exec/infer.py \
    output/9005_1102+1105+1106 \
    device=$DEVICE num_beams=1 dataset=test

docker-compose run --rm gpu python exec/infer.py \
    output/9006_1103+1106+1109 \
    device=$DEVICE num_beams=1 dataset=test
docker-compose run --rm gpu python exec/infer.py \
    output/9006_1103+1106+1109 \
    device=$DEVICE num_beams=4 dataset=test

docker-compose run --rm gpu python exec/infer.py \
    output/9007_1109+1113 \
    device=$DEVICE num_beams=1 dataset=test
docker-compose run --rm gpu python exec/infer.py \
    output/9007_1109+1113 \
    device=$DEVICE num_beams=4 dataset=test
docker-compose run --rm gpu python exec/infer.py \
    output/9007_1109+1113 \
    device=$DEVICE num_beams=8 dataset=test

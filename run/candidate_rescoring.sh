#!/bin/bash

DEVICE=0  # Set DEVICE to 0 or 1
CANDIDATE_PATH=./candidates.csv

docker-compose run --rm gpu python exec/rescore.py \
    output/1109_vtnt_bert_512-1024-denoise-5 \
    $CANDIDATE_PATH \
    device=$DEVICE batch_size=32 in_column=InChI train=False
docker-compose run --rm gpu python exec/rescore.py \
    output/1113_swin_large_bert_384 \
    $CANDIDATE_PATH \
    device=$DEVICE batch_size=32 in_column=InChI train=False
docker-compose run --rm gpu python exec/rescore.py \
    output/1124_swin_large_bert_384_pil_pseudo \
    $CANDIDATE_PATH \
    device=$DEVICE batch_size=32 in_column=InChI train=False
docker-compose run --rm gpu python exec/rescore.py \
    output/1126_swin_large_bert_384_pil_pseudo_no-denoise \
    $CANDIDATE_PATH \
    device=$DEVICE batch_size=32 in_column=InChI train=False
docker-compose run --rm gpu python exec/rescore.py \
    output/1127_vtnt_bert_512-1024_pseudo_no-denoise \
    $CANDIDATE_PATH \
    device=$DEVICE batch_size=32 in_column=InChI train=False

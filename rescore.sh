#!/bin/bash -ex

# ===== hyperparams =====
DEVICE=${DEVICE:-0}
BATCH_SIZE=${BATCH_SIZE:-32}

# ===== targets =====
TARGETs=(
    # valid sets
    valid_kf_0523.csv
    valid_kf_0525.csv
    valid_kf_0527.csv
    valid_yokoo_0527.csv
    valid_camaro_0525.csv
    valid_kf_0531_renormed.csv
    valid_kf_0531.csv

    # test sets
    test_kf_0523.csv
    test_kf_0525.csv
    test_kf_0527.csv
    test_yokoo_0527.csv
    test_camaro_0525.csv
    test_yokoo_0531.csv
    test_kf_0531_renormed.csv
    test_camaro_old_submissions.csv
    test_kf_0531.csv
    test_camaro_0531.csv
)

# ===== model =====
MODELDIRs=(
    output/1109_vtnt_bert_512-1024-denoise-5
    output/1113_swin_large_bert_384
    output/1119_swin_large_bert_384_bpe
    output/1124_swin_large_bert_384_pil_pseudo
    output/1126_swin_large_bert_384_pil_pseudo_no-denoise
    output/1127_vtnt_bert_512-1024_pseudo_no-denoise
)

for modeldir in ${MODELDIRs[@]}; do
    for target in ${TARGETs[@]}; do
        if [[ $target =~ test_.* ]]; then
            train=False
        else
            train=True
        fi
        docker-compose run --rm gpu python exec/rescore.py $modeldir input/kfujikawa/kf-bms-candidates-v2/${target} device=${DEVICE} batch_size=${BATCH_SIZE} in_column=InChI train=${train} resume=True
    done
done

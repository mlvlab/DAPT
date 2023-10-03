#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=DAPT

DATASET=$1
SHOTS=$2
DEVICE=$3

for SEED in 1 2 3
do
    if [ ${DATASET} == "imagenet" ]; then
            CFG=vit_b16_ep50
        elif [ ${SHOTS} -eq 1 ]; then
            CFG=vit_b16_ep50
        elif [ ${SHOTS} -eq 2 ] || [ ${SHOTS} -eq 4 ]; then
            CFG=vit_b16_ep100
        elif [ ${SHOTS} -eq 8 ] || [ ${SHOTS} -eq 16 ]; then
            CFG=vit_b16
    fi
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
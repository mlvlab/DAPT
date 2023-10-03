#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=DAPT
CFG=vit_b16_ep50
SHOTS=16

DEVICE=$1

for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r
do
    for SEED in 1 2 3
    do
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
            --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
            --load-epoch 50 \
            --eval-only
        fi
    done
done
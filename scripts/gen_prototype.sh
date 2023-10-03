#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=DAPT

DEVICE=$1

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
    do
    for SHOTS in 1 2 4 8 16
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

        for SEED in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=${DEVICE} \
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.DAPT.PROTOTYPE_GEN True
        done
    done
done
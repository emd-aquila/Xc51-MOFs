#!/bin/bash

python ./unimat/infer.py . \
    --user-dir ./unimat \
    --path ./logs_weighted/checkpoint.best_f1_0.52.pt \
    --task-name CoRE_PLD \
    --valid-subset train,valid,test \
    --num-workers 0 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --task unimof_ws24 \
    --arch unimof_ws24 \
    --loss cross_entropy_ws24 \
    --weight-by-class \
    --batch-size 1 \
    --seed 1 \
    --remove-hydrogen \
    --results-path ./evaluation_weighted
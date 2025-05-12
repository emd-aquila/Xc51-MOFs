#!/bin/bash

python ./unimat/infer.py . \
    --user-dir ./unimat \
    --path ./logs_mostly_freeze_weighted/\checkpoint.best_f1_0.48.pt \
    --task-name WS24 \
    --valid-subset train,valid,test \
    --num-workers 0 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --task unimof_ws24 \
    --arch unimof_ws24_mostly_freeze \
    --loss cross_entropy_ws24 \
    --batch-size 1 \
    --seed 1 \
    --remove-hydrogen \
    --results-path ./evaluation_mostly_freeze_weighted
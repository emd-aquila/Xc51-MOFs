#!/bin/bash

python ./unimat/infer.py . \
    --user-dir ./unimat \
    --path ./weights/unimof_CoRE_MOFX_DB_finetune_best.pt \
    --task-name MOFX-DB \
    --valid-subset test \
    --num-workers 0 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --task unimof_v2 \
    --arch unimof_v2 \
    --loss mof_v2_mse \
    --batch-size 1 \
    --seed 1 \
    --remove-hydrogen \
    --results-path ./evaluation_MOFX_DB
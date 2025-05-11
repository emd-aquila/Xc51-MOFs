#!/bin/bash

# Set environment variables
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

# Run the training command
nohup python $(which unicore-train) ./ \
  --user-dir ./unimat \
  --task-name CoRE_PLD \
  --train-subset train \
  --valid-subset valid,test \
  --num-workers 0 \
  --ddp-backend c10d \
  --task unimof_ws24 \
  --loss cross_entropy_ws24 \
  --arch unimof_ws24_freeze \
  --weight-by-class \
  --optimizer adam \
  --adam-betas '(0.9, 0.99)' \
  --adam-eps 1e-6 \
  --clip-norm 1.0 \
  --lr-scheduler polynomial_decay \
  --lr 3e-4 \
  --warmup-ratio 0.06 \
  --max-epoch 50 \
  --batch-size 1 \
  --update-freq 8 \
  --seed 1 \
  --num-classes 4 \
  --pooler-dropout 0.2 \
  --finetune-mol-model ./weights/unimof_CoRE_MOFX_DB_finetune_best.pt \
  --log-interval 10 \
  --log-format simple \
  --validate-interval 1 \
  --remove-hydrogen \
  --save-interval 1 \
  --save-interval-updates 0 \
  --keep-interval-updates 0 \
  --keep-last-epochs 1 \
  --keep-best-checkpoints 1 \
  --best-checkpoint-metric f1 \
  --maximize-best-checkpoint-metric \
  --save-dir ./logs_freeze_weighted \
  --max-atoms 500 \
  > "./logs_freeze_weighted/save_finetune_WS24_freeze_weighted.log" &
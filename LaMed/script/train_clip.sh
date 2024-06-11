#!/bin/bash

# run "accelerate config" first!
# 50 epoch / 10h

accelerate launch LaMed/src/train/train_CLIP.py \
    --language_model_name_or_path ./LaMed/pretrained_model/bert_base_uncased \
    --version v0 \
    --local_loss False \
    --gather_loss True \
    --bf16 True \
    --output_dir ./LaMed/output/CLIP-0000 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 16 \
    --report_to tensorboard
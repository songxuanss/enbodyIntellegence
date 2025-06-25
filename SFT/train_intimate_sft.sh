#!/bin/bash

# SFT训练脚本 - 亲密对话模型
# 生成时间: 2025-06-25 10:58:30

echo "🚀 开始SFT训练..."
echo "📊 训练数据: intimate_sft_dataset.json (1000条样本)"
echo "🤖 基础模型: Llama-2-7b-chat"

# 方法1: 使用Transformers直接训练
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29500 \
    train_sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --data_path intimate_sft_dataset.json \
    --bf16 True \
    --output_dir ./intimate-model-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed ds_config_zero2.json

echo "✅ 训练完成!"

# 方法2: 使用LLaMA-Factory训练 (推荐)
echo "🔄 或者使用LLaMA-Factory进行训练:"
echo "llamafactory-cli train examples/train_lora/llama2_lora_sft.yaml"

# 方法3: 使用Unsloth加速训练
echo "⚡ 或者使用Unsloth加速训练:"
echo "python train_unsloth.py"

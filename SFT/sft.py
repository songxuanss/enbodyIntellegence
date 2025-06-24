# SFT训练配置和启动脚本

import json
import os
from datetime import datetime

# training dataset
from sft_adult_doll_eng import all_conversations

# 1. 保存训练数据为JSON文件
def save_training_data():
    """保存1000条训练数据到JSON文件"""
    
    # 这里是完整的1000条训练数据
    training_data = [
        {
            "id": f"intimate_conversation_{i+1:04d}",
            "conversations": [
                {
                    "from": "human", 
                    "value": conversation["input"]
                },
                {
                    "from": "gpt",
                    "value": conversation["output"]
                }
            ]
        }
        for i, conversation in enumerate(all_conversations)  # 使用之前生成的1000条数据
    ]
    
    # 保存到文件
    with open('intimate_sft_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练数据已保存到 intimate_sft_dataset.json")
    print(f"📊 总样本数: {len(training_data)}")
    return 'intimate_sft_dataset.json'

# 2. SFT训练配置
sft_config = {
    "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",  # 基础模型
    "data_path": "intimate_sft_dataset.json",  # 训练数据路径
    "output_dir": "./intimate-model-sft",  # 输出目录
    "num_train_epochs": 3,  # 训练轮数
    "per_device_train_batch_size": 4,  # 批次大小
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,  # 梯度累积
    "learning_rate": 2e-5,  # 学习率
    "max_seq_length": 512,  # 最大序列长度
    "logging_steps": 10,  # 日志记录间隔
    "save_steps": 100,  # 保存间隔
    "eval_steps": 100,  # 评估间隔
    "warmup_steps": 100,  # 预热步数
    "save_total_limit": 3,  # 保存检查点数量限制
    "load_best_model_at_end": True,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "report_to": "tensorboard",  # 使用tensorboard记录
    "remove_unused_columns": False,
    "dataloader_pin_memory": False,
}

# 3. 使用LLaMA-Factory进行SFT训练的配置
llamafactory_config = {
    "stage": "sft",
    "model_name": "llama2_7b_chat",
    "dataset": "intimate_dataset",
    "template": "llama2",
    "finetuning_type": "lora",  # 使用LoRA进行高效微调
    "lora_target": "q_proj,v_proj",
    "output_dir": "intimate_llama2_lora",
    "overwrite_cache": True,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 500,
    "learning_rate": 5e-5,
    "num_train_epochs": 3.0,
    "max_samples": 1000,
    "max_grad_norm": 1.0,
    "quantization_bit": 4,  # 4-bit量化
    "loraplus_lr_ratio": 16.0,
    "use_unsloth": True,  # 使用Unsloth加速
}

# 4. 生成训练脚本
def generate_training_script():
    """生成SFT训练脚本"""
    
    script_content = f'''#!/bin/bash

# SFT训练脚本 - 亲密对话模型
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

echo "🚀 开始SFT训练..."
echo "📊 训练数据: intimate_sft_dataset.json (1000条样本)"
echo "🤖 基础模型: Llama-2-7b-chat"

# 方法1: 使用Transformers直接训练
python -m torch.distributed.launch \\
    --nproc_per_node=1 \\
    --master_port=29500 \\
    train_sft.py \\
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \\
    --data_path intimate_sft_dataset.json \\
    --bf16 True \\
    --output_dir ./intimate-model-sft \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 4 \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --evaluation_strategy "steps" \\
    --eval_steps 100 \\
    --save_strategy "steps" \\
    --save_steps 100 \\
    --save_total_limit 3 \\
    --learning_rate 2e-5 \\
    --weight_decay 0.1 \\
    --warmup_steps 100 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 10 \\
    --report_to "tensorboard" \\
    --gradient_checkpointing True \\
    --deepspeed ds_config_zero2.json

echo "✅ 训练完成!"

# 方法2: 使用LLaMA-Factory训练 (推荐)
echo "🔄 或者使用LLaMA-Factory进行训练:"
echo "llamafactory-cli train examples/train_lora/llama2_lora_sft.yaml"

# 方法3: 使用Unsloth加速训练
echo "⚡ 或者使用Unsloth加速训练:"
echo "python train_unsloth.py"
'''
    
    with open('train_intimate_sft.sh', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod('train_intimate_sft.sh', 0o755)  # 添加执行权限
    print("✅ 训练脚本已生成: train_intimate_sft.sh")

# 5. 生成Python训练脚本
def generate_python_training_script():
    """生成Python训练脚本"""
    
    python_script = '''
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_dataset(file_path):
    """加载训练数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为训练格式
    formatted_data = []
    for item in data:
        conversations = item['conversations']
        human_text = conversations[0]['value']
        gpt_text = conversations[1]['value']
        
        # 格式化为指令格式
        formatted_text = f"Human: {human_text}\\nAssistant: {gpt_text}"
        formatted_data.append({"text": formatted_text})
    
    return Dataset.from_list(formatted_data)

def tokenize_function(examples, tokenizer, max_length=512):
    """数据预处理"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    # 配置
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    dataset_path = "intimate_sft_dataset.json"
    output_dir = "./intimate-model-sft"
    
    # 加载tokenizer和模型
    print("🔄 加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 加载和预处理数据
    print("📊 加载训练数据...")
    dataset = load_dataset(dataset_path)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("🚀 开始SFT训练...")
    trainer.train()
    
    # 保存模型
    print("💾 保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("✅ 训练完成!")
    print(f"📁 模型保存在: {output_dir}")

if __name__ == "__main__":
    main()
'''
    
    with open('train_sft.py', 'w', encoding='utf-8') as f:
        f.write(python_script)
    
    print("✅ Python训练脚本已生成: train_sft.py")

# 6. 生成requirements.txt
def generate_requirements():
    """生成依赖文件"""
    requirements = """
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.6.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
tensorboard
scipy
scikit-learn
deepspeed
flash-attn>=2.3.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("✅ 依赖文件已生成: requirements.txt")

# 7. 生成LLaMA-Factory配置
def generate_llamafactory_config():
    """生成LLaMA-Factory训练配置"""
    
    # 数据集配置
    dataset_config = {
        "intimate_dataset": {
            "file_name": "intimate_sft_dataset.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            }
        }
    }
    
    with open('dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    
    # 训练配置
    train_config = """
### model
model_name: llama2_7b_chat
model_revision: main

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1

### dataset
dataset: intimate_dataset
template: llama2
cutoff_len: 512
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./saves/intimate-llama2-chat-lora
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_steps: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
evaluation_strategy: steps
eval_steps: 500
"""
    
    with open('intimate_sft_config.yaml', 'w') as f:
        f.write(train_config)
    
    print("✅ LLaMA-Factory配置已生成:")
    print("   - dataset_info.json")
    print("   - intimate_sft_config.yaml")

# 8. 主函数 - 执行所有准备工作
def main():
    """执行SFT训练准备"""
    print("🔧 准备SFT训练环境...")
    print("=" * 50)
    
    # 保存训练数据
    dataset_file = save_training_data()
    
    # 生成训练脚本
    generate_training_script()
    generate_python_training_script()
    
    # 生成配置文件
    generate_requirements()
    generate_llamafactory_config()
    
    print("\n" + "=" * 50)
    print("✅ SFT训练准备完成!")
    print("\n📋 生成的文件:")
    print("   📄 intimate_sft_dataset.json - 训练数据 (1000条)")
    print("   🐚 train_intimate_sft.sh - Bash训练脚本")
    print("   🐍 train_sft.py - Python训练脚本")
    print("   📦 requirements.txt - 依赖文件")
    print("   ⚙️  intimate_sft_config.yaml - LLaMA-Factory配置")
    print("   📊 dataset_info.json - 数据集信息")
    
    print("\n🚀 开始训练:")
    print("   方法1 (推荐): llamafactory-cli train intimate_sft_config.yaml")
    print("   方法2: bash train_intimate_sft.sh")
    print("   方法3: python train_sft.py")
    
    print("\n💡 训练建议:")
    print("   - 建议使用GPU进行训练 (至少8GB显存)")
    print("   - 使用LoRA可以减少显存需求")
    print("   - 可以调整batch_size和learning_rate优化效果")
    print("   - 训练过程中监控loss变化")
    
    print("\n🔍 监控训练:")
    print("   tensorboard --logdir ./intimate-model-sft/runs")

if __name__ == "__main__":
    main()

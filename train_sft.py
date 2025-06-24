
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
        formatted_text = f"Human: {human_text}\nAssistant: {gpt_text}"
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

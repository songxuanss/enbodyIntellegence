"""
训练模块 - 使用统一配置进行模型训练
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 设置日志器
logger = logging.getLogger(__name__)

# 尝试导入训练相关库
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForSeq2Seq
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ 成功导入 transformers 和相关依赖")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"⚠️  transformers 未安装: {e}")
    # 创建占位符类避免导入错误
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("transformers 未安装")
    
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("transformers 未安装")

from config import ModelConfig, TrainingConfig, DataConfig
from data_manager import DataManager
from evaluation import ModelEvaluator

class SFTTrainer:
    """监督微调训练器 - 支持LoRA微调"""
    
    def __init__(self, model_config: Optional[ModelConfig] = None, 
                 training_config: Optional[TrainingConfig] = None, 
                 data_config: Optional[DataConfig] = None):
        logger.info("🚀 初始化 SFT 训练器...")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ transformers 库未安装，无法进行训练")
            logger.info("💡 请运行: pip install torch transformers datasets peft")
            raise ImportError("需要安装 transformers 相关库")
        
        # 使用默认配置或传入的配置
        from config import config as default_config
        self.model_config = model_config or default_config.model
        self.training_config = training_config or default_config.training
        self.data_config = data_config or default_config.data
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.data_manager = None
        self.trainer = None
        self.train_dataset = None
        
        # 禁用flash attention
        os.environ["DISABLE_FLASH_ATTENTION"] = "1"
        
        logger.info("✅ SFT 训练器初始化完成")
    
    def setup_model(self):
        """设置模型和分词器"""
        logger.info(f"🔧 开始设置模型: {self.model_config.model_name}")
        
        try:
            # 加载分词器
            logger.info("📝 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,
                trust_remote_code=self.model_config.trust_remote_code,
                use_fast=False
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.debug("🔧 设置 pad_token 为 eos_token")
            
            logger.info("✅ 分词器加载成功")
            
            # 加载模型
            logger.info("🤖 加载模型...")
            model_kwargs = {
                "trust_remote_code": self.model_config.trust_remote_code,
                "device_map": self.model_config.device_map,
            }
            
            # 设置数据类型
            if hasattr(torch, self.model_config.torch_dtype):
                model_kwargs["torch_dtype"] = getattr(torch, self.model_config.torch_dtype)
                logger.debug(f"设置数据类型: {self.model_config.torch_dtype}")
            
            # 设置量化
            if self.model_config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                logger.info("⚡ 启用 8-bit 量化")
            elif self.model_config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                logger.info("⚡ 启用 4-bit 量化")
            
            logger.debug(f"模型参数: {model_kwargs}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name,
                **model_kwargs
            )
            
            logger.info("✅ 模型加载成功")
            
            # 设置LoRA
            if self.model_config.lora_target_modules:
                self.setup_lora()
            
            logger.info("🎉 模型设置完成")
            
        except Exception as e:
            logger.error(f"❌ 模型设置失败: {str(e)}", exc_info=True)
            raise
    
    def setup_lora(self):
        """设置LoRA配置"""
        logger.info("🔧 设置 LoRA 配置...")
        
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                target_modules=self.model_config.lora_target_modules,
                bias="none"
            )
            
            logger.debug(f"LoRA 配置: r={self.model_config.lora_r}, alpha={self.model_config.lora_alpha}")
            logger.debug(f"目标模块: {self.model_config.lora_target_modules}")
            
            self.model = get_peft_model(self.model, lora_config)
            
            # 统计可训练参数
            if hasattr(self.model, 'print_trainable_parameters'):
                logger.info("📊 模型参数统计:")
                self.model.print_trainable_parameters()
            
            logger.info("✅ LoRA 设置完成")
            
        except Exception as e:
            logger.error(f"❌ LoRA 设置失败: {str(e)}", exc_info=True)
            raise
    
    def prepare_data(self):
        """准备训练数据"""
        logger.info("📊 开始准备训练数据...")
        
        try:
            # 初始化数据管理器
            self.data_manager = DataManager(self.data_config)
            
            # 加载数据集
            logger.info("🔄 加载数据集...")
            datasets = self.data_manager.load_datasets()
            
            if not datasets:
                logger.error("❌ 没有加载到任何数据集")
                raise ValueError("数据集为空")
            
            # 合并数据集
            logger.info("🔄 合并数据集...")
            combined_data = self.data_manager.combine_datasets()
            
            if not combined_data:
                logger.error("❌ 合并后的数据集为空")
                raise ValueError("合并数据集失败")
            
            # 转换为训练格式
            logger.info("🔄 转换数据格式...")
            train_data = self.format_data_for_training(combined_data)
            
            if not train_data:
                logger.error("❌ 格式化后的训练数据为空")
                raise ValueError("数据格式化失败")
            
            # 创建Dataset对象
            logger.info("🔄 创建 Dataset 对象...")
            self.train_dataset = Dataset.from_list(train_data)
            
            logger.info(f"✅ 训练数据准备完成: {len(self.train_dataset)} 条样本")
            
            # 显示数据统计
            stats = self.data_manager.get_dataset_stats()
            logger.info("📈 数据集统计:")
            for name, details in stats["dataset_details"].items():
                logger.info(f"   {name}: {details['record_count']} 条记录")
            
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {str(e)}", exc_info=True)
            raise
    
    def format_data_for_training(self, data: List[Dict]) -> List[Dict]:
        """将数据格式化为训练格式"""
        logger.debug("🔄 格式化训练数据...")
        
        if not self.tokenizer:
            logger.error("❌ 分词器未初始化")
            return []
        
        formatted_data = []
        
        for i, item in enumerate(data):
            try:
                if 'conversations' in item:
                    # 构建完整的对话文本
                    conversation_text = ""
                    for turn in item['conversations']:
                        if turn.get('from') == 'human':
                            conversation_text += f"Human: {turn.get('value', '')}\n"
                        elif turn.get('from') == 'gpt':
                            conversation_text += f"Assistant: {turn.get('value', '')}\n"
                    
                    if not conversation_text.strip():
                        logger.warning(f"⚠️  空对话文本，跳过数据项 {i}")
                        continue
                    
                    # 分词和截断
                    tokens = self.tokenizer(
                        conversation_text,
                        truncation=True,
                        max_length=self.data_config.max_length,
                        return_tensors=None,
                        add_special_tokens=True
                    )
                    
                    if len(tokens["input_ids"]) < 10:  # 跳过太短的序列
                        logger.warning(f"⚠️  序列太短，跳过数据项 {i}")
                        continue
                    
                    formatted_data.append({
                        "input_ids": tokens["input_ids"],
                        "attention_mask": tokens["attention_mask"],
                        "labels": tokens["input_ids"].copy()  # 对于因果语言模型，labels和input_ids相同
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️  格式化数据项 {i} 失败: {str(e)}")
                continue
        
        logger.debug(f"✅ 格式化完成: {len(formatted_data)} 条训练样本")
        return formatted_data
    
    def setup_training(self):
        """设置训练参数"""
        logger.info("⚙️  设置训练参数...")
        
        try:
            if not self.model or not self.tokenizer or not self.train_dataset:
                logger.error("❌ 模型、分词器或数据集未准备就绪")
                raise ValueError("训练组件未完全初始化")
            
            # 创建训练参数
            training_args = TrainingArguments(
                output_dir=self.training_config.output_dir,
                num_train_epochs=self.training_config.num_train_epochs,
                per_device_train_batch_size=self.training_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                learning_rate=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                warmup_steps=self.training_config.warmup_steps,
                lr_scheduler_type=self.training_config.lr_scheduler_type,
                save_steps=self.training_config.save_steps,
                eval_steps=self.training_config.eval_steps,
                logging_steps=self.training_config.logging_steps,
                save_total_limit=self.training_config.save_total_limit,
                load_best_model_at_end=self.training_config.load_best_model_at_end,
                fp16=self.training_config.fp16,
                bf16=self.training_config.bf16,
                gradient_checkpointing=self.training_config.gradient_checkpointing,
                dataloader_num_workers=self.training_config.dataloader_num_workers,
                evaluation_strategy=self.training_config.evaluation_strategy,
                save_strategy=self.training_config.save_strategy,
                report_to=self.training_config.report_to,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            
            logger.info("📋 训练参数:")
            logger.info(f"   输出目录: {training_args.output_dir}")
            logger.info(f"   训练轮数: {training_args.num_train_epochs}")
            logger.info(f"   批次大小: {training_args.per_device_train_batch_size}")
            logger.info(f"   学习率: {training_args.learning_rate}")
            logger.info(f"   梯度累积步数: {training_args.gradient_accumulation_steps}")
            
            # 创建数据收集器
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True,
                return_tensors="pt"
            )
            
            # 创建训练器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            logger.info("✅ 训练设置完成")
            
        except Exception as e:
            logger.error(f"❌ 训练设置失败: {str(e)}", exc_info=True)
            raise
    
    def train(self):
        """开始训练"""
        logger.info("🚀 开始训练...")
        
        try:
            if not self.trainer:
                logger.error("❌ 训练器未初始化")
                raise ValueError("训练器未设置")
            
            # 确保输出目录存在
            os.makedirs(self.training_config.output_dir, exist_ok=True)
            
            # 保存训练配置
            config_path = os.path.join(self.training_config.output_dir, "training_config.json")
            self.save_training_config(config_path)
            
            # 开始训练
            logger.info("🎯 启动训练循环...")
            train_result = self.trainer.train()
            
            # 保存训练结果
            logger.info("💾 保存训练结果...")
            self.trainer.save_model()
            self.trainer.save_state()
            
            # 记录训练统计
            logger.info("📊 训练完成统计:")
            logger.info(f"   训练损失: {train_result.training_loss:.4f}")
            logger.info(f"   训练步数: {train_result.global_step}")
            
            # 保存最终模型
            final_model_path = os.path.join(self.training_config.output_dir, "final_model")
            self.trainer.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            logger.info(f"✅ 训练成功完成! 模型保存至: {final_model_path}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {str(e)}", exc_info=True)
            raise
    
    def save_training_config(self, config_path: str):
        """保存训练配置"""
        logger.debug(f"💾 保存训练配置到: {config_path}")
        
        try:
            config_data = {
                "model_config": {
                    "model_name": self.model_config.model_name,
                    "torch_dtype": self.model_config.torch_dtype,
                    "load_in_8bit": self.model_config.load_in_8bit,
                    "lora_r": self.model_config.lora_r,
                    "lora_alpha": self.model_config.lora_alpha,
                    "lora_target_modules": self.model_config.lora_target_modules,
                },
                "training_config": {
                    "num_train_epochs": self.training_config.num_train_epochs,
                    "per_device_train_batch_size": self.training_config.per_device_train_batch_size,
                    "learning_rate": self.training_config.learning_rate,
                    "output_dir": self.training_config.output_dir,
                },
                "data_config": {
                    "active_datasets": self.data_config.active_datasets,
                    "max_length": self.data_config.max_length,
                    "data_format": self.data_config.data_format,
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("✅ 训练配置保存成功")
            
        except Exception as e:
            logger.warning(f"⚠️  保存训练配置失败: {str(e)}")
    
    def run_full_training(self):
        """运行完整的训练流程"""
        logger.info("🎯 开始完整训练流程...")
        
        try:
            # 1. 设置模型
            logger.info("📋 步骤 1/4: 设置模型")
            self.setup_model()
            
            # 2. 准备数据
            logger.info("📋 步骤 2/4: 准备数据")
            self.prepare_data()
            
            # 3. 设置训练
            logger.info("📋 步骤 3/4: 设置训练")
            self.setup_training()
            
            # 4. 开始训练
            logger.info("📋 步骤 4/4: 开始训练")
            result = self.train()
            
            logger.info("🎉 完整训练流程成功完成!")
            return result
            
        except Exception as e:
            logger.error(f"❌ 训练流程失败: {str(e)}", exc_info=True)
            raise

# 便捷函数
def quick_train(model_key: Optional[str] = None, datasets: Optional[List[str]] = None, 
                epochs: int = 3, batch_size: int = 4) -> str:
    """快速训练函数"""
    logger.info("🚀 启动快速训练...")
    
    try:
        from config import config
        
        # 设置配置
        if model_key:
            config.update_model(model_key)
            logger.info(f"🔧 使用模型: {model_key}")
        
        if datasets:
            config.update_datasets(datasets)
            logger.info(f"📊 使用数据集: {datasets}")
        
        # 更新训练参数
        config.training.num_train_epochs = epochs
        config.training.per_device_train_batch_size = batch_size
        
        logger.info(f"⚙️  训练配置: {epochs} 轮, 批次大小 {batch_size}")
        
        # 创建训练器
        trainer = SFTTrainer(
            model_config=config.model,
            training_config=config.training,
            data_config=config.data
        )
        
        # 开始训练
        result = trainer.run_full_training()
        
        logger.info("✅ 快速训练完成!")
        return config.training.output_dir
        
    except Exception as e:
        logger.error(f"❌ 快速训练失败: {str(e)}", exc_info=True)
        return ""

def train_with_config(config_updates: Optional[Dict] = None) -> str:
    """使用自定义配置训练"""
    logger.info("🔧 使用自定义配置训练...")
    
    try:
        from config import config
        
        # 应用配置更新
        if config_updates:
            for section, updates in config_updates.items():
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    for key, value in updates.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                            logger.debug(f"更新配置: {section}.{key} = {value}")
        
        # 创建训练器
        trainer = SFTTrainer(
            model_config=config.model,
            training_config=config.training,
            data_config=config.data
        )
        
        # 开始训练
        result = trainer.run_full_training()
        
        logger.info("✅ 自定义配置训练完成!")
        return config.training.output_dir
        
    except Exception as e:
        logger.error(f"❌ 自定义配置训练失败: {str(e)}", exc_info=True)
        return ""

if __name__ == "__main__":
    # 示例使用
    print("🔧 SFT 训练器示例")
    
    # 快速训练示例
    print("\n🚀 启动快速训练...")
    try:
        result = quick_train(
            model_key="qwen_7b",  # 使用较小的模型进行测试
            datasets=["english_adult", "chinese_shy"],
            epochs=1,
            batch_size=2
        )
        print(f"✅ 训练完成，模型保存在: {result}")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("💡 请确保已安装必要的依赖库") 
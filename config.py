# 项目配置文件 - 统一管理所有配置
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# 配置日志
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置统一的日志配置"""
    # 创建日志格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根日志器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

@dataclass
class ModelConfig:
    """模型配置 - 支持灵活的模型名称配置"""
    # 模型选择 - 可以是任何HuggingFace模型名称或本地路径
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    
    # 常用模型快捷方式（可选，用于快速切换）
    model_shortcuts = {
        "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen_14b": "Qwen/Qwen2.5-14B-Instruct", 
        "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "phi3_14b": "microsoft/Phi-3-medium-4k-instruct",
        "phi3_4b": "microsoft/Phi-3-mini-4k-instruct", 
        "gemma_27b": "google/gemma-2-27b-it",
        "gemma_9b": "google/gemma-2-9b-it",
        "gemma_2b": "google/gemma-2-2b-it",
        "mixtral_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama3.1_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1_70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "llama3.2_1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
        "yi_9b": "01-ai/Yi-1.5-9B-Chat",
        "yi_34b": "01-ai/Yi-1.5-34B-Chat",
        "deepseek_7b": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "deepseek_33b": "deepseek-ai/deepseek-coder-33b-instruct",
        "chatglm_6b": "THUDM/chatglm3-6b",
        "baichuan_7b": "baichuan-inc/Baichuan2-7B-Chat",
        "baichuan_13b": "baichuan-inc/Baichuan2-13B-Chat",
    }
    
    # 模型加载配置
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # 根据模型类型自动设置target_modules
            if "qwen" in self.model_name.lower():
                self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "llama" in self.model_name.lower():
                self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "phi" in self.model_name.lower():
                self.lora_target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
            elif "gemma" in self.model_name.lower():
                self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

@dataclass
class DataConfig:
    """训练数据配置"""
    # 数据源配置 - 在这里统一修改训练数据
    data_sources = {
        "english_adult": "sft_adult_doll_eng.py",
        "chinese_shy": "sft_adult_doll_shy_chn.py", 
        "male_dominant": "sft_adult_doll_having_sex.py",
        "before_sex": "sft_adult_doll_before_sex_chn.py",
    }
    
    # 当前使用的数据源
    active_datasets: Optional[List[str]] = None
    
    # 数据处理配置
    max_length: int = 512
    train_test_split: float = 0.1
    data_format: str = "sharegpt"  # sharegpt, alpaca, chat
    
    # 输出文件配置
    output_file: str = "combined_sft_dataset.json"
    
    def __post_init__(self):
        if self.active_datasets is None:
            self.active_datasets = ["english_adult", "chinese_shy"]  # 默认使用的数据集

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    output_dir: str = "./models/sft_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # 优化器配置
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    
    # 保存和日志配置
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # 训练优化
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # 评估配置
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    report_to: str = "tensorboard"

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估数据集
    eval_datasets = {
        "conversation_quality": "eval_conversation_quality.json",
        "response_appropriateness": "eval_response_appropriateness.json", 
        "safety_check": "eval_safety_check.json",
    }
    
    # 评估指标
    metrics = [
        "bleu",
        "rouge", 
        "bertscore",
        "perplexity",
        "response_length",
        "conversation_coherence"
    ]
    
    # 评估配置
    eval_batch_size: int = 8
    max_eval_samples: int = 100
    
    # 模型比较配置
    baseline_models = [
        "base_model",  # 未训练的基础模型
        "previous_checkpoint",  # 之前的检查点
    ]

@dataclass
class LoggingConfig:
    """日志配置类"""
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: Optional[str] = "logs/training.log"  # 日志文件路径
    console_output: bool = True  # 是否输出到控制台
    file_output: bool = True  # 是否输出到文件
    
    def setup(self):
        """设置日志配置"""
        return setup_logging(
            log_level=self.log_level,
            log_file=self.log_file if self.file_output else None
        )

class ProjectConfig:
    """项目主配置类"""
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig() 
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.logging = LoggingConfig()
        
        # 项目路径配置
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.project_root, "data")
        self.model_dir = os.path.join(self.project_root, "models")
        self.log_dir = os.path.join(self.project_root, "logs")
        self.eval_dir = os.path.join(self.project_root, "evaluation")
        
        # 创建必要的目录
        for dir_path in [self.data_dir, self.model_dir, self.log_dir, self.eval_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def update_model(self, model_identifier: str):
        """更新模型 - 支持快捷方式、完整模型名称或本地路径"""
        if model_identifier in self.model.model_shortcuts:
            # 使用快捷方式
            self.model.model_name = self.model.model_shortcuts[model_identifier]
            print(f"✅ 模型已切换为: {self.model.model_name} (使用快捷方式: {model_identifier})")
        else:
            # 直接使用完整的模型名称或本地路径
            self.model.model_name = model_identifier
            print(f"✅ 模型已设置为: {self.model.model_name}")
            if "/" in model_identifier or "\\" in model_identifier:
                print("💡 检测到可能是本地路径或完整模型名称")
        
        # 重新初始化LoRA配置
        self.model.__post_init__()
    
    def update_datasets(self, dataset_keys: List[str]):
        """快速切换训练数据"""
        valid_keys = []
        for key in dataset_keys:
            if key in self.data.data_sources:
                valid_keys.append(key)
            else:
                print(f"❌ 未找到数据集: {key}")
        
        if valid_keys:
            self.data.active_datasets = valid_keys
            print(f"✅ 训练数据已更新为: {valid_keys}")
        else:
            print(f"可用数据集: {list(self.data.data_sources.keys())}")
    
    def get_model_info(self):
        """获取当前模型信息"""
        return {
            "model_name": self.model.model_name,
            "available_shortcuts": list(self.model.model_shortcuts.keys()),
            "lora_config": {
                "r": self.model.lora_r,
                "alpha": self.model.lora_alpha,
                "dropout": self.model.lora_dropout,
                "target_modules": self.model.lora_target_modules
            },
            "load_config": {
                "torch_dtype": self.model.torch_dtype,
                "load_in_8bit": self.model.load_in_8bit,
                "load_in_4bit": self.model.load_in_4bit,
                "device_map": self.model.device_map
            }
        }
    
    def list_model_shortcuts(self):
        """列出所有可用的模型快捷方式"""
        print("📋 可用的模型快捷方式:")
        print("=" * 60)
        
        # 按类别分组显示
        categories = {
            "Qwen系列": [k for k in self.model.model_shortcuts.keys() if k.startswith("qwen")],
            "LLaMA系列": [k for k in self.model.model_shortcuts.keys() if k.startswith("llama")],
            "Phi系列": [k for k in self.model.model_shortcuts.keys() if k.startswith("phi")],
            "Gemma系列": [k for k in self.model.model_shortcuts.keys() if k.startswith("gemma")],
            "其他模型": [k for k in self.model.model_shortcuts.keys() 
                       if not any(k.startswith(prefix) for prefix in ["qwen", "llama", "phi", "gemma"])]
        }
        
        for category, models in categories.items():
            if models:
                print(f"\n{category}:")
                for model in sorted(models):
                    print(f"  {model:15} -> {self.model.model_shortcuts[model]}")
        
        print("\n💡 使用方法:")
        print("  1. 使用快捷方式: --model qwen_7b")
        print("  2. 使用完整名称: --model Qwen/Qwen2.5-7B-Instruct")
        print("  3. 使用本地路径: --model /path/to/local/model")
        print("  4. 使用HuggingFace用户模型: --model username/model-name")
    
    def validate_model_identifier(self, model_identifier: str) -> dict:
        """验证并分析模型标识符"""
        import os
        
        result = {
            "valid": False,
            "type": None,
            "resolved_name": None,
            "message": ""
        }
        
        # 检查是否是快捷方式
        if model_identifier in self.model.model_shortcuts:
            result.update({
                "valid": True,
                "type": "shortcut",
                "resolved_name": self.model.model_shortcuts[model_identifier],
                "message": f"快捷方式 '{model_identifier}' -> '{self.model.model_shortcuts[model_identifier]}'"
            })
            return result
        
        # 检查是否是本地路径
        if os.path.exists(model_identifier):
            result.update({
                "valid": True,
                "type": "local_path",
                "resolved_name": model_identifier,
                "message": f"本地模型路径: {model_identifier}"
            })
            return result
        
        # 检查是否是合法的HuggingFace模型名称格式
        if "/" in model_identifier and len(model_identifier.split("/")) == 2:
            org, model = model_identifier.split("/")
            if org and model:  # 确保组织名和模型名都不为空
                result.update({
                    "valid": True,
                    "type": "huggingface_repo",
                    "resolved_name": model_identifier,
                    "message": f"HuggingFace模型: {model_identifier}"
                })
                return result
        
        # 如果都不匹配，返回错误信息
        result["message"] = f"无效的模型标识符: {model_identifier}。请使用快捷方式、完整的HuggingFace模型名称或本地路径。"
        return result
    
    def get_model_suggestions(self, partial_name: str) -> List[str]:
        """根据部分名称获取模型建议"""
        suggestions = []
        partial_lower = partial_name.lower()
        
        # 从快捷方式中搜索
        for shortcut in self.model.model_shortcuts.keys():
            if partial_lower in shortcut.lower():
                suggestions.append(shortcut)
        
        return sorted(suggestions)
    
    def get_data_info(self):
        """获取当前数据配置信息"""
        active_datasets = self.data.active_datasets or []
        return {
            "active_datasets": active_datasets,
            "data_sources": {k: v for k, v in self.data.data_sources.items() if k in active_datasets},
            "max_length": self.data.max_length,
            "output_file": self.data.output_file
        }

# 创建全局配置实例
config = ProjectConfig()

# 快速配置函数
def quick_config(model_key: Optional[str] = None, datasets: Optional[List[str]] = None):
    """快速配置模型和数据"""
    if model_key:
        config.update_model(model_key)
    if datasets:
        config.update_datasets(datasets)
    return config

# 额外的配置工具函数
def list_available_models():
    """列出所有可用的模型"""
    config.list_model_shortcuts()

def validate_model(model_name: str):
    """验证模型名称"""
    result = config.validate_model_identifier(model_name)
    if result["valid"]:
        print(f"✅ {result['message']}")
        return True
    else:
        print(f"❌ {result['message']}")
        suggestions = config.get_model_suggestions(model_name)
        if suggestions:
            print(f"💡 相似的快捷方式: {', '.join(suggestions[:5])}")
        return False

# 使用示例
if __name__ == "__main__":
    # 显示当前配置
    print("🔧 当前模型配置:")
    print(config.get_model_info())
    
    print("\n📊 当前数据配置:")
    print(config.get_data_info())
    
    # 列出可用模型
    print("\n" + "="*50)
    list_available_models()
    
    # 验证模型示例
    print("\n" + "="*50)
    print("🧪 模型验证测试:")
    test_models = ["qwen_7b", "invalid_model", "Qwen/Qwen2.5-7B-Instruct", "/local/path"]
    for model in test_models:
        print(f"\n测试: {model}")
        validate_model(model)
    
    # 快速切换示例
    # quick_config(model_key="qwen_14b", datasets=["chinese_shy", "male_dominant"]) 
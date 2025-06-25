# SFT训练系统 - 重构版

一个模块化、易配置的SFT (Supervised Fine-Tuning) 训练系统，支持多种模型和数据集的灵活组合。

## 🏗️ 项目结构

```
enbodyIntellegence/
├── config.py              # 🔧 统一配置文件
├── data_manager.py         # 📊 数据管理模块
├── trainer.py             # 🚀 训练模块
├── evaluation.py          # 📈 评估模块
├── run_training.py        # 🎯 主训练脚本
├── SFT/                   # 📂 训练数据源
│   ├── sft_adult_doll_eng.py          # 英文成人对话
│   ├── sft_adult_doll_shy_chn.py      # 中文害羞对话
│   ├── sft_adult_doll_having_sex.py   # 男性主导对话
│   └── sft_adult_doll_before_sex_chn.py # 事前对话
├── data/                  # 📁 生成的数据集
├── models/                # 🤖 训练输出的模型
├── logs/                  # 📋 训练日志
└── evaluation/            # 📊 评估结果
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install torch transformers datasets peft accelerate bitsandbytes tensorboard
```

### 2. 查看可用配置
```bash
python run_training.py config    # 查看当前配置
python run_training.py datasets  # 查看可用数据集
python run_training.py demo      # 查看配置示例
```

### 3. 开始训练
```bash
python run_training.py           # 使用默认配置训练
```

## 🔧 配置系统

### 模型配置 (config.py)

**1. 快速切换模型**
```python
from config import quick_config

# 使用Qwen2.5-14B模型
quick_config(model_key="qwen_14b")

# 使用Llama3-8B模型  
quick_config(model_key="llama3_8b")

# 使用Phi-3-14B模型
quick_config(model_key="phi3_14b")
```

**2. 可用模型列表**
```python
qwen_72b    # Qwen/Qwen2.5-72B-Instruct      (72B参数)
qwen_14b    # Qwen/Qwen2.5-14B-Instruct      (14B参数)
qwen_7b     # Qwen/Qwen2.5-7B-Instruct       (7B参数)
phi3_14b    # microsoft/Phi-3-medium-4k-instruct
gemma_27b   # google/gemma-2-27b-it
mixtral_8x7b # mistralai/Mixtral-8x7B-Instruct-v0.1
llama3_8b   # meta-llama/Meta-Llama-3-8B-Instruct
llama3_70b  # meta-llama/Meta-Llama-3-70B-Instruct
```

**3. 自定义模型**
```python
from config import config

config.model.model_name = "your/custom-model"
config.model.lora_r = 16
config.model.lora_alpha = 64
```

### 数据配置

**1. 快速切换数据集**
```python
# 使用单个数据集
quick_config(datasets=["chinese_shy"])

# 使用多个数据集
quick_config(datasets=["english_adult", "chinese_shy", "male_dominant"])
```

**2. 可用数据集**
```python
english_adult    # 英文成人对话 (1000条)
chinese_shy      # 中文害羞对话 (1000条)  
male_dominant    # 男性主导对话 (1000条)
before_sex       # 事前对话 (开发中)
```

## 🎯 使用示例

### 示例1: 快速测试 (小模型)
```python
from trainer import quick_train

# 使用7B模型快速测试
model_path = quick_train(
    model_key="qwen_7b",
    datasets=["chinese_shy"],
    num_train_epochs=1,
    per_device_train_batch_size=2
)
```

### 示例2: 生产训练 (大模型)
```python
from trainer import quick_train

# 使用72B模型完整训练
model_path = quick_train(
    model_key="qwen_72b", 
    datasets=["english_adult", "chinese_shy", "male_dominant"],
    num_train_epochs=3,
    learning_rate=5e-5
)
```

### 示例3: 自定义配置训练
```python
from trainer import train_with_config

config_updates = {
    "model": {
        "model_name": "microsoft/Phi-3-medium-4k-instruct",
        "lora_r": 32,
        "load_in_4bit": True
    },
    "training": {
        "num_train_epochs": 5,
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 1
    },
    "data": {
        "active_datasets": ["chinese_shy", "male_dominant"],
        "max_length": 1024
    }
}

model_path = train_with_config(config_updates)
```

## 📊 数据管理

### 创建自定义数据集
```python
from data_manager import create_dataset

# 创建组合数据集
dataset_path = create_dataset(
    datasets=["english_adult", "chinese_shy"],
    format_type="sharegpt"  # sharegpt, alpaca, chat
)
```

### 数据格式支持
- **ShareGPT格式**: 对话格式，适合聊天模型
- **Alpaca格式**: 指令格式，适合指令跟随
- **Chat格式**: 简单问答格式

## 📈 模型评估

```python
from evaluation import evaluate_model

# 评估训练后的模型
results = evaluate_model("./models/sft_output")
print(results)
```

## 💡 最佳实践

### 硬件配置建议

**小模型 (7B-14B)**
- GPU: 16GB+ (单卡)
- 内存: 32GB+
- 存储: 50GB+

**大模型 (27B-72B)**  
- GPU: 40GB+ (多卡更佳)
- 内存: 64GB+
- 存储: 200GB+

### 训练参数调优

**快速测试**
```python
num_train_epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=5e-5
```

**生产训练**
```python
num_train_epochs=3
per_device_train_batch_size=4  
gradient_accumulation_steps=4
learning_rate=2e-5
```

## 🔍 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir ./models/sft_output/runs --port 6006

# 访问 http://localhost:6006
```

## 🛠️ 故障排除

### 常见问题

**1. 模型下载失败**
```python
# 解决方案: 使用镜像或本地模型
config.model.model_name = "本地模型路径"
```

**2. 显存不足**
```python
# 解决方案: 启用量化
config.model.load_in_8bit = True
config.training.per_device_train_batch_size = 1
config.training.gradient_accumulation_steps = 16
```

**3. 数据加载失败**
```python
# 解决方案: 检查数据路径
python run_training.py datasets
```

## 📝 更新日志

### v2.0 (重构版)
- ✅ 模块化设计
- ✅ 统一配置系统
- ✅ 灵活数据管理
- ✅ 简化使用接口
- ✅ 支持多种模型
- ✅ 支持多种数据格式

### v1.0 (原版)
- ✅ 基础SFT训练功能
- ✅ LoRA微调支持 

# SFT 训练项目 - 增强日志系统

这是一个专业的监督微调(SFT)训练项目，具有完善的日志系统，可以清楚地追踪训练过程的每个阶段，并在出错时提供详细的错误信息。

## 🎯 项目特性

### 🚀 核心功能
- **多模型支持**: 支持 Qwen、LLaMA、Phi-3、Gemma 等主流模型
- **LoRA 微调**: 高效的参数微调技术
- **多数据集管理**: 灵活的数据集配置和管理
- **完整训练流程**: 从数据加载到模型评估的完整自动化流程

### 📊 增强日志系统

#### 🔧 日志配置特性
- **多级别日志**: DEBUG、INFO、WARNING、ERROR、CRITICAL
- **双重输出**: 同时支持控制台和文件输出
- **自动文件管理**: 自动创建日志目录和文件
- **时间戳记录**: 精确到秒的时间追踪
- **emoji 标识**: 直观的视觉标识，便于快速定位信息类型

#### 📈 阶段性进度追踪
- **步骤编号**: 清晰的 "步骤 X/Y" 格式
- **进度百分比**: 实时进度显示
- **阶段描述**: 详细的操作描述
- **完成状态**: 明确的完成标识

#### ❌ 错误处理和诊断
- **完整堆栈跟踪**: 详细的错误堆栈信息
- **错误分类**: 按错误类型分类记录
- **恢复建议**: 针对常见错误的解决建议
- **上下文保留**: 错误发生时的环境信息

#### ⚡ 性能监控
- **操作耗时**: 每个操作的精确耗时
- **性能警告**: 耗时较长操作的自动警告
- **总时间统计**: 完整流程的时间统计
- **效率分析**: 性能瓶颈识别

## 📋 使用方法

### 🚀 快速开始

1. **演示模式** - 查看当前配置和可用数据集:
```bash
python run_training.py --mode demo
```

2. **快速训练** - 使用默认配置进行训练:
```bash
python run_training.py --mode quick --model qwen_7b --datasets english_adult
```

3. **自定义训练** - 自定义训练参数:
```bash
python run_training.py --mode custom --epochs 5 --batch-size 8 --learning-rate 1e-5
```

### 📊 数据管理

4. **列出可用数据集**:
```bash
python run_training.py --mode list-data
```

5. **数据加载测试**:
```bash
python -c "from data_manager import DataManager; from config import config; dm = DataManager(config.data); dm.load_datasets()"
```

### 🎯 模型评估

6. **评估已训练模型**:
```bash
python run_training.py --mode evaluate --model-path ./models/sft_output
```

### 🧪 日志功能测试

7. **完整日志功能测试**:
```bash
python test_logging.py
```

## 📁 项目结构

```
enbodyIntellegence/
├── config.py              # 统一配置管理
├── data_manager.py         # 数据加载和处理
├── trainer.py              # 训练器实现
├── evaluation.py           # 模型评估
├── run_training.py         # 主训练脚本
├── test_logging.py         # 日志功能测试
├── logs/                   # 日志文件目录
│   ├── training.log        # 主训练日志
│   └── enhanced_test.log   # 测试日志
├── SFT/                    # 训练数据目录
│   ├── sft_adult_doll_eng.py
│   ├── sft_adult_doll_shy_chn.py
│   └── ...
├── models/                 # 模型输出目录
└── data/                   # 数据文件目录
```

## 🎨 日志输出示例

### 📊 阶段性进度追踪
```
2025-06-25 21:55:04 - root - INFO - 🔧 步骤 1/6: 环境检查
2025-06-25 21:55:04 - root - DEBUG - 🔍 检查CUDA可用性...
2025-06-25 21:55:04 - root - INFO - ✅ 环境检查完成
2025-06-25 21:55:04 - root - INFO - 📊 步骤 2/6: 数据准备
2025-06-25 21:55:04 - root - INFO - 🔄 加载训练数据...
2025-06-25 21:55:04 - root - INFO - ✅ 数据准备完成: 1000 条训练样本
```

### ❌ 错误处理
```
2025-06-25 21:54:59 - root - ERROR - ❌ 文件读取失败: [Errno 2] No such file or directory: 'non_existent_file.txt'
Traceback (most recent call last):
  File "test_logging.py", line 100, in test_error_handling
    with open("non_existent_file.txt", "r") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'non_existent_file.txt'
```

### ⚡ 性能监控
```
2025-06-25 21:55:00 - root - INFO - 🔄 开始执行: 模型初始化
2025-06-25 21:55:00 - root - INFO - ✅ 模型初始化 完成
2025-06-25 21:55:00 - root - INFO - ⏱️  耗时: 1.00秒
2025-06-25 21:55:00 - root - WARNING - ⚠️  模型初始化 执行时间较长: 1.00秒
```

### 📈 数据加载详情
```
2025-06-25 21:55:03 - data_manager - INFO - 🔄 开始加载数据模块: sft_adult_doll_eng.py
2025-06-25 21:55:03 - data_manager - DEBUG - 📂 读取文件: /path/to/sft_adult_doll_eng.py
2025-06-25 21:55:03 - data_manager - DEBUG - ✅ 找到数据变量: sft_training_data
2025-06-25 21:55:03 - data_manager - INFO - ✅ 成功加载 1000 条数据记录
2025-06-25 21:55:03 - data_manager - INFO - ✅ 数据格式验证通过: english_adult
```

## ⚙️ 配置说明

### 🔧 日志配置
```python
# 在 config.py 中配置日志
@dataclass
class LoggingConfig:
    log_level: str = "INFO"                    # 日志级别
    log_file: Optional[str] = "logs/training.log"  # 日志文件路径
    console_output: bool = True                # 控制台输出
    file_output: bool = True                   # 文件输出
```

### 🤖 模型配置
- 支持多种预训练模型
- 自动配置 LoRA 参数
- 灵活的量化选项

### 📊 数据配置
- 多数据集支持
- 自动格式验证
- 统计信息生成

### 🎯 训练配置
- 完整的训练参数控制
- 自动保存和恢复
- 详细的训练记录

## 🔍 故障排除

### 常见问题

1. **transformers 未安装**:
```bash
pip install torch transformers datasets peft
```

2. **数据文件不存在**:
检查 `SFT/` 目录中的数据文件，确保文件路径正确。

3. **内存不足**:
调整 `batch_size` 或启用 `load_in_8bit` 量化。

4. **日志文件无法创建**:
确保有足够的磁盘空间和写入权限。

## 📈 性能优化建议

1. **使用量化**: 启用 8-bit 或 4-bit 量化节省内存
2. **调整批次大小**: 根据GPU内存调整 `batch_size`
3. **梯度累积**: 使用 `gradient_accumulation_steps` 模拟大批次
4. **混合精度**: 启用 `bf16` 或 `fp16` 加速训练

## 🎉 功能亮点

### ✨ 用户友好
- 清晰的emoji标识系统
- 直观的进度显示
- 详细的状态反馈

### 🔧 开发友好
- 完整的错误堆栈跟踪
- 详细的调试信息
- 模块化的设计

### 📊 监控友好
- 实时性能监控
- 详细的统计信息
- 自动化的报告生成

## 📞 支持

如果在使用过程中遇到问题，请查看日志文件中的详细错误信息，大多数问题都有相应的解决建议。

---

🎯 **目标**: 让每一次训练都有迹可循，让每一个错误都有解可依！ 
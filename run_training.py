#!/usr/bin/env python3
"""
主训练脚本 - 支持多种训练模式和配置选项
"""

import os
import sys
import argparse
import logging
from typing import Optional, List, Dict

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import config, LoggingConfig
from trainer import SFTTrainer, quick_train, train_with_config
from data_manager import list_available_datasets, create_dataset
from evaluation import ModelEvaluator

def setup_logging_from_config():
    """从配置设置日志"""
    try:
        # 设置日志配置
        logger = config.logging.setup()
        logger.info("🚀 日志系统初始化完成")
        return logger
    except Exception as e:
        print(f"❌ 日志设置失败: {e}")
        # 使用基本日志配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging_from_config()

def parse_arguments():
    """解析命令行参数"""
    logger.info("📋 解析命令行参数...")
    
    parser = argparse.ArgumentParser(
        description="SFT 训练脚本 - 支持多种训练模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 快速训练
  python run_training.py --mode quick --model qwen_7b --datasets english_adult chinese_shy
  
  # 自定义训练
  python run_training.py --mode custom --epochs 5 --batch-size 8
  
  # 列出可用数据集
  python run_training.py --mode list-data
  
  # 评估模型
  python run_training.py --mode evaluate --model-path ./models/sft_output
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--mode", 
        choices=["quick", "custom", "full", "list-data", "evaluate", "demo"],
        default="demo",
        help="训练模式 (默认: demo)"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        help="模型键名 (如: qwen_7b, llama3_8b)"
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+",
        help="数据集列表 (如: english_adult chinese_shy)"
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="训练轮数 (默认: 3)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="批次大小 (默认: 4)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-5,
        help="学习率 (默认: 2e-5)"
    )
    
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="最大序列长度 (默认: 512)"
    )
    
    # 输出和日志
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="输出目录 (默认使用配置文件设置)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    parser.add_argument(
        "--log-file", 
        type=str,
        help="日志文件路径"
    )
    
    # 评估参数
    parser.add_argument(
        "--model-path", 
        type=str,
        help="评估模型路径"
    )
    
    # 其他选项
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="仅显示配置，不执行训练"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="显示详细信息"
    )
    
    args = parser.parse_args()
    
    logger.info(f"✅ 解析完成，模式: {args.mode}")
    if args.verbose:
        logger.info(f"📋 命令行参数: {vars(args)}")
    
    return args

def update_config_from_args(args):
    """根据命令行参数更新配置"""
    global logger
    logger.info("🔧 根据命令行参数更新配置...")
    
    try:
        # 更新日志配置
        if args.log_level:
            config.logging.log_level = args.log_level
        if args.log_file:
            config.logging.log_file = args.log_file
        
        # 重新设置日志
        if args.log_level or args.log_file:
            logger = config.logging.setup()
            logger.info("🔄 日志配置已更新")
        
        # 更新模型配置
        if args.model:
            config.update_model(args.model)
            logger.info(f"🤖 模型已设置为: {args.model}")
        
        # 更新数据配置
        if args.datasets:
            config.update_datasets(args.datasets)
            logger.info(f"📊 数据集已设置为: {args.datasets}")
        
        # 更新训练配置
        if args.epochs:
            config.training.num_train_epochs = args.epochs
            logger.info(f"🔄 训练轮数: {args.epochs}")
        
        if args.batch_size:
            config.training.per_device_train_batch_size = args.batch_size
            logger.info(f"📦 批次大小: {args.batch_size}")
        
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
            logger.info(f"📈 学习率: {args.learning_rate}")
        
        if args.output_dir:
            config.training.output_dir = args.output_dir
            logger.info(f"📁 输出目录: {args.output_dir}")
        
        # 更新数据配置
        if args.max_length:
            config.data.max_length = args.max_length
            logger.info(f"📏 最大长度: {args.max_length}")
        
        logger.info("✅ 配置更新完成")
        
    except Exception as e:
        logger.error(f"❌ 配置更新失败: {str(e)}", exc_info=True)
        raise

def show_current_config():
    """显示当前配置"""
    logger.info("📋 当前配置信息:")
    
    try:
        # 模型配置
        model_info = config.get_model_info()
        logger.info("🤖 模型配置:")
        logger.info(f"   模型名称: {model_info['model_name']}")
        logger.info(f"   数据类型: {model_info['load_config']['torch_dtype']}")
        logger.info(f"   8位量化: {model_info['load_config']['load_in_8bit']}")
        logger.info(f"   LoRA r: {model_info['lora_config']['r']}")
        logger.info(f"   LoRA alpha: {model_info['lora_config']['alpha']}")
        
        # 数据配置
        data_info = config.get_data_info()
        logger.info("📊 数据配置:")
        logger.info(f"   活跃数据集: {data_info['active_datasets']}")
        logger.info(f"   最大长度: {data_info['max_length']}")
        logger.info(f"   输出文件: {data_info['output_file']}")
        
        # 训练配置
        logger.info("⚙️  训练配置:")
        logger.info(f"   训练轮数: {config.training.num_train_epochs}")
        logger.info(f"   批次大小: {config.training.per_device_train_batch_size}")
        logger.info(f"   学习率: {config.training.learning_rate}")
        logger.info(f"   输出目录: {config.training.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ 显示配置失败: {str(e)}")

def mode_quick_train(args):
    """快速训练模式"""
    logger.info("🚀 启动快速训练模式...")
    
    try:
        if args.dry_run:
            logger.info("🔍 干运行模式 - 仅显示配置")
            show_current_config()
            return True
        
        # 执行快速训练
        result = quick_train(
            model_key=args.model,
            datasets=args.datasets,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        if result:
            logger.info(f"🎉 快速训练成功完成! 模型保存在: {result}")
            return True
        else:
            logger.error("❌ 快速训练失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 快速训练异常: {str(e)}", exc_info=True)
        return False

def mode_custom_train(args):
    """自定义训练模式"""
    logger.info("🔧 启动自定义训练模式...")
    
    try:
        if args.dry_run:
            logger.info("🔍 干运行模式 - 仅显示配置")
            show_current_config()
            return True
        
        # 构建配置更新
        config_updates = {}
        
        if args.epochs or args.batch_size or args.learning_rate or args.output_dir:
            config_updates["training"] = {}
            if args.epochs:
                config_updates["training"]["num_train_epochs"] = args.epochs
            if args.batch_size:
                config_updates["training"]["per_device_train_batch_size"] = args.batch_size
            if args.learning_rate:
                config_updates["training"]["learning_rate"] = args.learning_rate
            if args.output_dir:
                config_updates["training"]["output_dir"] = args.output_dir
        
        if args.max_length:
            config_updates["data"] = {"max_length": args.max_length}
        
        # 执行自定义训练
        result = train_with_config(config_updates)
        
        if result:
            logger.info(f"🎉 自定义训练成功完成! 模型保存在: {result}")
            return True
        else:
            logger.error("❌ 自定义训练失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 自定义训练异常: {str(e)}", exc_info=True)
        return False

def mode_full_train(args):
    """完整训练模式"""
    logger.info("🎯 启动完整训练模式...")
    
    try:
        if args.dry_run:
            logger.info("🔍 干运行模式 - 仅显示配置")
            show_current_config()
            return True
        
        # 创建训练器
        trainer = SFTTrainer(
            model_config=config.model,
            training_config=config.training,
            data_config=config.data
        )
        
        # 运行完整训练流程
        result = trainer.run_full_training()
        
        if result:
            logger.info("🎉 完整训练成功完成!")
            return True
        else:
            logger.error("❌ 完整训练失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 完整训练异常: {str(e)}", exc_info=True)
        return False

def mode_list_data(args):
    """列出数据集模式"""
    logger.info("📋 列出可用数据集...")
    
    try:
        list_available_datasets()
        logger.info("✅ 数据集列表显示完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 列出数据集失败: {str(e)}", exc_info=True)
        return False

def mode_evaluate(args):
    """评估模式"""
    logger.info("📊 启动评估模式...")
    
    try:
        model_path = args.model_path or config.training.output_dir
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"❌ 模型路径不存在: {model_path}")
            return False
        
        logger.info(f"🔍 评估模型: {model_path}")
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(model_path)
        
        logger.info("📊 评估结果:")
        for metric, value in results.items():
            logger.info(f"   {metric}: {value}")
        
        logger.info("✅ 评估完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 评估失败: {str(e)}", exc_info=True)
        return False

def mode_demo(args):
    """演示模式"""
    logger.info("🎪 启动演示模式...")
    
    print("\n" + "="*60)
    print("🎯 SFT 训练系统演示")
    print("="*60)
    
    # 显示当前配置
    print("\n📋 当前配置:")
    show_current_config()
    
    # 显示可用数据集
    print("\n📊 可用数据集:")
    list_available_datasets()
    
    # 显示使用示例
    print("\n💡 使用示例:")
    print("1. 快速训练:")
    print("   python run_training.py --mode quick --model qwen_7b --datasets english_adult")
    
    print("\n2. 自定义训练:")
    print("   python run_training.py --mode custom --epochs 5 --batch-size 8")
    
    print("\n3. 完整训练:")
    print("   python run_training.py --mode full")
    
    print("\n4. 评估模型:")
    print("   python run_training.py --mode evaluate --model-path ./models/sft_output")
    
    print("\n" + "="*60)
    
    logger.info("✅ 演示完成")
    return True

def main():
    """主函数"""
    logger.info("🚀 启动 SFT 训练系统...")
    
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 更新配置
        update_config_from_args(args)
        
        # 根据模式执行相应操作
        success = False
        
        if args.mode == "quick":
            success = mode_quick_train(args)
        elif args.mode == "custom":
            success = mode_custom_train(args)
        elif args.mode == "full":
            success = mode_full_train(args)
        elif args.mode == "list-data":
            success = mode_list_data(args)
        elif args.mode == "evaluate":
            success = mode_evaluate(args)
        elif args.mode == "demo":
            success = mode_demo(args)
        else:
            logger.error(f"❌ 未知模式: {args.mode}")
            success = False
        
        # 返回结果
        if success:
            logger.info("🎉 任务执行成功!")
            sys.exit(0)
        else:
            logger.error("❌ 任务执行失败!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️  用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ 系统异常: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
日志功能全面测试脚本 - 展示增强的日志功能
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import config, LoggingConfig
from data_manager import DataManager
from trainer import SFTTrainer
from evaluation import ModelEvaluator

class LoggingTestSuite:
    """日志功能测试套件"""
    
    def __init__(self):
        self.logger = self.setup_enhanced_logging()
        self.start_time = time.time()
        
    def setup_enhanced_logging(self):
        """设置增强的日志配置"""
        print("🔧 设置增强日志配置...")
        
        # 创建自定义日志配置
        log_config = LoggingConfig(
            log_level="DEBUG",
            log_file="logs/enhanced_test.log",
            console_output=True,
            file_output=True
        )
        
        logger = log_config.setup()
        logger.info("🚀 增强日志系统初始化完成")
        return logger
    
    def test_logging_levels(self):
        """测试不同日志级别"""
        self.logger.info("=" * 60)
        self.logger.info("🧪 测试不同日志级别")
        self.logger.info("=" * 60)
        
        self.logger.debug("🔍 DEBUG: 调试信息 - 数据格式检查")
        self.logger.info("ℹ️  INFO: 常规信息 - 任务开始执行")
        self.logger.warning("⚠️  WARNING: 警告信息 - 配置参数可能需要调整")
        self.logger.error("❌ ERROR: 错误信息 - 文件读取失败")
        self.logger.critical("🚨 CRITICAL: 严重错误 - 系统内存不足")
        
        self.logger.info("✅ 日志级别测试完成")
    
    def test_phase_tracking(self):
        """测试阶段性进度追踪"""
        self.logger.info("=" * 60)
        self.logger.info("📊 测试阶段性进度追踪")
        self.logger.info("=" * 60)
        
        phases = [
            ("初始化环境", "🔧"),
            ("加载配置文件", "📋"),
            ("验证数据源", "🔍"),
            ("准备训练数据", "📊"),
            ("设置模型参数", "🤖"),
            ("开始训练过程", "🚀"),
            ("保存训练结果", "💾"),
            ("清理临时文件", "🧹")
        ]
        
        total_phases = len(phases)
        
        for i, (phase_name, emoji) in enumerate(phases, 1):
            progress = (i / total_phases) * 100
            
            self.logger.info(f"{emoji} 步骤 {i}/{total_phases}: {phase_name}")
            self.logger.debug(f"📈 当前进度: {progress:.1f}%")
            
            # 模拟工作时间
            time.sleep(0.2)
            
            self.logger.info(f"✅ 步骤 {i} 完成: {phase_name}")
        
        self.logger.info("🎉 所有阶段执行完成!")
    
    def test_error_handling(self):
        """测试错误处理和详细错误信息记录"""
        self.logger.info("=" * 60)
        self.logger.info("🔥 测试错误处理和详细错误记录")
        self.logger.info("=" * 60)
        
        # 测试文件不存在错误
        try:
            self.logger.info("🔄 尝试读取不存在的文件...")
            with open("non_existent_file.txt", "r") as f:
                content = f.read()
        except FileNotFoundError as e:
            self.logger.error(f"❌ 文件读取失败: {str(e)}", exc_info=True)
        
        # 测试数据格式错误
        try:
            self.logger.info("🔄 尝试处理错误格式的数据...")
            invalid_data = {"key": "value"}
            result = invalid_data["nonexistent_key"]
        except KeyError as e:
            self.logger.error(f"❌ 数据键值错误: {str(e)}", exc_info=True)
        
        # 测试类型错误
        try:
            self.logger.info("🔄 尝试执行类型不匹配的操作...")
            result = "string" + 123
        except TypeError as e:
            self.logger.error(f"❌ 类型错误: {str(e)}", exc_info=True)
        
        self.logger.info("✅ 错误处理测试完成")
    
    def test_performance_monitoring(self):
        """测试性能监控和时间统计"""
        self.logger.info("=" * 60)
        self.logger.info("⚡ 测试性能监控和时间统计")
        self.logger.info("=" * 60)
        
        # 测试操作耗时记录
        operations = [
            ("数据加载", 0.5),
            ("模型初始化", 1.0),
            ("训练执行", 2.0),
            ("结果保存", 0.3)
        ]
        
        for op_name, duration in operations:
            start_time = time.time()
            self.logger.info(f"🔄 开始执行: {op_name}")
            
            # 模拟操作时间
            time.sleep(duration)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            self.logger.info(f"✅ {op_name} 完成")
            self.logger.info(f"⏱️  耗时: {elapsed:.2f}秒")
            
            # 性能分析
            if elapsed > 1.0:
                self.logger.warning(f"⚠️  {op_name} 执行时间较长: {elapsed:.2f}秒")
            else:
                self.logger.debug(f"🚀 {op_name} 执行效率良好: {elapsed:.2f}秒")
        
        total_elapsed = time.time() - self.start_time
        self.logger.info(f"📊 总执行时间: {total_elapsed:.2f}秒")
    
    def test_data_operations_logging(self):
        """测试数据操作的详细日志记录"""
        self.logger.info("=" * 60)
        self.logger.info("📊 测试数据操作日志记录")
        self.logger.info("=" * 60)
        
        try:
            # 测试数据管理器的详细日志
            self.logger.info("🔄 初始化数据管理器...")
            data_manager = DataManager(config.data)
            
            self.logger.info("🔄 加载数据集...")
            datasets = data_manager.load_datasets()
            
            if datasets:
                self.logger.info(f"✅ 数据加载成功: {len(datasets)} 个数据集")
                
                # 获取统计信息
                stats = data_manager.get_dataset_stats()
                self.logger.info("📊 数据集统计信息:")
                for key, value in stats.items():
                    self.logger.info(f"   {key}: {value}")
            else:
                self.logger.warning("⚠️  没有成功加载任何数据集")
                
        except Exception as e:
            self.logger.error(f"❌ 数据操作测试失败: {str(e)}", exc_info=True)
    
    def test_configuration_logging(self):
        """测试配置相关的日志记录"""
        self.logger.info("=" * 60)
        self.logger.info("⚙️  测试配置日志记录")
        self.logger.info("=" * 60)
        
        # 记录当前配置
        self.logger.info("📋 当前项目配置:")
        
        # 模型配置
        self.logger.info("🤖 模型配置详情:")
        self.logger.info(f"   模型名称: {config.model.model_name}")
        self.logger.info(f"   数据类型: {config.model.torch_dtype}")
        self.logger.info(f"   8位量化: {config.model.load_in_8bit}")
        self.logger.info(f"   LoRA参数: r={config.model.lora_r}, alpha={config.model.lora_alpha}")
        
        # 训练配置
        self.logger.info("🎯 训练配置详情:")
        self.logger.info(f"   训练轮数: {config.training.num_train_epochs}")
        self.logger.info(f"   批次大小: {config.training.per_device_train_batch_size}")
        self.logger.info(f"   学习率: {config.training.learning_rate}")
        self.logger.info(f"   输出目录: {config.training.output_dir}")
        
        # 数据配置
        self.logger.info("📊 数据配置详情:")
        self.logger.info(f"   活跃数据集: {config.data.active_datasets}")
        self.logger.info(f"   最大长度: {config.data.max_length}")
        self.logger.info(f"   输出文件: {config.data.output_file}")
        
        # 日志配置
        self.logger.info("📝 日志配置详情:")
        self.logger.info(f"   日志级别: {config.logging.log_level}")
        self.logger.info(f"   日志文件: {config.logging.log_file}")
        self.logger.info(f"   控制台输出: {config.logging.console_output}")
        
    def test_simulated_training_flow(self):
        """模拟完整的训练流程日志记录"""
        self.logger.info("=" * 60)
        self.logger.info("🎭 模拟完整训练流程")
        self.logger.info("=" * 60)
        
        try:
            # 阶段1: 环境检查
            self.logger.info("🔧 阶段 1/6: 环境检查")
            self.logger.debug("🔍 检查CUDA可用性...")
            self.logger.debug("🔍 检查内存使用情况...")
            self.logger.debug("🔍 验证依赖库版本...")
            self.logger.info("✅ 环境检查完成")
            
            # 阶段2: 数据准备
            self.logger.info("📊 阶段 2/6: 数据准备")
            self.logger.info("🔄 加载训练数据...")
            time.sleep(0.3)
            self.logger.info("🔄 数据格式验证...")
            self.logger.info("🔄 数据预处理...")
            self.logger.info("✅ 数据准备完成: 1000 条训练样本")
            
            # 阶段3: 模型设置
            self.logger.info("🤖 阶段 3/6: 模型设置")
            self.logger.info("🔄 加载基础模型...")
            time.sleep(0.5)
            self.logger.info("🔄 配置LoRA参数...")
            self.logger.info("🔄 设置优化器...")
            self.logger.info("✅ 模型设置完成")
            
            # 阶段4: 训练执行
            self.logger.info("🚀 阶段 4/6: 训练执行")
            for epoch in range(1, 4):
                self.logger.info(f"📈 Epoch {epoch}/3 开始")
                time.sleep(0.2)
                self.logger.info(f"📊 Epoch {epoch} 损失: 2.{3-epoch}45")
                self.logger.info(f"✅ Epoch {epoch}/3 完成")
            self.logger.info("✅ 训练执行完成")
            
            # 阶段5: 模型保存
            self.logger.info("💾 阶段 5/6: 模型保存")
            self.logger.info("🔄 保存模型权重...")
            self.logger.info("🔄 保存训练配置...")
            self.logger.info("🔄 生成模型摘要...")
            self.logger.info("✅ 模型保存完成")
            
            # 阶段6: 评估验证
            self.logger.info("🎯 阶段 6/6: 评估验证")
            self.logger.info("🔄 加载测试数据...")
            self.logger.info("🔄 生成评估报告...")
            self.logger.info("📊 评估指标: BLEU=0.75, ROUGE=0.68")
            self.logger.info("✅ 评估验证完成")
            
            self.logger.info("🎉 完整训练流程模拟成功!")
            
        except Exception as e:
            self.logger.error(f"❌ 训练流程模拟失败: {str(e)}", exc_info=True)
    
    def run_all_tests(self):
        """运行所有日志测试"""
        self.logger.info("🎪 开始日志功能全面测试")
        self.logger.info(f"🕐 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_methods = [
            self.test_logging_levels,
            self.test_phase_tracking,
            self.test_error_handling,
            self.test_performance_monitoring,
            self.test_data_operations_logging,
            self.test_configuration_logging,
            self.test_simulated_training_flow
        ]
        
        total_tests = len(test_methods)
        
        for i, test_method in enumerate(test_methods, 1):
            try:
                self.logger.info(f"🧪 执行测试 {i}/{total_tests}: {test_method.__name__}")
                test_method()
                self.logger.info(f"✅ 测试 {i} 通过")
            except Exception as e:
                self.logger.error(f"❌ 测试 {i} 失败: {str(e)}", exc_info=True)
        
        total_time = time.time() - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("📊 测试总结")
        self.logger.info("=" * 60)
        self.logger.info(f"🕐 总执行时间: {total_time:.2f}秒")
        self.logger.info(f"📈 执行测试数量: {total_tests}")
        self.logger.info("🎉 日志功能全面测试完成!")

def main():
    """主函数"""
    print("🚀 启动日志功能全面测试...")
    print("📋 测试将验证日志系统的各项功能...")
    print()
    
    # 创建测试套件并运行
    test_suite = LoggingTestSuite()
    test_suite.run_all_tests()
    
    print()
    print("✅ 测试完成! 请查看日志文件: logs/enhanced_test.log")

if __name__ == "__main__":
    main() 
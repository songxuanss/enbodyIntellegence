"""
数据管理模块 - 统一管理训练数据
"""

import json
import importlib
import importlib.util
import os
import logging
from typing import List, Dict, Any, Optional
from config import config, DataConfig

# 设置日志器
logger = logging.getLogger(__name__)

class DataManager:
    """数据管理器 - 统一管理训练数据的加载和处理"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.sft_dir = "SFT"
        self.datasets: Dict[str, List[Dict]] = {}
        logger.info("📊 数据管理器初始化完成")
    
    def load_module_data(self, module_name: str) -> List[Dict]:
        """从Python模块文件加载数据"""
        logger.info(f"🔄 开始加载数据模块: {module_name}")
        
        try:
            file_path = os.path.join(self.project_root, self.sft_dir, module_name)
            
            if not os.path.exists(file_path):
                logger.error(f"❌ 数据文件不存在: {file_path}")
                return []
            
            logger.debug(f"📂 读取文件: {file_path}")
            
            # 动态加载模块
            spec = importlib.util.spec_from_file_location("data_module", file_path)
            if spec is None or spec.loader is None:
                logger.error(f"❌ 无法创建模块规范: {file_path}")
                return []
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找数据变量
            data = None
            for attr_name in ['sft_training_data', 'sft_data', 'data', 'dataset', 'conversations', 'all_conversations']:
                if hasattr(module, attr_name):
                    data = getattr(module, attr_name)
                    logger.debug(f"✅ 找到数据变量: {attr_name}")
                    break
            
            if data is None:
                logger.warning(f"⚠️  模块中未找到数据变量: {module_name}")
                return []
            
            if not isinstance(data, list):
                logger.error(f"❌ 数据格式错误，期望列表但得到: {type(data)}")
                return []
            
            # 转换为统一格式
            formatted_data = self.format_conversations(data, module_name)
            
            logger.info(f"✅ 成功加载 {len(formatted_data)} 条数据记录从: {module_name}")
            return formatted_data
            
        except Exception as e:
            logger.error(f"❌ 加载数据模块失败 {module_name}: {str(e)}", exc_info=True)
            return []
    
    def format_conversations(self, data: List[Dict], source_name: str) -> List[Dict]:
        """将数据转换为统一的对话格式"""
        logger.debug(f"🔄 格式化对话数据: {source_name}")
        
        formatted_data = []
        
        for i, item in enumerate(data):
            try:
                if 'conversations' in item:
                    # 已经是标准格式
                    formatted_data.append(item)
                elif 'input' in item and 'output' in item:
                    # input/output格式转换为conversations格式
                    formatted_item = {
                        "conversations": [
                            {"from": "human", "value": item['input']},
                            {"from": "gpt", "value": item['output']}
                        ]
                    }
                    formatted_data.append(formatted_item)
                else:
                    logger.warning(f"⚠️  跳过未知格式的数据项 {i} 在 {source_name}")
                    
            except Exception as e:
                logger.error(f"❌ 格式化数据项 {i} 失败 在 {source_name}: {str(e)}")
        
        logger.debug(f"✅ 格式化完成: {len(formatted_data)} 条记录")
        return formatted_data
    
    def validate_data_format(self, data: List[Dict], source_name: str) -> bool:
        """验证数据格式"""
        logger.debug(f"🔍 验证数据格式: {source_name}")
        
        if not data:
            logger.warning(f"⚠️  数据为空: {source_name}")
            return False
        
        try:
            # 检查必要字段
            sample = data[0]
            if 'conversations' not in sample:
                logger.error(f"❌ 缺少必要字段 'conversations' 在数据源: {source_name}")
                return False
            
            # 检查conversations格式
            if not isinstance(sample['conversations'], list):
                logger.error(f"❌ conversations 字段应为列表: {source_name}")
                return False
            
            if sample['conversations'] and 'from' not in sample['conversations'][0]:
                logger.error(f"❌ conversation 缺少 'from' 字段: {source_name}")
                return False
                
            logger.info(f"✅ 数据格式验证通过: {source_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据格式验证失败 {source_name}: {str(e)}")
            return False
    
    def load_datasets(self) -> Dict[str, List[Dict]]:
        """加载所有指定的数据集"""
        logger.info("🚀 开始加载数据集...")
        
        if not self.config.active_datasets:
            logger.warning("⚠️  没有指定活跃数据集")
            return {}
        
        logger.info(f"📋 活跃数据集: {self.config.active_datasets}")
        
        for dataset_name in self.config.active_datasets:
            try:
                logger.info(f"🔄 处理数据集: {dataset_name}")
                
                if dataset_name not in self.config.data_sources:
                    logger.error(f"❌ 未知数据集: {dataset_name}")
                    continue
                
                module_file = self.config.data_sources[dataset_name]
                logger.debug(f"📄 模块文件: {module_file}")
                
                data = self.load_module_data(module_file)
                
                if data and self.validate_data_format(data, dataset_name):
                    self.datasets[dataset_name] = data
                    logger.info(f"✅ 数据集加载成功: {dataset_name} ({len(data)} 条记录)")
                else:
                    logger.warning(f"⚠️  跳过无效数据集: {dataset_name}")
                    
            except Exception as e:
                logger.error(f"❌ 加载数据集失败 {dataset_name}: {str(e)}", exc_info=True)
        
        total_records = sum(len(data) for data in self.datasets.values())
        logger.info(f"🎉 数据集加载完成! 总计 {len(self.datasets)} 个数据集，{total_records} 条记录")
        
        return self.datasets
    
    def combine_datasets(self) -> List[Dict]:
        """合并所有数据集"""
        logger.info("🔄 开始合并数据集...")
        
        if not self.datasets:
            logger.warning("⚠️  没有已加载的数据集可合并")
            self.load_datasets()
        
        combined_data = []
        
        for dataset_name, data in self.datasets.items():
            logger.debug(f"📊 合并数据集: {dataset_name} ({len(data)} 条记录)")
            combined_data.extend(data)
        
        logger.info(f"✅ 数据集合并完成! 总计 {len(combined_data)} 条记录")
        return combined_data
    
    def save_combined_dataset(self, output_file: Optional[str] = None) -> str:
        """保存合并后的数据集"""
        output_file = output_file or self.config.output_file
        logger.info(f"💾 开始保存合并数据集到: {output_file}")
        
        try:
            combined_data = self.combine_datasets()
            
            if not combined_data:
                logger.error("❌ 没有数据可保存")
                return ""
            
            # 确保目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 保存为JSON格式
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 数据集保存成功: {output_file} ({len(combined_data)} 条记录)")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ 保存数据集失败: {str(e)}", exc_info=True)
            return ""
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        logger.debug("📈 生成数据集统计信息...")
        
        stats = {
            "total_datasets": len(self.datasets),
            "dataset_details": {},
            "total_records": 0,
            "data_sources_used": list(self.datasets.keys())
        }
        
        for name, data in self.datasets.items():
            dataset_stats = {
                "record_count": len(data),
                "avg_conversation_length": 0.0,
                "unique_roles": set()
            }
            
            # 计算平均对话长度和角色统计
            total_conversations = 0
            for record in data:
                if 'conversations' in record:
                    total_conversations += len(record['conversations'])
                    for conv in record['conversations']:
                        if 'from' in conv:
                            dataset_stats["unique_roles"].add(conv['from'])
            
            if data:
                dataset_stats["avg_conversation_length"] = total_conversations / len(data)
            
            dataset_stats["unique_roles"] = list(dataset_stats["unique_roles"])
            stats["dataset_details"][name] = dataset_stats
            stats["total_records"] += len(data)
        
        logger.info(f"📊 统计信息生成完成: {stats['total_records']} 条总记录")
        return stats
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        logger.debug("📋 获取数据集信息...")
        
        info = {
            "data_sources": self.config.data_sources,
            "active_datasets": self.config.active_datasets or [],
            "available_files": {}
        }
        
        # 检查文件存在状态
        for key, filename in self.config.data_sources.items():
            file_path = os.path.join(self.sft_dir, filename)
            info["available_files"][key] = {
                "filename": filename,
                "exists": os.path.exists(file_path),
                "path": file_path
            }
        
        return info

# 便捷函数
def quick_load_data(datasets: Optional[List[str]] = None, config: Optional[DataConfig] = None) -> Optional["DataManager"]:
    """快速加载数据的便捷函数"""
    logger.info("🚀 启动快速数据加载...")
    
    try:
        if config is None:
            from config import config as default_config
            config = default_config.data
        
        if datasets:
            config.active_datasets = datasets
            logger.info(f"📋 使用指定数据集: {datasets}")
        
        manager = DataManager(config)
        datasets_loaded = manager.load_datasets()
        
        if datasets_loaded:
            logger.info("✅ 快速数据加载成功!")
            return manager
        else:
            logger.error("❌ 快速数据加载失败!")
            return None
            
    except Exception as e:
        logger.error(f"❌ 快速数据加载异常: {str(e)}", exc_info=True)
        return None

def create_dataset(datasets: Optional[List[str]] = None, format_type: Optional[str] = None) -> str:
    """快速创建数据集"""
    logger.info("🚀 开始创建数据集...")
    
    try:
        from config import config
        manager = DataManager(config.data)
        
        if datasets:
            manager.config.active_datasets = datasets
        
        output_file = manager.save_combined_dataset()
        
        if output_file:
            logger.info(f"✅ 数据集创建成功: {output_file}")
        else:
            logger.error("❌ 数据集创建失败")
            
        return output_file
        
    except Exception as e:
        logger.error(f"❌ 创建数据集异常: {str(e)}", exc_info=True)
        return ""

def list_available_datasets():
    """列出可用的数据集"""
    logger.info("📋 列出可用数据集...")
    
    try:
        from config import config
        manager = DataManager(config.data)
        info = manager.get_dataset_info()
        
        print("\n📊 可用数据集:")
        print(f"活跃数据集: {info['active_datasets']}")
        print("\n文件状态:")
        
        for key, file_info in info["available_files"].items():
            status = "✅" if file_info["exists"] else "❌"
            print(f"   {key}: {file_info['filename']} {status}")
            
        logger.info("✅ 数据集列表显示完成")
        
    except Exception as e:
        logger.error(f"❌ 列出数据集失败: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # 示例使用
    print("🔧 数据管理器示例")
    list_available_datasets()
    
    # 创建示例数据集
    print("\n🚀 创建示例数据集...")
    output = create_dataset(["english_adult", "chinese_shy"])
    if output:
        print(f"✅ 数据集已保存到: {output}") 
#!/usr/bin/env python3
"""
模型配置增强演示 - 展示灵活的模型配置功能
"""

import os
import sys
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import config

def list_available_models():
    """列出所有可用的模型"""
    print("📋 可用的模型快捷方式:")
    print("=" * 60)
    
    # 按类别分组显示
    shortcuts = config.model.model_shortcuts
    categories = {
        "Qwen系列": [k for k in shortcuts.keys() if k.startswith("qwen")],
        "LLaMA系列": [k for k in shortcuts.keys() if k.startswith("llama")],
        "Phi系列": [k for k in shortcuts.keys() if k.startswith("phi")],
        "Gemma系列": [k for k in shortcuts.keys() if k.startswith("gemma")],
        "其他模型": [k for k in shortcuts.keys() 
                   if not any(k.startswith(prefix) for prefix in ["qwen", "llama", "phi", "gemma"])]
    }
    
    for category, models in categories.items():
        if models:
            print(f"\n{category}:")
            for model in sorted(models):
                print(f"  {model:15} -> {shortcuts[model]}")
    
    print("\n💡 使用方法:")
    print("  1. 使用快捷方式: --model qwen_7b")
    print("  2. 使用完整名称: --model Qwen/Qwen2.5-7B-Instruct")
    print("  3. 使用本地路径: --model /path/to/local/model")
    print("  4. 使用HuggingFace用户模型: --model username/model-name")

def validate_model_identifier(model_identifier: str) -> dict:
    """验证并分析模型标识符"""
    result = {
        "valid": False,
        "type": None,
        "resolved_name": None,
        "message": ""
    }
    
    # 检查是否是快捷方式
    if model_identifier in config.model.model_shortcuts:
        result.update({
            "valid": True,
            "type": "shortcut",
            "resolved_name": config.model.model_shortcuts[model_identifier],
            "message": f"快捷方式 '{model_identifier}' -> '{config.model.model_shortcuts[model_identifier]}'"
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

def get_model_suggestions(partial_name: str) -> List[str]:
    """根据部分名称获取模型建议"""
    suggestions = []
    partial_lower = partial_name.lower()
    
    # 从快捷方式中搜索
    for shortcut in config.model.model_shortcuts.keys():
        if partial_lower in shortcut.lower():
            suggestions.append(shortcut)
    
    return sorted(suggestions)

def validate_model(model_name: str):
    """验证模型名称"""
    result = validate_model_identifier(model_name)
    if result["valid"]:
        print(f"✅ {result['message']}")
        return True
    else:
        print(f"❌ {result['message']}")
        suggestions = get_model_suggestions(model_name)
        if suggestions:
            print(f"💡 相似的快捷方式: {', '.join(suggestions[:5])}")
        return False

def enhanced_update_model(model_identifier: str):
    """增强的模型更新功能"""
    print(f"\n🔄 准备设置模型: {model_identifier}")
    
    # 验证模型标识符
    result = validate_model_identifier(model_identifier)
    
    if not result["valid"]:
        print(f"❌ {result['message']}")
        suggestions = get_model_suggestions(model_identifier)
        if suggestions:
            print(f"💡 相似的快捷方式: {', '.join(suggestions[:3])}")
        return False
    
    # 更新模型配置
    old_model = config.model.model_name
    config.model.model_name = result["resolved_name"]
    
    # 重新初始化LoRA配置
    config.model.__post_init__()
    
    print(f"✅ 模型已从 '{old_model}' 更新为 '{config.model.model_name}'")
    print(f"🔧 模型类型: {result['type']}")
    print(f"📋 LoRA目标模块: {config.model.lora_target_modules}")
    
    return True

def demo_flexible_model_config():
    """演示灵活的模型配置功能"""
    print("🎯 模型配置灵活性演示")
    print("=" * 60)
    
    print("\n📋 当前配置:")
    print(f"当前模型: {config.model.model_name}")
    print(f"LoRA目标模块: {config.model.lora_target_modules}")
    
    # 测试不同类型的模型标识符
    test_cases = [
        "qwen_7b",  # 快捷方式
        "Qwen/Qwen2.5-3B-Instruct",  # 完整HuggingFace名称
        "microsoft/Phi-3-mini-4k-instruct",  # 另一个完整名称
        "invalid_model",  # 无效模型
        "qwen",  # 部分匹配
        "./local_model",  # 本地路径（不存在）
    ]
    
    for i, model_id in enumerate(test_cases, 1):
        print(f"\n🧪 测试 {i}/{len(test_cases)}: {model_id}")
        print("-" * 40)
        enhanced_update_model(model_id)
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")

def main():
    """主函数"""
    print("🚀 启动模型配置增强演示...")
    print()
    
    # 列出可用模型
    list_available_models()
    
    print("\n" + "=" * 60)
    
    # 演示灵活配置
    demo_flexible_model_config()
    
    print(f"\n💡 总结:")
    print("现在支持三种模型配置方式:")
    print("1. 快捷方式 - 方便快速切换常用模型")
    print("2. 完整名称 - 支持任何HuggingFace模型")
    print("3. 本地路径 - 支持本地训练或微调的模型")
    print("\n这使得模型配置更加灵活，不再局限于预定义的键值对！")

if __name__ == "__main__":
    main() 
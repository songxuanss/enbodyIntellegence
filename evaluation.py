"""
模型评估模块 - 统一的模型性能评估
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from config import config

# 设置日志器
logger = logging.getLogger(__name__)

# 尝试导入评估相关库
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    EVAL_LIBRARIES_AVAILABLE = True
    logger.info("✅ 成功导入评估相关依赖")
except ImportError as e:
    EVAL_LIBRARIES_AVAILABLE = False
    logger.warning(f"⚠️  评估库未安装: {e}")

from config import EvaluationConfig

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    model_path: str
    metrics: Dict[str, float]
    sample_outputs: List[Dict[str, str]]
    evaluation_time: float
    error_message: Optional[str] = None

class ModelEvaluator:
    """模型评估器 - 支持多种评估指标"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        logger.info("🔧 初始化模型评估器...")
        
        if not EVAL_LIBRARIES_AVAILABLE:
            logger.warning("⚠️  评估库未完全安装，某些功能可能不可用")
        
        # 使用默认配置或传入的配置
        from config import config as default_config
        self.config = config or default_config.evaluation
        
        # 初始化组件
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        
        logger.info("✅ 模型评估器初始化完成")
    
    def load_model(self, model_path: str):
        """加载要评估的模型"""
        logger.info(f"🔄 加载评估模型: {model_path}")
        
        try:
            if not EVAL_LIBRARIES_AVAILABLE:
                logger.error("❌ 评估库未安装，无法加载模型")
                raise ImportError("需要安装 torch 和 transformers")
            
            if not os.path.exists(model_path):
                logger.error(f"❌ 模型路径不存在: {model_path}")
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            # 加载分词器
            logger.info("📝 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.debug("🔧 设置 pad_token 为 eos_token")
            
            # 加载模型
            logger.info("🤖 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.current_model_path = model_path
            logger.info("✅ 模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}", exc_info=True)
            raise
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """生成模型响应"""
        logger.debug(f"🔄 生成响应，提示长度: {len(prompt)}")
        
        try:
            if not self.model or not self.tokenizer:
                logger.error("❌ 模型或分词器未加载")
                raise ValueError("模型未初始化")
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length // 2  # 留一半长度给生成
            )
            
            # 移动到正确的设备
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取新生成的部分
            response = generated_text[len(prompt):].strip()
            
            logger.debug(f"✅ 生成完成，响应长度: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"❌ 生成响应失败: {str(e)}")
            return f"[生成失败: {str(e)}]"
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """计算困惑度"""
        logger.info("📊 计算困惑度...")
        
        try:
            if not self.model or not self.tokenizer:
                logger.error("❌ 模型或分词器未加载")
                return float('inf')
            
            total_loss = 0.0
            total_tokens = 0
            
            for i, text in enumerate(texts[:self.config.max_eval_samples]):
                logger.debug(f"🔄 处理文本 {i+1}/{len(texts)}")
                
                try:
                    # 编码文本
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    
                    # 移动到正确的设备
                    if hasattr(self.model, 'device'):
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # 计算损失
                    with torch.no_grad():
                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                    
                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)
                    
                except Exception as e:
                    logger.warning(f"⚠️  跳过文本 {i}: {str(e)}")
                    continue
            
            if total_tokens == 0:
                logger.warning("⚠️  没有有效的文本用于计算困惑度")
                return float('inf')
            
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            logger.info(f"✅ 困惑度计算完成: {perplexity:.4f}")
            return perplexity
            
        except Exception as e:
            logger.error(f"❌ 困惑度计算失败: {str(e)}", exc_info=True)
            return float('inf')
    
    def calculate_response_quality(self, prompts: List[str]) -> Dict[str, float]:
        """评估响应质量"""
        logger.info("📊 评估响应质量...")
        
        try:
            if not prompts:
                logger.warning("⚠️  没有提供评估提示")
                return {"avg_response_length": 0.0, "response_rate": 0.0}
            
            responses = []
            valid_responses = 0
            total_length = 0
            
            for i, prompt in enumerate(prompts[:self.config.max_eval_samples]):
                logger.debug(f"🔄 评估提示 {i+1}/{len(prompts)}")
                
                response = self.generate_response(prompt)
                responses.append(response)
                
                if response and not response.startswith("[生成失败"):
                    valid_responses += 1
                    total_length += len(response)
            
            # 计算指标
            response_rate = valid_responses / len(prompts) if prompts else 0.0
            avg_length = total_length / valid_responses if valid_responses > 0 else 0.0
            
            metrics = {
                "avg_response_length": avg_length,
                "response_rate": response_rate,
                "total_prompts": len(prompts),
                "valid_responses": valid_responses
            }
            
            logger.info(f"✅ 响应质量评估完成: 响应率 {response_rate:.2%}, 平均长度 {avg_length:.1f}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ 响应质量评估失败: {str(e)}", exc_info=True)
            return {"avg_response_length": 0.0, "response_rate": 0.0}
    
    def run_conversation_test(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """运行对话测试"""
        logger.info(f"🔄 运行对话测试，测试用例: {len(test_cases)}")
        
        try:
            results = []
            
            for i, case in enumerate(test_cases[:self.config.max_eval_samples]):
                logger.debug(f"🔄 测试用例 {i+1}/{len(test_cases)}")
                
                try:
                    prompt = case.get("input", "")
                    expected = case.get("expected", "")
                    
                    if not prompt:
                        logger.warning(f"⚠️  跳过空提示的测试用例 {i}")
                        continue
                    
                    # 生成响应
                    response = self.generate_response(f"Human: {prompt}\nAssistant:")
                    
                    # 简单的相似度评分（基于长度和关键词）
                    similarity_score = self.calculate_simple_similarity(response, expected) if expected else 0.0
                    
                    result = {
                        "case_id": i,
                        "input": prompt,
                        "expected": expected,
                        "generated": response,
                        "similarity_score": similarity_score,
                        "response_length": len(response)
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"⚠️  测试用例 {i} 失败: {str(e)}")
                    continue
            
            logger.info(f"✅ 对话测试完成，处理了 {len(results)} 个测试用例")
            return results
            
        except Exception as e:
            logger.error(f"❌ 对话测试失败: {str(e)}", exc_info=True)
            return []
    
    def calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """计算简单的文本相似度"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # 转换为小写并分词
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # 计算Jaccard相似度
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0.0
            return similarity
            
        except Exception as e:
            logger.warning(f"⚠️  相似度计算失败: {str(e)}")
            return 0.0
    
    def create_test_cases(self) -> List[Dict[str, str]]:
        """创建默认测试用例"""
        logger.debug("🔧 创建默认测试用例...")
        
        test_cases = [
            {
                "input": "你好，请介绍一下自己",
                "expected": "",
                "category": "greeting"
            },
            {
                "input": "请解释一下人工智能的概念",
                "expected": "",
                "category": "knowledge"
            },
            {
                "input": "如何学习编程？",
                "expected": "",
                "category": "advice"
            },
            {
                "input": "今天天气怎么样？",
                "expected": "",
                "category": "casual"
            },
            {
                "input": "请写一个简单的Python函数",
                "expected": "",
                "category": "coding"
            }
        ]
        
        logger.debug(f"✅ 创建了 {len(test_cases)} 个测试用例")
        return test_cases
    
    def evaluate_model(self, model_path: str, custom_test_cases: Optional[List[Dict]] = None) -> EvaluationResult:
        """评估模型的完整流程"""
        logger.info(f"🎯 开始评估模型: {model_path}")
        
        import time
        start_time = time.time()
        
        try:
            # 1. 加载模型
            logger.info("📋 步骤 1/4: 加载模型")
            self.load_model(model_path)
            
            # 2. 准备测试用例
            logger.info("📋 步骤 2/4: 准备测试用例")
            test_cases = custom_test_cases or self.create_test_cases()
            
            # 3. 运行评估
            logger.info("📋 步骤 3/4: 运行评估")
            
            # 基本指标
            metrics = {}
            
            # 响应质量评估
            prompts = [case["input"] for case in test_cases]
            quality_metrics = self.calculate_response_quality(prompts)
            metrics.update(quality_metrics)
            
            # 对话测试
            conversation_results = self.run_conversation_test(test_cases)
            
            # 计算平均相似度
            similarities = [r["similarity_score"] for r in conversation_results if r.get("similarity_score") is not None]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            metrics["avg_similarity"] = avg_similarity
            
            # 困惑度计算（使用生成的响应）
            generated_texts = [r["generated"] for r in conversation_results if r.get("generated")]
            if generated_texts:
                perplexity = self.calculate_perplexity(generated_texts)
                metrics["perplexity"] = perplexity
            
            # 4. 整理结果
            logger.info("📋 步骤 4/4: 整理结果")
            
            evaluation_time = time.time() - start_time
            
            result = EvaluationResult(
                model_path=model_path,
                metrics=metrics,
                sample_outputs=conversation_results[:5],  # 保存前5个样本
                evaluation_time=evaluation_time
            )
            
            logger.info(f"🎉 模型评估完成! 耗时: {evaluation_time:.2f}秒")
            logger.info("📊 评估指标:")
            for metric, value in metrics.items():
                logger.info(f"   {metric}: {value}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 模型评估失败: {str(e)}", exc_info=True)
            
            evaluation_time = time.time() - start_time
            return EvaluationResult(
                model_path=model_path,
                metrics={},
                sample_outputs=[],
                evaluation_time=evaluation_time,
                error_message=str(e)
            )
    
    def save_evaluation_result(self, result: EvaluationResult, output_path: str):
        """保存评估结果"""
        logger.info(f"💾 保存评估结果到: {output_path}")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 准备保存数据
            save_data = {
                "model_path": result.model_path,
                "evaluation_time": result.evaluation_time,
                "metrics": result.metrics,
                "sample_outputs": result.sample_outputs,
                "error_message": result.error_message,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存为JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ 评估结果保存成功")
            
        except Exception as e:
            logger.error(f"❌ 保存评估结果失败: {str(e)}", exc_info=True)

# 便捷函数
def quick_evaluate(model_path: str, output_path: Optional[str] = None) -> EvaluationResult:
    """快速评估函数"""
    logger.info(f"🚀 启动快速评估: {model_path}")
    
    try:
        evaluator = ModelEvaluator()
        result = evaluator.evaluate_model(model_path)
        
        # 保存结果
        if output_path:
            evaluator.save_evaluation_result(result, output_path)
        
        logger.info("✅ 快速评估完成")
        return result
        
    except Exception as e:
        logger.error(f"❌ 快速评估失败: {str(e)}", exc_info=True)
        return EvaluationResult(
            model_path=model_path,
            metrics={},
            sample_outputs=[],
            evaluation_time=0.0,
            error_message=str(e)
        )

def compare_models(model_paths: List[str]) -> Dict[str, EvaluationResult]:
    """比较多个模型"""
    logger.info(f"🔄 比较 {len(model_paths)} 个模型...")
    
    results = {}
    
    for i, path in enumerate(model_paths):
        logger.info(f"📊 评估模型 {i+1}/{len(model_paths)}: {path}")
        
        try:
            result = quick_evaluate(path)
            results[path] = result
            
        except Exception as e:
            logger.error(f"❌ 模型 {path} 评估失败: {str(e)}")
            results[path] = EvaluationResult(
                model_path=path,
                metrics={},
                sample_outputs=[],
                evaluation_time=0.0,
                error_message=str(e)
            )
    
    logger.info("✅ 模型比较完成")
    return results

if __name__ == "__main__":
    # 示例使用
    print("🔧 模型评估器示例")
    
    # 模拟评估
    print("\n📊 创建示例评估...")
    evaluator = ModelEvaluator()
    
    # 创建测试用例
    test_cases = evaluator.create_test_cases()
    print(f"✅ 创建了 {len(test_cases)} 个测试用例")
    
    # 显示测试用例
    for i, case in enumerate(test_cases):
        print(f"   {i+1}. {case['input']} ({case['category']})")
    
    print("\n💡 使用示例:")
    print("from evaluation import quick_evaluate")
    print("result = quick_evaluate('./models/sft_output')")
    print("print(result.metrics)") 
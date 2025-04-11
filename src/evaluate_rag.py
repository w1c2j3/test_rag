"""
使用Ragas和MS MARCO数据集评估RAG系统的综合脚本
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AspectCritic
from src.custom_api_client import LangchainCustomLLMWrapper
import json
from typing import List, Dict, Any, Optional, Union

def create_metrics(metric_names: List[str]) -> List[AspectCritic]:
    """
    创建评估指标
    
    参数:
        metric_names: 指标名称列表
        
    返回:
        指标对象列表
    """
    metrics = []
    
    metrics_definitions = {
        "accuracy": "评估回答的准确性，验证回答是否与上下文内容一致，且不包含虚假信息。",
        "completeness": "评估回答的完整性，验证回答是否涵盖了上下文中的所有关键信息。",
        "relevance": "评估回答与问题的相关性，验证回答是否直接解答了用户的问题。",
        "coherence": "评估回答的连贯性，验证回答是否逻辑清晰且结构良好。",
        "conciseness": "评估回答的简洁性，验证回答是否简明扼要，不包含冗余信息。"
    }
    
    for name in metric_names:
        if name in metrics_definitions:
            metrics.append(AspectCritic(
                name=name,
                definition=metrics_definitions[name]
            ))
        else:
            print(f"警告: 未知的评估指标 '{name}'，已忽略")
    
    return metrics

def evaluate_dataset(
    dataset_path: str, 
    sample_count: int, 
    metrics: List[AspectCritic], 
    output_file: Optional[str] = None, 
    json_output: Optional[str] = None, 
    batch_size: int = 50, 
    visualize: bool = False,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    api_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    评估数据集
    
    参数:
        dataset_path: 数据集路径
        sample_count: 要评估的样本数量，0表示全部
        metrics: 评估指标列表
        output_file: 评估结果输出文件路径，不指定则只输出到控制台
        json_output: 评估结果JSON格式输出文件路径，便于可视化
        batch_size: 批处理大小，避免一次性评估太多样本
        visualize: 评估完成后是否直接生成可视化结果
        api_key: API密钥，直接从main.py传递
        api_base_url: API基础URL，直接从main.py传递
        api_model: 模型名称，直接从main.py传递
        
    返回:
        评估结果字典
    """
    # 设置输出目标
    original_stdout = sys.stdout
    file_out = None
    
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        file_out = open(output_file, 'w', encoding='utf-8')
    
    def log(message, to_console=True):
        """同时输出到文件和控制台"""
        if file_out:
            print(message, file=file_out, flush=True)
        if to_console:
            print(message, flush=True)
    
    try:
        # 输出评估信息头部
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"=========== RAG系统评估报告 ===========")
        log(f"评估时间: {current_time}")
        log(f"数据集路径: {dataset_path}")
        log(f"评估指标: {', '.join([m.name for m in metrics])}")
        log("=======================================\n")
        
        # 加载数据集
        if os.path.exists(dataset_path):
            log(f"加载数据集: {dataset_path}")
            dataset = Dataset.load_from_disk(dataset_path)
        else:
            log(f"错误: 数据集路径不存在: {dataset_path}")
            return {}
        
        # 检查和处理样本数量
        total_samples = len(dataset)
        if sample_count <= 0 or sample_count > total_samples:
            sample_count = total_samples
            log(f"将评估全部 {total_samples} 个样本")
        else:
            log(f"数据集总样本数: {total_samples}")
            log(f"将评估 {sample_count} 个样本")
        
        # 创建LLM客户端，直接使用传入的API参数
        llm_wrapper = LangchainCustomLLMWrapper(
            api_key=api_key,
            api_url=api_base_url,
            model_name=api_model
        )
        log(f"使用API: {llm_wrapper.client.api_url}")
        log(f"使用模型: {llm_wrapper.client.model_name}")
        
        # 批量评估，避免一次性处理太多样本
        all_results = {}
        
        # 计时
        start_time = time.time()
        
        # 分批处理
        for i in range(0, sample_count, batch_size):
            batch_end = min(i + batch_size, sample_count)
            progress_msg = f"\n正在评估批次 {i//batch_size + 1}/{(sample_count+batch_size-1)//batch_size}: 样本 {i+1}-{batch_end}/{sample_count}"
            log(progress_msg)
            
            # 准备评估数据集
            test_dataset = dataset.select(range(i, batch_end))
            
            # 运行评估
            batch_start_time = time.time()
            try:
                # 评估前输出信息
                log(f"开始评估批次 {i//batch_size + 1}...", to_console=True)
                
                result = evaluate(
                    dataset=test_dataset,
                    metrics=metrics,
                    llm=llm_wrapper
                )
                
                # 提取结果
                if hasattr(result, '_scores_dict'):
                    scores_dict = result._scores_dict
                    for metric_name, scores in scores_dict.items():
                        if metric_name not in all_results:
                            all_results[metric_name] = []
                        all_results[metric_name].extend(scores)
                    
                    # 输出当前批次结果
                    log(f"\n批次 {i//batch_size + 1} 评估结果:", to_console=False)
                    for metric_name, scores in scores_dict.items():
                        log(f"{metric_name}: {scores}", to_console=False)
                    
                    # 在控制台输出简要结果
                    for metric_name, scores in scores_dict.items():
                        if scores:
                            valid_scores = [x for x in scores if x is not None and not np.isnan(x)]
                            if valid_scores:
                                avg = np.mean(valid_scores)
                                log(f"批次 {i//batch_size + 1} {metric_name} 平均分: {avg:.4f}")
                else:
                    log("无法获取结果分数字典")
                    
            except Exception as e:
                log(f"批次 {i//batch_size + 1} 评估过程中出现错误: {e}")
                if file_out:
                    import traceback
                    traceback.print_exc(file=file_out)
                log("继续评估下一批次...")
                continue
            
            batch_time = time.time() - batch_start_time
            log(f"批次 {i//batch_size + 1} 耗时: {batch_time:.2f} 秒")
            
            # 输出总进度
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + batch_size)) * (sample_count - (i + batch_size)) if i + batch_size < sample_count else 0
            log(f"总进度: {min((i + batch_size), sample_count)}/{sample_count} 样本, 已用时: {elapsed:.2f}秒, 预计剩余: {remaining:.2f}秒")
        
        # 评估完成，输出统计结果
        total_time = time.time() - start_time
        log("\n=========== 评估完成 ===========")
        log(f"总耗时: {total_time:.2f} 秒")
        log(f"评估样本数: {sample_count}")
        log(f"每样本平均耗时: {total_time/sample_count:.2f} 秒")
        log("=================================\n")
        
        # 处理汇总结果
        summary_results = {}
        if all_results:
            log("\n========== 汇总结果 ==========")
            log("\n原始分数:", to_console=False)
            for metric_name, scores in all_results.items():
                log(f"{metric_name}: {scores}", to_console=False)
            
            # 计算和输出平均分
            log("\n平均分:")
            for metric_name, scores in all_results.items():
                valid_scores = [x for x in scores if x is not None and not np.isnan(x)]
                if valid_scores:
                    avg = np.mean(valid_scores)
                    summary_results[metric_name] = avg
                    log(f"{metric_name}: {avg:.4f}")
                else:
                    log(f"{metric_name}: 无有效分数")
            log("=================================\n")
        
        # 保存JSON结果
        if json_output:
            log(f"\n保存JSON格式评估结果到: {json_output}")
            full_results = {
                "timestamp": current_time,
                "dataset_path": dataset_path,
                "total_samples": total_samples,
                "evaluated_samples": sample_count,
                "metrics": [m.name for m in metrics],
                "summary": summary_results,
                "details": all_results,
                "api_info": {
                    "api_url": llm_wrapper.client.api_url,
                    "model": llm_wrapper.client.model_name
                }
            }
            try:
                os.makedirs(os.path.dirname(os.path.abspath(json_output)), exist_ok=True)
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(full_results, f, ensure_ascii=False, indent=2)
                log("JSON结果文件保存成功。")
            except Exception as e:
                log(f"保存JSON结果文件时出错: {e}")
        
        # 如果需要可视化
        if visualize:
            log("\n正在生成可视化结果...")
            try:
                from src.visualize import create_visualization
                if json_output:
                    vis_file = json_output.replace(".json", "_viz.html")
                    create_visualization(json_output, vis_file)
                    log(f"可视化结果已保存到: {vis_file}")
                else:
                    log("无JSON结果文件，无法生成可视化。")
            except Exception as e:
                log(f"生成可视化过程中出错: {e}")
        
        return summary_results
    
    except Exception as e:
        log(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc(file=file_out)
        return {}
    
    finally:
        # 关闭文件输出
        if file_out:
            file_out.close()

# 当直接运行此脚本时
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="RAG系统评估工具")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--output", type=str, default="", help="评估结果输出文件路径")
    parser.add_argument("--json-output", type=str, default="", help="评估结果JSON格式输出文件路径")
    parser.add_argument("--samples", type=int, default=0, help="要评估的样本数量，0表示全部")
    parser.add_argument("--batch-size", type=int, default=5, help="批处理大小")
    parser.add_argument("--metrics", type=str, default="accuracy,completeness,relevance", help="评估指标，用逗号分隔")
    parser.add_argument("--visualize", action="store_true", help="评估完成后直接生成可视化结果")
    parser.add_argument("--api-key", type=str, required=True, help="API密钥")
    parser.add_argument("--api-url", type=str, default="https://api.ppai.pro/v1/chat/completions", help="API基础URL")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="模型名称")
    
    args = parser.parse_args()
    
    # 创建评估指标
    metric_names = args.metrics.split(",")
    metrics = create_metrics(metric_names)
    
    # 运行评估
    evaluate_dataset(
        dataset_path=args.dataset,
        sample_count=args.samples,
        metrics=metrics,
        output_file=args.output,
        json_output=args.json_output,
        batch_size=args.batch_size,
        visualize=args.visualize,
        api_key=args.api_key,
        api_base_url=args.api_url,
        api_model=args.model
    ) 
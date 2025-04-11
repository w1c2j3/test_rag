import os
import argparse
from datasets import load_dataset
import numpy as np
from random import random
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_evaluation_metrics(count, seed=None):
    """
    生成评估指标（在实际应用中应替换为真实计算结果）
    
    Args:
        count (int): 需要生成的指标数量
        seed (int, optional): 随机种子
        
    Returns:
        list: 生成的随机评估指标
    """
    if seed is not None:
        np.random.seed(seed)
    return [round(random(), 2) for _ in range(count)]

def load_data(dataset_name, subset=None, trust_remote_code=True):
    """
    加载数据集
    
    Args:
        dataset_name (str): 数据集名称
        subset (str, optional): 数据集子集
        trust_remote_code (bool): 是否信任远程代码
        
    Returns:
        Dataset: 加载的数据集
    """
    logger.info(f"正在加载数据集: {dataset_name}" + (f" ({subset})" if subset else ""))
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, trust_remote_code=trust_remote_code)
        else:
            dataset = load_dataset(dataset_name, trust_remote_code=trust_remote_code)
        return dataset
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

def prepare_evaluation_data(dataset, split="eval", sample_size=None, include_metrics=True):
    """
    准备评估数据
    
    Args:
        dataset: 数据集对象
        split (str): 数据集分割
        sample_size (int, optional): 样本大小
        include_metrics (bool): 是否包含生成的评估指标
        
    Returns:
        dict: 准备好的评估批次结果
    """
    # 获取指定分割的数据
    if split not in dataset:
        available_splits = list(dataset.keys())
        logger.warning(f"指定的分割 '{split}' 不存在, 可用分割: {available_splits}")
        split = available_splits[0]
        logger.info(f"使用分割: {split}")
    
    eval_data = dataset[split]
    
    # 确定样本大小
    if sample_size is None or sample_size > len(eval_data):
        sample_size = len(eval_data)
    
    logger.info(f"数据集大小: {len(eval_data)}, 使用样本数: {sample_size}")
    logger.info(f"数据集字段: {eval_data.column_names}")
    
    # 映射数据集字段到标准字段名
    field_mapping = {
        'question': ['question', 'query', 'user_input', 'input'],
        'context': ['context', 'contexts', 'retrieved_contexts', 'documents'],
        'answer': ['answer', 'response', 'output', 'generated_text'],
        'ground_truth': ['ground_truth', 'reference', 'target', 'expected']
    }
    
    # 准备评估批次结果
    batch_results = {}
    
    # 处理每个标准字段
    for standard_field, possible_names in field_mapping.items():
        found = False
        for field_name in possible_names:
            if field_name in eval_data.column_names:
                if standard_field == 'context':
                    # 处理上下文字段，确保是列表格式
                    batch_results['retrieved_contexts'] = [
                        ctx if isinstance(ctx, list) else [ctx] 
                        for ctx in eval_data[field_name][:sample_size]
                    ]
                elif standard_field == 'ground_truth':
                    # 处理参考答案字段
                    batch_results['reference'] = [
                        gt if not isinstance(gt, list) else gt[0] if len(gt) > 0 else "" 
                        for gt in eval_data[field_name][:sample_size]
                    ]
                elif standard_field == 'question':
                    batch_results['user_input'] = eval_data[field_name][:sample_size]
                elif standard_field == 'answer':
                    batch_results['response'] = eval_data[field_name][:sample_size]
                
                found = True
                break
        
        if not found:
            logger.warning(f"未找到字段: {standard_field}, 可能的字段名: {possible_names}")
    
    # 添加评估指标
    if include_metrics:
        metrics = [
            'context_precision',
            'answer_relevancy',
            'faithfulness',
            'context_recall',
            'context_entities_recall',
            'accuracy',
            'completeness',
            'relevance'
        ]
        
        for metric in metrics:
            batch_results[metric] = generate_evaluation_metrics(sample_size)
    
    return batch_results

def save_evaluation_results(batch_results, output_file):
    """
    保存评估结果到文件
    
    Args:
        batch_results (dict): 评估批次结果
        output_file (str): 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"批次 1 评估结果:\n")
        for key, value in batch_results.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"评估结果已保存到: {output_file}")

def process_dataset(dataset_name, subset=None, split="eval", sample_size=100, output_file=None):
    """
    处理数据集并生成评估结果
    
    Args:
        dataset_name (str): 数据集名称
        subset (str, optional): 数据集子集
        split (str): 数据集分割
        sample_size (int): 样本大小
        output_file (str, optional): 输出文件路径
        
    Returns:
        str: 输出文件路径
    """
    # 加载数据集
    dataset = load_data(dataset_name, subset)
    
    # 准备评估数据
    batch_results = prepare_evaluation_data(dataset, split, sample_size)
    
    # 确定输出文件路径
    if output_file is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_dir = os.path.join(base_dir, "evaluation_results")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        
        dataset_basename = dataset_name.split('/')[-1]
        if subset:
            output_file = os.path.join(eval_dir, f"{dataset_basename}_{subset}_eval_results.txt")
        else:
            output_file = os.path.join(eval_dir, f"{dataset_basename}_eval_results.txt")
    
    # 保存评估结果
    save_evaluation_results(batch_results, output_file)
    
    return output_file

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='准备RAG评估数据')
    parser.add_argument('--dataset', default='explodinggradients/amnesty_qa', help='数据集名称')
    parser.add_argument('--subset', default='english_v2', help='数据集子集')
    parser.add_argument('--split', default='eval', help='数据集分割')
    parser.add_argument('--sample_size', type=int, default=100, help='样本大小')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--no-metrics', action='store_true', help='不生成评估指标')
    args = parser.parse_args()
    
    try:
        output_file = process_dataset(
            args.dataset, 
            args.subset, 
            args.split, 
            args.sample_size,
            args.output
        )
        logger.info(f"处理完成，结果保存在: {output_file}")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
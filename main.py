#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG评估工具包主程序
提供一站式的RAG系统评估功能
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
from pathlib import Path

# 导入自定义模块
from src.test_api import ApiTester
from src.evaluate_rag import create_metrics, evaluate_dataset
from src.dataset_manager import ensure_dataset, get_default_dataset_path, DEFAULT_DATASETS

# ======== API 配置 ========
# API密钥配置，用户需要在此处填入自己的API密钥
API_KEY = "sk-BooxqYCwuiRGLmUFE756C40dB9674dAcAf7fB76900E1F62a"  # 用户提供的API密钥
API_BASE_URL = "https://api.ppai.pro/v1/chat/completions"  # API基础URL
API_MODEL = "deepseek-v3"  # 使用的模型名称

# ======== 目录检查函数 ========
def ensure_directories():
    """
    确保必要的目录存在
    
    返回:
        bool: 是否成功创建所有目录
    """
    directories = [
        "data",
        "data/mldr",
        "data/msmarco_for_ragas",
        "data/export",
        "evaluation_results"
    ]
    
    print("\n" + "=" * 50)
    print("检查目录结构...")
    
    all_created = True
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ 目录 '{directory}' 已确认存在")
        except Exception as e:
            print(f"✗ 无法创建目录 '{directory}': {e}")
            all_created = False
    
    if all_created:
        print("所有必要的目录结构已就绪。")
    else:
        print("警告: 部分目录创建失败，可能影响程序运行。")
    
    print("=" * 50 + "\n")
    return all_created

# ======== API测试函数 ========
def test_api_connection(verbose=True):
    """
    测试API连接是否正常
    
    参数:
        verbose: 是否输出详细信息
        
    返回:
        bool: API连接是否正常
    """
    if verbose:
        print("\n" + "=" * 50)
        print("测试API连接...")
        print("=" * 50)
    
    # 使用ApiTester类进行测试，直接传入API信息而不使用环境变量
    tester = ApiTester(api_key=API_KEY, api_base_url=API_BASE_URL, api_model=API_MODEL)
    
    # 只测试普通API调用，简化流程
    result = tester.test_normal_call()
    
    if result:
        if verbose:
            print("API连接测试通过!")
        return True
    else:
        if verbose:
            print("\n" + "=" * 50)
            print("API连接测试失败!")
            print("=" * 50)
            print("可能的原因:")
            print("1. API密钥错误:")
            print(f"   - 检查main.py中的API密钥是否正确 (当前设置: {API_KEY[:8]}...)")
            print("2. API服务不可用:")
            print(f"   - 检查API服务状态 (当前设置: {API_BASE_URL})")
            print("3. 网络问题:")
            print("   - 检查网络连接")
            print("   - 确认是否需要设置代理")
            print("4. API调用限制:")
            print("   - 您可能已达到API调用限制")
            print("   - 等待一段时间后重试")
            print("=" * 50 + "\n")
        return False

def download_and_prepare_dataset(dataset_type="mldr", force_download=False):
    """
    下载并准备评估所需的数据集
    
    参数:
        dataset_type: 数据集类型，支持"mldr"和"msmarco"
        force_download: 是否强制重新下载
        
    返回:
        数据集路径
    """
    print("\n" + "=" * 50)
    print(f"开始准备{dataset_type}数据集...")
    print("=" * 50)
    
    try:
        # 使用dataset_manager中的函数确保数据集存在
        dataset_path = ensure_dataset(dataset_type, force_download)
        print(f"数据集准备完成: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"数据集准备失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ======== 命令行解析 ========
def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="RAG系统评估工具 - 一站式评估解决方案",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集相关参数
    data_group = parser.add_argument_group('数据集选项')
    data_group.add_argument("--dataset", type=str, default="", 
                         help="评估数据集的路径，留空则自动检测或下载")
    data_group.add_argument("--dataset-type", type=str, choices=["mldr", "msmarco"], default="mldr", 
                         help="要使用的数据集类型（当--dataset为空时生效）")
    data_group.add_argument("--samples", type=int, default=5, 
                         help="要评估的样本数量，0表示评估全部")
    data_group.add_argument("--force-download", action="store_true", 
                         help="强制重新下载数据集")
    
    # 评估相关参数
    eval_group = parser.add_argument_group('评估选项')
    eval_group.add_argument("--batch-size", type=int, default=5, 
                          help="批处理大小，避免一次性评估太多样本")
    eval_group.add_argument("--metrics", type=str, default="accuracy,completeness,relevance", 
                          help="要使用的评估指标，用逗号分隔")
    
    # 输出相关参数
    output_group = parser.add_argument_group('输出选项')
    output_group.add_argument("--output", type=str, default="", 
                           help="评估结果输出文件路径，不指定则使用时间戳生成")
    output_group.add_argument("--json-output", type=str, default="", 
                           help="评估结果JSON格式输出文件路径")
    output_group.add_argument("--visualize", action="store_true", 
                           help="评估完成后直接生成可视化结果")
    
    # 其他选项
    other_group = parser.add_argument_group('其他选项')
    other_group.add_argument("--test-only", action="store_true", 
                          help="仅测试API连接，不执行评估")
    other_group.add_argument("--skip-api-test", action="store_true", 
                          help="跳过API测试，直接进行数据集处理和评估")
    other_group.add_argument("--prepare-datasets", action="store_true", 
                          help="下载并准备所有支持的数据集")
    other_group.add_argument("--verbose", action="store_true", 
                          help="输出详细的日志信息")
    
    return parser.parse_args()

# ======== 主函数 ========
def main():
    """主函数 - 整个评估流程的入口点"""
    # 获取命令行参数
    args = parse_args()
    
    # 打印欢迎信息
    print("=" * 60)
    print(f"{'RAG评估工具包 - 主程序':^58}")
    print("=" * 60)
    print(f"API基础URL: {API_BASE_URL}")
    print(f"使用模型: {API_MODEL}")
    print(f"API密钥: {API_KEY[:8]}...")
    print("-" * 60)
    
    # 确保必要的目录存在
    if not ensure_directories():
        print("错误: 无法创建必要的目录结构，程序终止。")
        return
    
    # 测试API连接
    api_test_pass = True
    if not args.skip_api_test:
        api_test_pass = test_api_connection()
        if not api_test_pass:
            print("错误: API连接测试失败，请检查API密钥和网络连接。")
            return
    
    # 如果只是测试API，则直接返回
    if args.test_only:
        print("API测试完成，程序终止。")
        return
    
    # 处理数据集路径
    dataset_path = args.dataset
    if not dataset_path:
        # 如果没有指定数据集路径，自动下载并准备数据集
        dataset_path = download_and_prepare_dataset(args.dataset_type, args.force_download)
        if not dataset_path:
            print("错误: 数据集准备失败，程序终止。")
            return
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 指定的数据集路径不存在: {dataset_path}")
        print("提示: 您可以使用 --dataset-type 参数指定要下载的数据集类型。")
        return
    
    # 准备输出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置评估结果输出文件
    if not args.output:
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/rag_eval_{args.dataset_type}_{timestamp}.txt"
    else:
        output_file = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 设置JSON输出文件
    if not args.json_output:
        json_dir = "evaluation_results"
        os.makedirs(json_dir, exist_ok=True)
        json_output = f"{json_dir}/rag_eval_{args.dataset_type}_{timestamp}.json"
    else:
        json_output = args.json_output
        os.makedirs(os.path.dirname(os.path.abspath(json_output)), exist_ok=True)
    
    # 执行评估
    try:
        print("\n开始执行RAG评估...")
        # 创建评估指标
        metric_objects = create_metrics(args.metrics.split(","))
        
        # 使用evaluate_rag模块进行评估
        evaluate_dataset(
            dataset_path=dataset_path,
            sample_count=args.samples,
            metrics=metric_objects,
            output_file=output_file,
            json_output=json_output,
            batch_size=args.batch_size,
            visualize=args.visualize,
            api_key=API_KEY,
            api_base_url=API_BASE_URL,
            api_model=API_MODEL
        )
        print(f"评估完成! 结果已保存到: {output_file}")
        print(f"JSON格式评估结果已保存到: {json_output}")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序运行时出现未处理的异常: {e}")
        import traceback
        traceback.print_exc() 
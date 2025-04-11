#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一的数据集管理模块
负责检测、下载和转换RAG评估所需的数据集
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
from typing import Dict, Tuple, List, Optional, Union, Any

# 导入现有模块中的下载和转换函数
from src.download_mldr import download_mldr
from src.convert_mldr import convert_mldr_to_ragas
from src.download_dataset import download_dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_DATASETS = {
    "mldr": {
        "language": "zh",
        "splits": ["test", "corpus"],
        "input_dir": "data/mldr",
        "output_dir": "data/mldr_for_ragas",
        "export_dir": "data/export",
        "split_to_use": "test",
        "samples": 10
    },
    "msmarco": {
        "dataset_name": "msmarco",
        "output_dir": "data/msmarco_for_ragas",
        "export_dir": "data/export"
    }
}

def check_directory_exists(path: str) -> bool:
    """检查目录是否存在"""
    return os.path.isdir(path)

def check_dataset_exists(dataset_type: str = "mldr") -> Tuple[bool, bool, Dict[str, Any]]:
    """
    检查指定类型的数据集是否已下载
    
    参数:
        dataset_type: 数据集类型，支持"mldr"和"msmarco"
        
    返回:
        原始数据是否存在, 处理后数据是否存在, 详细状态信息
    """
    status_info = {
        "raw_data_exists": False,
        "processed_data_exists": False,
        "paths": {},
        "details": {}
    }
    
    if dataset_type == "mldr":
        # 检查MLDR原始数据和转换后的数据
        raw_data_path = DEFAULT_DATASETS["mldr"]["input_dir"]
        processed_data_path = DEFAULT_DATASETS["mldr"]["output_dir"]
        
        # 检查具体文件
        language = DEFAULT_DATASETS["mldr"]["language"]
        test_path = os.path.join(raw_data_path, f"{language}-test")
        corpus_path = os.path.join(raw_data_path, f"{language}-corpus")
        
        raw_exists = check_directory_exists(test_path) and check_directory_exists(corpus_path)
        processed_exists = check_directory_exists(processed_data_path)
        
        # 添加详细信息
        status_info["raw_data_exists"] = raw_exists
        status_info["processed_data_exists"] = processed_exists
        status_info["paths"] = {
            "raw_test": test_path,
            "raw_corpus": corpus_path,
            "processed": processed_data_path,
            "export": DEFAULT_DATASETS["mldr"]["export_dir"]
        }
        status_info["details"] = {
            "language": language,
            "test_exists": check_directory_exists(test_path),
            "corpus_exists": check_directory_exists(corpus_path)
        }
        
        return raw_exists, processed_exists, status_info
    
    elif dataset_type == "msmarco":
        # 检查MS MARCO数据
        data_path = DEFAULT_DATASETS["msmarco"]["output_dir"]
        export_dir = DEFAULT_DATASETS["msmarco"]["export_dir"]
        
        processed_exists = check_directory_exists(data_path)
        
        # 添加详细信息
        status_info["raw_data_exists"] = processed_exists  # 对于msmarco，我们直接下载处理好的数据
        status_info["processed_data_exists"] = processed_exists
        status_info["paths"] = {
            "processed": data_path,
            "export": export_dir
        }
        
        return processed_exists, processed_exists, status_info
    
    # 不支持的数据集类型
    logger.warning(f"不支持的数据集类型: {dataset_type}")
    return False, False, status_info

def ensure_dataset(dataset_type: str = "mldr", force_download: bool = False) -> str:
    """
    确保数据集存在，不存在则下载并处理
    
    参数:
        dataset_type: 数据集类型，支持"mldr"和"msmarco"
        force_download: 是否强制重新下载
        
    返回:
        数据集路径
    """
    # 检查数据集状态
    raw_exists, processed_exists, status_info = check_dataset_exists(dataset_type)
    
    # 1. 如果强制下载或原始数据不存在，则下载数据集
    if force_download or not raw_exists:
        if dataset_type == "mldr":
            logger.info("正在下载MLDR数据集...")
            
            # 获取配置
            config = DEFAULT_DATASETS["mldr"]
            
            # 下载数据集
            download_mldr(
                language=config["language"],
                output_dir=config["input_dir"],
                splits=config["splits"]
            )
            logger.info(f"MLDR数据集下载完成，保存在 {config['input_dir']}")
        
        elif dataset_type == "msmarco":
            logger.info("正在下载MS MARCO数据集...")
            
            # 获取配置
            config = DEFAULT_DATASETS["msmarco"]
            
            # 下载数据集
            download_dataset(
                dataset=config["dataset_name"],
                output_dir=config["output_dir"]
            )
            logger.info(f"MS MARCO数据集下载完成，保存在 {config['output_dir']}")
    
    # 2. 如果原始数据存在但处理后的数据不存在(或强制下载)，则处理数据
    raw_exists, processed_exists, status_info = check_dataset_exists(dataset_type)
    if (raw_exists or force_download) and (not processed_exists or force_download):
        if dataset_type == "mldr":
            logger.info("正在处理MLDR数据集...")
            
            # 获取配置
            config = DEFAULT_DATASETS["mldr"]
            
            # 转换数据集
            convert_mldr_to_ragas(
                input_dir=config["input_dir"],
                output_dir=config["output_dir"],
                language=config["language"],
                split=config["split_to_use"],
                samples=config["samples"]
            )
            logger.info(f"MLDR数据集处理完成，处理后数据保存在 {config['output_dir']}")
    
    # 3. 导出为JSON格式
    export_dataset_to_json(dataset_type)
    
    # 再次检查，确认数据集已准备好
    raw_exists, processed_exists, status_info = check_dataset_exists(dataset_type)
    if not processed_exists:
        raise FileNotFoundError(f"无法找到或准备{dataset_type}数据集，请检查数据路径和下载过程")
    
    # 返回数据集路径
    if dataset_type == "mldr":
        return DEFAULT_DATASETS["mldr"]["output_dir"]
    elif dataset_type == "msmarco":
        return DEFAULT_DATASETS["msmarco"]["output_dir"]
    
    raise ValueError(f"不支持的数据集类型: {dataset_type}")

def export_dataset_to_json(dataset_type: str = "mldr") -> Optional[str]:
    """
    将数据集导出为JSON格式
    
    参数:
        dataset_type: 数据集类型，支持"mldr"和"msmarco"
        
    返回:
        导出的JSON文件路径，或None（如果出错）
    """
    try:
        export_path = None
        
        if dataset_type == "mldr":
            # 获取配置
            config = DEFAULT_DATASETS["mldr"]
            
            # 加载数据集
            input_dir = config["output_dir"]
            export_dir = config["export_dir"]
            
            # 确保导出目录存在
            os.makedirs(export_dir, exist_ok=True)
            
            # 导出文件路径
            export_path = os.path.join(export_dir, "mldr_samples.json")
            
            # 检查数据集是否存在
            if not os.path.exists(input_dir):
                logger.warning(f"MLDR处理后数据集不存在: {input_dir}")
                return None
            
            # 加载数据集
            try:
                dataset = load_from_disk(input_dir)
                
                # 转换为DataFrame再导出为JSON
                df = dataset.to_pandas()
                df.to_json(export_path, orient="records", force_ascii=False, indent=2)
                
                logger.info(f"已将MLDR数据集导出为JSON: {export_path}")
                return export_path
            except Exception as e:
                logger.error(f"加载或导出MLDR数据集时出错: {e}")
                return None
            
        elif dataset_type == "msmarco":
            # 获取配置
            config = DEFAULT_DATASETS["msmarco"]
            
            # 加载数据集
            input_dir = config["output_dir"]
            export_dir = config["export_dir"]
            
            # 确保导出目录存在
            os.makedirs(export_dir, exist_ok=True)
            
            # 导出文件路径
            export_path = os.path.join(export_dir, "msmarco_samples.json")
            
            # 检查数据集是否存在
            if not os.path.exists(input_dir):
                logger.warning(f"MS MARCO数据集不存在: {input_dir}")
                return None
            
            # 加载数据集
            try:
                dataset = load_from_disk(input_dir)
                
                # 转换为DataFrame再导出为JSON
                df = dataset.to_pandas()
                df.to_json(export_path, orient="records", force_ascii=False, indent=2)
                
                logger.info(f"已将MS MARCO数据集导出为JSON: {export_path}")
                return export_path
            except Exception as e:
                logger.error(f"加载或导出MS MARCO数据集时出错: {e}")
                return None
    
    except Exception as e:
        logger.error(f"导出数据集时出错: {e}")
        return None
    
    logger.warning(f"不支持的数据集类型: {dataset_type}")
    return None

def get_default_dataset_path(dataset_type="mldr") -> str:
    """
    获取默认数据集路径
    
    参数:
        dataset_type: 数据集类型，支持"mldr"和"msmarco"
        
    返回:
        str: 数据集路径
    """
    if dataset_type not in DEFAULT_DATASETS:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    if dataset_type == "mldr":
        return DEFAULT_DATASETS["mldr"]["output_dir"]
    elif dataset_type == "msmarco":
        return DEFAULT_DATASETS["msmarco"]["output_dir"]
    
    raise ValueError(f"不支持的数据集类型: {dataset_type}")

def ensure_all_datasets(force_download: bool = False) -> Dict[str, str]:
    """
    确保所有支持的数据集都已下载并处理
    
    参数:
        force_download: 是否强制重新下载
        
    返回:
        数据集类型到路径的映射
    """
    dataset_paths = {}
    
    # 处理所有支持的数据集类型
    for dataset_type in DEFAULT_DATASETS.keys():
        try:
            logger.info(f"确保{dataset_type}数据集可用...")
            path = ensure_dataset(dataset_type, force_download)
            dataset_paths[dataset_type] = path
            logger.info(f"{dataset_type}数据集已准备就绪: {path}")
        except Exception as e:
            logger.error(f"准备{dataset_type}数据集时出错: {e}")
            # 继续处理其他数据集，不中断流程
    
    return dataset_paths

if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集管理工具")
    parser.add_argument("--check", action="store_true", help="检查所有数据集状态")
    parser.add_argument("--download", choices=["mldr", "msmarco", "all"], help="下载指定数据集")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--export", choices=["mldr", "msmarco", "all"], help="导出数据集为JSON")
    
    args = parser.parse_args()
    
    if args.check:
        print("检查数据集状态:")
        for dataset in DEFAULT_DATASETS.keys():
            raw, processed, status = check_dataset_exists(dataset)
            print(f"{dataset} 数据集状态:")
            print(f"  - 原始数据: {'存在' if raw else '不存在'}")
            print(f"  - 处理后数据: {'存在' if processed else '不存在'}")
            print(f"  - 详细信息: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    if args.download:
        if args.download == "all":
            paths = ensure_all_datasets(args.force)
            print(f"所有数据集下载完成: {json.dumps(paths, indent=2, ensure_ascii=False)}")
        else:
            path = ensure_dataset(args.download, args.force)
            print(f"{args.download} 数据集下载完成: {path}")
    
    if args.export:
        if args.export == "all":
            for dataset in DEFAULT_DATASETS.keys():
                export_path = export_dataset_to_json(dataset)
                if export_path:
                    print(f"{dataset} 数据集已导出到: {export_path}")
                else:
                    print(f"{dataset} 数据集导出失败")
        else:
            export_path = export_dataset_to_json(args.export)
            if export_path:
                print(f"{args.export} 数据集已导出到: {export_path}")
            else:
                print(f"{args.export} 数据集导出失败") 
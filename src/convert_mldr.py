#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLDR数据集转换脚本
将MLDR数据集转换为RAG评估工具可用的格式
"""

import os
import argparse
import json
from datasets import Dataset, load_from_disk
import pandas as pd
import time
import shutil

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MLDR数据集转换工具")
    parser.add_argument("--input_dir", type=str, default="data/mldr", help="MLDR数据集目录")
    parser.add_argument("--output_dir", type=str, default="data/mldr_for_ragas", help="转换后数据集保存目录")
    parser.add_argument("--language", type=str, default="zh", help="要处理的语言代码")
    parser.add_argument("--split", type=str, default="test", help="要处理的数据分割(train/dev/test)")
    parser.add_argument("--samples", type=int, default=0, help="要处理的样本数量，0表示全部")
    return parser.parse_args()

def convert_mldr_to_ragas(input_dir, output_dir, language="zh", split="test", samples=0):
    """将MLDR数据集转换为RAG评估工具可用的格式"""
    print(f"开始转换MLDR数据集 ({language}-{split})...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载语料库
        corpus_path = os.path.join(input_dir, f"{language}-corpus")
        print(f"加载语料库: {corpus_path}")
        corpus = load_from_disk(corpus_path)
        
        # 创建文档ID到文本的映射
        doc_map = {}
        for doc in corpus:
            doc_map[doc["docid"]] = doc["text"]
        
        print(f"已加载语料库，共 {len(doc_map)} 个文档")
        
        # 加载查询数据集
        dataset_path = os.path.join(input_dir, f"{language}-{split}")
        print(f"加载查询数据集: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        
        # 限制样本数量
        if samples > 0 and samples < len(dataset):
            dataset = dataset.select(range(samples))
            print(f"已选择 {samples} 个样本进行处理")
        else:
            print(f"将处理全部 {len(dataset)} 个样本")
        
        # 准备转换后的数据
        converted_data = []
        
        # 处理每个查询
        for idx, item in enumerate(dataset):
            query = item["query"]
            query_id = item["query_id"]
            
            # 获取正面段落
            positive_passages = item["positive_passages"]
            if not positive_passages:
                print(f"警告: 查询 {query_id} 没有正面段落，跳过")
                continue
            
            # 为每个正面段落创建一条记录
            for passage in positive_passages:
                doc_id = passage["docid"]
                # 获取文档内容
                context = passage["text"] if "text" in passage and passage["text"] else doc_map.get(doc_id, "")
                
                if not context:
                    print(f"警告: 文档 {doc_id} 内容为空，跳过")
                    continue
                
                # 添加到转换数据中
                converted_data.append({
                    "query": query,
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "context": context,
                    "answer": ""  # RAG评估通常需要回答，但MLDR不提供，留空
                })
        
        # 创建Pandas DataFrame
        df = pd.DataFrame(converted_data)
        
        # 保存为Arrow格式
        print(f"转换完成，共处理 {len(df)} 条记录")
        output_dataset = Dataset.from_pandas(df)
        output_dataset.save_to_disk(output_dir)
        
        # 保存数据集信息
        dataset_info = {
            "language": language,
            "split": split,
            "original_samples": len(dataset),
            "converted_samples": len(df),
            "description": "从MLDR数据集转换的RAG评估数据"
        }
        
        with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n转换成功! 数据已保存到: {output_dir}")
        print(f"转换后样本数量: {len(df)}")
        
        return True
    
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 执行转换
    convert_mldr_to_ragas(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        language=args.language,
        split=args.split,
        samples=args.samples
    )

if __name__ == "__main__":
    main() 
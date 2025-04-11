#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLDR数据集下载脚本
专门用于下载MLDR数据集的中文部分
"""

import os
import argparse
from datasets import load_dataset
import time

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MLDR数据集下载工具")
    parser.add_argument("--language", type=str, default="zh", help="要下载的语言代码, 默认为中文(zh)")
    parser.add_argument("--output_dir", type=str, default="data/mldr", help="数据集保存目录")
    parser.add_argument("--splits", type=str, default="train,dev,test,corpus", help="要下载的数据集分割，用逗号分隔")
    return parser.parse_args()

def download_mldr(language="zh", output_dir="data/mldr", splits=None):
    """下载MLDR数据集的指定语言部分"""
    if splits is None:
        splits = ["train", "dev", "test"]
    
    print(f"开始下载MLDR数据集 ({language})...")
    print(f"数据将保存到: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 下载各个分割的数据集
        for split in splits:
            if split == "corpus":
                # 语料库需要特殊处理
                corpus_name = f"corpus-{language}"
                split_name = "corpus"
                print(f"下载语料库 ({corpus_name})...")
                start_time = time.time()
                corpus = load_dataset("Shitao/MLDR", corpus_name, split=split_name, trust_remote_code=True)
                save_path = os.path.join(output_dir, f"{language}-{split_name}")
                corpus.save_to_disk(save_path)
                print(f"语料库已保存到: {save_path}")
                print(f"语料库大小: {len(corpus)} 条文档")
                print(f"下载耗时: {time.time() - start_time:.2f} 秒")
            else:
                # 下载普通分割数据
                print(f"下载 {split} 数据集...")
                start_time = time.time()
                dataset = load_dataset("Shitao/MLDR", language, split=split, trust_remote_code=True)
                save_path = os.path.join(output_dir, f"{language}-{split}")
                dataset.save_to_disk(save_path)
                print(f"{split} 数据集已保存到: {save_path}")
                print(f"数据集大小: {len(dataset)} 条数据")
                print(f"下载耗时: {time.time() - start_time:.2f} 秒")
        
        print("\n下载完成! MLDR数据集已成功下载并保存。")
        print(f"总共下载了 {len(splits)} 个数据分割:")
        for split in splits:
            if split == "corpus":
                print(f"- {language}-corpus (语料库)")
            else:
                print(f"- {language}-{split}")
        
        return True
    
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        print("请检查网络连接或数据集名称是否正确。")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 解析要下载的分割
    splits = args.splits.split(",")
    
    # 下载数据集
    download_mldr(
        language=args.language,
        output_dir=args.output_dir,
        splits=splits
    )

if __name__ == "__main__":
    main() 
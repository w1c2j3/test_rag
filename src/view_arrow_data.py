#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Arrow数据文件查看工具
用于查看和导出Arrow格式数据文件的内容
"""

import os
import argparse
import pandas as pd
from datasets import load_from_disk

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Arrow数据查看工具")
    parser.add_argument("--input_dir", type=str, required=True, help="Arrow数据集目录路径")
    parser.add_argument("--output", type=str, default="", help="导出文件路径，支持.csv/.json/.xlsx格式")
    parser.add_argument("--limit", type=int, default=10, help="显示的记录数量，默认10条")
    parser.add_argument("--columns", type=str, default="", help="要显示的列，多列用逗号分隔，默认显示所有列")
    return parser.parse_args()

def view_arrow_data(input_dir, output="", limit=10, columns=""):
    """查看和导出Arrow格式数据文件的内容"""
    print(f"正在加载Arrow数据：{input_dir}")
    
    try:
        # 加载数据集
        dataset = load_from_disk(input_dir)
        
        # 显示数据集基本信息
        print("\n数据集信息:")
        print(f"记录数量: {len(dataset)}")
        print(f"字段列表: {list(dataset.features.keys())}")
        
        # 选择要显示的列
        selected_columns = columns.split(",") if columns else None
        
        # 转换为DataFrame以便更好地展示
        if selected_columns and all(col in dataset.features.keys() for col in selected_columns):
            df = dataset.select_columns(selected_columns).to_pandas()
        else:
            df = dataset.to_pandas()
        
        # 显示前N条记录
        print(f"\n前{min(limit, len(df))}条记录:")
        pd.set_option('display.max_colwidth', 50)  # 设置列宽，避免文本太长
        print(df.head(limit))
        
        # 导出数据（如果指定了输出路径）
        if output:
            output_dir = os.path.dirname(output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # 根据文件扩展名选择导出格式
            ext = os.path.splitext(output)[1].lower()
            
            if ext == '.csv':
                df.to_csv(output, index=False, encoding='utf-8')
                print(f"\n数据已导出为CSV: {output}")
            elif ext == '.json':
                df.to_json(output, orient='records', force_ascii=False, indent=2)
                print(f"\n数据已导出为JSON: {output}")
            elif ext == '.xlsx':
                df.to_excel(output, index=False)
                print(f"\n数据已导出为Excel: {output}")
            else:
                print(f"\n不支持的导出格式: {ext}，支持的格式为.csv/.json/.xlsx")
        
        return True
    
    except Exception as e:
        print(f"处理Arrow数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    args = parse_args()
    view_arrow_data(
        input_dir=args.input_dir,
        output=args.output,
        limit=args.limit,
        columns=args.columns
    )

if __name__ == "__main__":
    main() 
import os
import re
import pandas as pd
import json
import numpy as np
import argparse
from datetime import datetime

def parse_evaluation_file(file_path):
    """
    解析评估结果文件，提取批次评估结果
    
    Args:
        file_path (str): 评估结果文件路径
        
    Returns:
        list: 包含评估结果数据的列表
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取所有批次的评估结果
    batch_results = []
    
    # 尝试匹配更多字段的模式
    expanded_pattern = r"批次 (\d+) 评估结果:\n(?:.*?user_input: (\[.*?\]).*?)?(?:.*?retrieved_contexts: (\[.*?\]).*?)?(?:.*?response: (\[.*?\]).*?)?(?:.*?reference: (\[.*?\]).*?)?(?:.*?context_precision: (\[.*?\]).*?)?(?:.*?answer_relevancy: (\[.*?\]).*?)?(?:.*?faithfulness: (\[.*?\]).*?)?(?:.*?context_recall: (\[.*?\]).*?)?(?:.*?context_entities_recall: (\[.*?\]).*?)?(?:.*?accuracy: (\[.*?\]).*?)?(?:.*?completeness: (\[.*?\]).*?)?(?:.*?relevance: (\[.*?\]).*?)?"
    
    # 首先尝试使用扩展模式
    batch_matches = list(re.finditer(expanded_pattern, content, re.DOTALL))
    
    # 如果扩展模式没有匹配到任何结果，尝试使用基本模式
    if not batch_matches:
        print("使用基本匹配模式...")
        batch_pattern = r"批次 (\d+) 评估结果:\naccuracy: (\[.*?\])\ncompleteness: (\[.*?\])\nrelevance: (\[.*?\])"
        batch_matches = list(re.finditer(batch_pattern, content))
        
        # 如果仍然没有匹配，尝试提取单个指标
        if not batch_matches:
            print("使用单指标匹配模式...")
            # 逐个提取不同指标的结果
            metrics = {
                'accuracy': r"accuracy: (\[.*?\])",
                'completeness': r"completeness: (\[.*?\])",
                'relevance': r"relevance: (\[.*?\])",
                'context_precision': r"context_precision: (\[.*?\])",
                'answer_relevancy': r"answer_relevancy: (\[.*?\])",
                'faithfulness': r"faithfulness: (\[.*?\])",
                'context_recall': r"context_recall: (\[.*?\])",
                'context_entities_recall': r"context_entities_recall: (\[.*?\])"
            }
            
            results = {}
            for metric_name, pattern in metrics.items():
                matches = re.search(pattern, content)
                if matches:
                    try:
                        results[metric_name] = safe_eval(matches.group(1))
                    except:
                        print(f"无法解析 {metric_name} 结果")
            
            if results:
                # 如果至少提取到了一个指标，创建一个批次记录
                max_len = max([len(values) for values in results.values()]) if results.values() else 0
                for i in range(max_len):
                    result = {'batch': 1, 'index': i}
                    for metric, values in results.items():
                        if i < len(values):
                            result[metric] = values[i]
                        else:
                            result[metric] = np.nan
                    batch_results.append(result)
    else:
        # 处理扩展模式匹配结果
        for match in batch_matches:
            batch_num = match.group(1)
            
            # 提取各个指标
            field_mapping = {
                'user_input': 2,
                'retrieved_contexts': 3,
                'response': 4,
                'reference': 5,
                'context_precision': 6,
                'answer_relevancy': 7,
                'faithfulness': 8,
                'context_recall': 9,
                'context_entities_recall': 10,
                'accuracy': 11,
                'completeness': 12,
                'relevance': 13
            }
            
            # 尝试使用扩展模式字段
            fields = {}
            for field_name, group_idx in field_mapping.items():
                if group_idx <= len(match.groups()):
                    group_value = match.group(group_idx)
                    fields[field_name] = safe_eval(group_value) if group_value else []
            
            # 如果没有提取到扩展字段，使用原始字段
            if not fields.get('accuracy') and len(match.groups()) >= 3:
                fields['accuracy'] = safe_eval(match.group(2))
                fields['completeness'] = safe_eval(match.group(3))
                fields['relevance'] = safe_eval(match.group(4))
            
            # 确定结果数量（使用最长的指标列表长度）
            max_len = max([len(values) for values in fields.values()]) if fields.values() else 0
            
            # 将每个批次的结果添加到列表中
            for i in range(max_len):
                result = {
                    'batch': int(batch_num),
                    'index': i
                }
                
                # 为每个字段安全获取值
                for field_name, values in fields.items():
                    if i < len(values):
                        result[field_name] = values[i]
                    else:
                        result[field_name] = np.nan
                
                batch_results.append(result)
    
    # 如果batch_results为空，尝试提取单个评估结果
    if not batch_results:
        print("尝试提取单个评估结果...")
        single_result_pattern = r"(accuracy|completeness|relevance|context_precision|answer_relevancy|faithfulness|context_recall|context_entities_recall): ([\d\.]+)"
        matches = re.findall(single_result_pattern, content)
        if matches:
            result = {'batch': 1, 'index': 0}
            for metric, value in matches:
                try:
                    result[metric] = float(value)
                except:
                    result[metric] = np.nan
            batch_results.append(result)
    
    return batch_results

def safe_eval(s):
    """
    安全地解析字符串为Python对象
    
    Args:
        s (str): 要解析的字符串
        
    Returns:
        list: 解析后的Python对象
    """
    if not s:
        return []
    # 替换'nan'为'np.nan'，使eval可以正确处理
    s = s.replace('nan', 'np.nan')
    try:
        return eval(s)
    except:
        # 如果解析失败，尝试使用json解析（去除单引号）
        try:
            return json.loads(s.replace("'", "\""))
        except:
            # 如果仍然失败，返回空列表
            return []

def truncate_text(text, max_length=50):
    """
    截断长文本以便于显示
    
    Args:
        text: 要截断的文本
        max_length (int): 最大长度
        
    Returns:
        str: 截断后的文本
    """
    if not isinstance(text, str):
        return text
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def create_html_visualization(results):
    """
    创建HTML可视化展示
    
    Args:
        results (list): 评估结果列表
        
    Returns:
        str: HTML内容
    """
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 确定有哪些指标
    content_columns = ['user_input', 'retrieved_contexts', 'response', 'reference']
    metric_columns = [col for col in df.columns if col not in ['batch', 'index'] + content_columns]
    
    # 计算平均评分 (忽略NaN值)
    summary_data = {}
    for metric in metric_columns:
        if metric in df.columns:
            summary_data[metric] = df[metric].mean()
    
    # 计算总分数 (将NaN视为0)
    score_columns = [col for col in metric_columns if col in df.columns]
    if score_columns:
        df['total_score'] = df[score_columns].fillna(0).sum(axis=1)
        metrics_count = len(score_columns)
    else:
        df['total_score'] = 0
        metrics_count = 1
    
    avg_total = df['total_score'].mean() / metrics_count if metrics_count > 0 else 0
    
    # 创建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Evaluation Results</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            .summary-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .summary-box {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                width: 18%;
                text-align: center;
                margin-bottom: 10px;
            }}
            .summary-box h3 {{
                margin-top: 0;
                color: #333;
            }}
            .summary-box .score {{
                font-size: 24px;
                font-weight: bold;
                color: #2c7be5;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                font-size: 12px;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .score-1 {{
                background-color: #d4edda;
                color: #155724;
            }}
            .score-0 {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .score-nan {{
                background-color: #e9ecef;
                color: #6c757d;
            }}
            .tooltip {{
              position: relative;
              display: inline-block;
              cursor: pointer;
            }}
            .tooltip .tooltiptext {{
              visibility: hidden;
              width: 500px;
              background-color: #f8f9fa;
              color: #333;
              text-align: left;
              border-radius: 6px;
              border: 1px solid #ddd;
              padding: 10px;
              position: absolute;
              z-index: 1;
              bottom: 125%;
              left: 50%;
              margin-left: -250px;
              opacity: 0;
              transition: opacity 0.3s;
              white-space: pre-wrap;
              overflow-y: auto;
              max-height: 300px;
            }}
            .tooltip:hover .tooltiptext {{
              visibility: visible;
              opacity: 1;
            }}
            @media print {{
                .tooltip .tooltiptext {{
                    display: none;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>RAG Evaluation Results</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-container">
    """
    
    # 添加指标摘要框
    for metric, value in summary_data.items():
        display_name = metric.replace('_', ' ').title()
        html_content += f"""
            <div class="summary-box">
                <h3>{display_name}</h3>
                <div class="score">{value:.2f}</div>
            </div>
        """
    
    # 添加总体评分框
    html_content += f"""
            <div class="summary-box">
                <h3>Overall</h3>
                <div class="score">{avg_total:.2f}</div>
            </div>
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Batch</th>
                <th>Index</th>
    """
    
    # 添加所有字段的表头
    for col in content_columns:
        if col in df.columns:
            display_name = col.replace('_', ' ').title()
            html_content += f"<th>{display_name}</th>\n"
    
    # 添加评估指标的表头
    for metric in metric_columns:
        if metric in df.columns:
            display_name = metric.replace('_', ' ').title()
            html_content += f"<th>{display_name}</th>\n"
    
    html_content += "<th>Total</th>\n</tr>\n"
    
    # 添加每一行数据
    for _, row in df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['batch']}</td>
                <td>{row['index']}</td>
        """
        
        # 添加内容字段
        for col in content_columns:
            if col in df.columns:
                value = row[col]
                if isinstance(value, str):
                    # 对长文本使用工具提示
                    truncated = truncate_text(value)
                    html_content += f"""
                    <td class="tooltip">
                        <span class="long-text">{truncated}</span>
                        <span class="tooltiptext">{value}</span>
                    </td>
                    """
                else:
                    html_content += f"<td>-</td>"
            
        # 为每个指标添加适当的样式
        for metric in metric_columns:
            if metric in df.columns:
                value = row[metric]
                if pd.isna(value):
                    html_content += f'<td class="score-nan">NaN</td>'
                elif isinstance(value, (int, float)):
                    css_class = f"score-{int(value)}" if value in [0, 1] else ""
                    html_content += f'<td class="{css_class}">{value:.2f}</td>'
                else:
                    html_content += f'<td>{value}</td>'
        
        # 添加总分
        html_content += f"<td>{row['total_score']:.2f}</td></tr>\n"
    
    html_content += """
        </table>
        
        <script>
        // 添加展开/收起长文本的功能
        document.querySelectorAll('.long-text').forEach(element => {
            element.addEventListener('click', function() {
                this.classList.toggle('expanded');
            });
        });
        </script>
    </body>
    </html>
    """
    
    return html_content

def visualize_evaluation(input_file=None, output_file=None):
    """
    可视化评估结果的主函数
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        
    Returns:
        str: 输出文件路径
    """
    # 确定输入文件路径
    if input_file is None:
        # 查找最新的评估结果文件
        eval_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results")
        if not os.path.exists(eval_results_dir):
            os.makedirs(eval_results_dir)
            
        result_files = [f for f in os.listdir(eval_results_dir) 
                      if (f.startswith("rag_eval_") or f.endswith("_eval_results.txt"))
                      and f.endswith(".txt")]
        if not result_files:
            print("错误：没有找到评估结果文件")
            return None
        
        # 按修改时间排序，选择最新的文件
        latest_file = sorted(result_files, key=lambda x: os.path.getmtime(os.path.join(eval_results_dir, x)))[-1]
        input_file = os.path.join(eval_results_dir, latest_file)
        print(f"使用最新的评估结果文件: {input_file}")
    
    print(f"正在解析文件: {input_file}")
    
    # 解析评估结果
    results = parse_evaluation_file(input_file)
    
    if not results:
        print("错误：未能解析出评估结果，请检查文件格式")
        return None
    
    print(f"成功解析 {len(results)} 条评估结果")
    
    # 创建HTML可视化
    html_content = create_html_visualization(results)
    
    # 生成输出文件名
    if output_file is None:
        eval_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results")
        output_base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(eval_results_dir, f"visualization_{output_base}.html")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"可视化结果已保存到: {output_file}")
    return output_file

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='可视化RAG评估结果')
    parser.add_argument('input_file', nargs='?', help='输入评估结果文件路径')
    parser.add_argument('--output', '-o', help='输出HTML文件路径')
    args = parser.parse_args()
    
    visualize_evaluation(args.input_file, args.output)

if __name__ == "__main__":
    main() 
"""
下载并准备数据集用于Ragas评估
"""

import os
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas.metrics import AspectCritic
from ragas import evaluate
from src.custom_api_client import LangchainCustomLLMWrapper

def download_dataset(dataset="msmarco", output_dir="data/msmarco_for_ragas"):
    """
    下载指定数据集并保存到指定目录
    
    参数:
        dataset: 数据集名称，支持"msmarco"
        output_dir: 输出目录路径
        
    返回:
        bool: 下载是否成功
    """
    print(f"开始下载数据集: {dataset}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset.lower() == "msmarco":
        try:
            # 下载并准备MS MARCO数据集
            prepared_dataset = download_and_prepare_msmarco(output_dir)
            return True
        except Exception as e:
            print(f"下载MS MARCO数据集失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"不支持的数据集: {dataset}")
        return False

def download_and_prepare_msmarco(output_dir="data/msmarco_for_ragas"):
    """
    下载MS MARCO数据集并准备用于Ragas评估，保存到指定目录
    
    参数:
        output_dir: 输出目录路径
        
    返回:
        prepared_dataset: 准备好的数据集
    """
    print("开始下载MS MARCO数据集...")
    # 加载MS MARCO数据集，这是一个大型问答数据集
    # 仅加载一个较小的子集用于测试 - v2.1格式
    dataset = load_dataset("ms_marco", "v2.1", split="train[:5000]")
    
    print(f"数据集加载完成，大小: {len(dataset)} 个样本")
    print(f"数据集结构: {dataset.column_names}")
    
    # 将MS MARCO数据集转换为Ragas评估数据集格式
    # MS MARCO有passages列表和答案，我们需要重组
    print("准备评估数据集...")
    
    # 构建Ragas需要的数据格式
    ragas_data = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truths": []
    }
    
    # 从MS MARCO数据集中提取数据
    for sample in dataset:
        # 只处理有答案的数据
        if len(sample['answers']) > 0:
            question = sample['query']
            # 获取相关段落作为上下文
            contexts = []
            for i, is_selected in enumerate(sample['passages']['is_selected']):
                if is_selected:
                    contexts.append(sample['passages']['passage_text'][i])
            
            # 只有当有上下文和答案时才添加
            if contexts and sample['answers']:
                ragas_data["question"].append(question)
                ragas_data["contexts"].append(contexts)
                ragas_data["answer"].append(sample['answers'][0])
                ragas_data["ground_truths"].append([ans for ans in sample['answers']])
    
    print(f"准备完成 {len(ragas_data['question'])} 个有效样本")
    
    # 保存数据集以备将来使用
    from datasets import Dataset as HFDataset
    prepared_dataset = HFDataset.from_dict(ragas_data)
    prepared_dataset.save_to_disk(output_dir)
    
    print(f"数据集已保存到 {output_dir}")
    return prepared_dataset

def download_and_prepare_dataset():
    """
    下载MS MARCO数据集并准备用于Ragas评估
    """
    return download_and_prepare_msmarco()

def evaluate_with_dataset(dataset):
    """
    使用下载的数据集进行Ragas评估
    """
    print("\n开始使用MS MARCO数据集进行评估...")
    
    # 创建评估指标
    accuracy_critic = AspectCritic(
        name="accuracy",
        definition="评估回答的准确性，验证回答是否与上下文内容一致"
    )
    
    completeness_critic = AspectCritic(
        name="completeness", 
        definition="评估回答的完整性，验证回答是否涵盖了上下文中的所有关键信息"
    )
    
    try:
        # 只评估前10个样本，以便快速测试
        test_dataset = dataset.select(range(min(10, len(dataset))))
        
        # 在评估时指定我们的自定义LLM客户端
        llm_wrapper = LangchainCustomLLMWrapper(
            api_key="your_api_key_here",
            api_url="https://api.ppai.pro/v1/chat/completions",
            model_name="gpt-4o-2024-11-20"
        )
        
        # 在评估时指定LLM
        print("\n正在进行评估，这可能需要一些时间...")
        result = evaluate(
            dataset=test_dataset,
            metrics=[accuracy_critic, completeness_critic],
            llm=llm_wrapper
        )
        
        # 输出结果
        print("\n评估结果:")
        if hasattr(result, '_scores_dict'):
            scores_dict = result._scores_dict
            print(scores_dict)
            for metric_name in scores_dict:
                try:
                    print(f"{metric_name}: {scores_dict[metric_name]:.4f}")
                except:
                    print(f"{metric_name}: {scores_dict[metric_name]}")
        else:
            print(result)
            for metric_name in ["accuracy", "completeness"]:
                if hasattr(result, metric_name):
                    value = getattr(result, metric_name)
                    try:
                        print(f"{metric_name}: {value:.4f}")
                    except:
                        print(f"{metric_name}: {value}")
                else:
                    print(f"{metric_name}: 无法获取结果")
                
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("注意: 请确保API连接正常，并且API服务支持ragas所需的功能")

if __name__ == "__main__":
    # 创建数据目录(如果不存在)
    os.makedirs("data", exist_ok=True)
    
    # 检查数据集是否已下载
    if os.path.exists("data/msmarco_for_ragas"):
        print("使用已下载的数据集...")
        from datasets import Dataset as HFDataset
        dataset = HFDataset.load_from_disk("data/msmarco_for_ragas")
    else:
        # 下载并准备数据集
        dataset = download_and_prepare_dataset()
    
    # 输出数据集大小和前几个示例
    print(f"处理后的数据集大小: {len(dataset)} 个样本")
    if len(dataset) > 0:
        print("\n数据集前3个示例:")
        for i in range(min(3, len(dataset))):
            print(f"\n示例 {i+1}:")
            print(f"问题: {dataset[i]['question']}")
            print(f"上下文片段数: {len(dataset[i]['contexts'])}")
            print(f"答案: {dataset[i]['answer']}")
    
    # 使用数据集进行评估
    evaluate_with_dataset(dataset) 
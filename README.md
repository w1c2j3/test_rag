# RAG评估工具包 - 新手友好指南

一个简单易用的RAG（检索增强生成）系统评估工具包，帮助您评估大语言模型在检索和生成任务中的表现。

## 项目文件结构

整个项目的文件结构如下：

```
/
├── main.py              # 主程序入口，包含API配置和命令行参数处理
├── requirements.txt     # 依赖包列表
├── src/                 # 源代码目录
│   ├── custom_api_client.py   # API客户端实现
│   ├── evaluate_rag.py        # RAG评估核心功能
│   ├── dataset_manager.py     # 数据集管理功能（包含数据集语言设置）
│   ├── download_mldr.py       # MLDR数据集下载工具
│   ├── convert_mldr.py        # MLDR数据转换工具
│   ├── view_arrow_data.py     # Arrow数据查看和导出工具
│   └── test_api.py            # API测试工具
├── data/                # 数据集存储目录
│   ├── mldr/            # MLDR数据集存储位置
│   │   ├── zh-corpus/   # 中文语料库数据
│   │   └── zh-test/     # 中文测试数据
│   ├── export/          # 导出的数据集存储位置
│   └── processed/       # 处理后的数据集
└── evaluation_results/  # 评估结果存储目录
```

## 数据集配置指南

### 设置评估样本数量

您可以通过`--samples`参数控制用于评估的样本数量，这对于区分测试环境和生产环境非常有用：

1. **测试环境使用**（少量样本，快速验证）：
   ```bash
   # 仅使用2个样本进行测试，快速验证系统是否正常工作
   python main.py --samples 2
   
   # 使用5个样本进行小规模测试
   python main.py --samples 5
   ```

2. **生产环境使用**（大量或全部样本，全面评估）：
   ```bash
   # 使用50个样本进行中等规模评估
   python main.py --samples 50
   
   # 使用100个样本进行大规模评估
   python main.py --samples 100
   
   # 使用全部数据集（不指定--samples参数或设置为0）
   python main.py
   # 或
   python main.py --samples 0
   ```

3. **结合批处理大小**（优化处理效率）：
   ```bash
   # 处理大量样本时，增加批处理大小可提高效率
   python main.py --samples 100 --batch-size 20
   ```

### 修改数据集语言

默认情况下，系统会自动下载并使用MLDR中文(zh)数据集。您可以通过以下方式修改语言设置：

1. **通过命令行参数修改**：
   ```bash
   # 下载英文(en)数据集而非默认的中文(zh)
   python -m src.download_mldr --language en --splits test,corpus --output_dir data/mldr
   ```

2. **直接修改源码**：
   打开`src/dataset_manager.py`文件，找到MLDR数据集下载相关的代码部分，将默认语言参数从'zh'修改为您需要的语言代码，如'en'(英文)、'de'(德文)等。

### 数据集格式转换与可视化

MLDR数据集默认以Arrow格式存储，可以通过以下命令查看和导出为不同格式：

```bash
# 查看数据（显示前10条记录）
python -m src.view_arrow_data --input_dir data/mldr/zh-test

# 导出为CSV格式
python -m src.view_arrow_data --input_dir data/mldr/zh-test --output data/export/mldr_test.csv

# 导出为JSON格式（推荐）
python -m src.view_arrow_data --input_dir data/mldr/zh-test --output data/export/mldr_test.json

# 仅显示特定字段
python -m src.view_arrow_data --input_dir data/mldr/zh-test --columns "query,query_id"
```

## 快速开始（5分钟上手）

### 一键安装与配置 (新手推荐)

我们提供了简便的安装脚本，可以一步完成环境配置：

```bash
# 下载并安装依赖，同时创建目录结构和下载示例数据
python setup.py --api-key "您的API密钥" --create-dirs --download-sample

# 如果您暂时没有API密钥，也可以先安装环境
python setup.py --create-dirs --download-sample
```

安装完成后，您可以直接运行评估：

```bash
# 使用内置数据集运行评估
python main.py --samples 5
```

### 手动安装与配置

如果您希望手动控制安装过程，请按照以下步骤操作：

#### 1. 环境准备

确保您已安装Python 3.8或更高版本，然后按照以下步骤操作：

```bash
#  安装依赖
pip install -r requirements.txt
```

#### 2. 设置API密钥

在`main.py`文件中设置您的API密钥：

```python
# 打开main.py文件，找到并修改以下部分(位于文件顶部的API配置部分)
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 替换为您的API密钥
API_BASE_URL = "https://api.ppai.pro/v1/chat/completions"  # API基础URL
API_MODEL = "deepseek-v3"  # 使用的模型名称
```

#### 3. 测试API连接

```bash
# 测试API连接是否正常
python main.py --test-only
```

如果看到"API连接测试通过!"，则表示配置正确，可以继续下一步。

#### 4. 运行评估（使用默认数据集）

```bash
# 使用内置数据集运行评估
python main.py --samples 10 --batch-size 5
```

#### 5. 查看评估结果

评估完成后，结果会保存在`evaluation_results`目录中，同时会在控制台显示摘要信息。

## 使用自定义数据集

### 方法1：使用MLDR中文数据集

默认情况下，系统会自动下载并使用MLDR中文(zh)数据集。如果您想修改语言设置或自定义下载参数，可以按以下步骤操作：

```bash
# 1. 下载MLDR数据集（修改语言参数）
# 默认是中文(zh)，您可以修改为其他语言，如英文(en)
python -m src.download_mldr --language zh --splits test,corpus --output_dir data/mldr

# 2. 转换为评估工具可用格式（选择10个样本）
python -m src.convert_mldr --input_dir data/mldr --output_dir data/mldr_for_ragas --language zh --split test --samples 10

# 3. 运行评估
python main.py --dataset data/mldr_for_ragas --samples 10 --batch-size 5
```

您也可以直接修改源码中的默认设置：
- 打开`src/dataset_manager.py`文件，找到MLDR数据集下载相关的代码部分
- 修改默认的语言参数(默认为'zh')为您需要的语言代码

### 方法2：使用MS MARCO数据集

```bash
# 1. 下载MS MARCO数据集
python -m src.download_dataset --dataset msmarco --output_dir data/msmarco_for_ragas

# 2. 运行评估
python main.py --dataset data/msmarco_for_ragas --samples 10 --batch-size 5
```

### 方法3：导入您自己的数据

准备一个包含以下列的CSV文件：
- `query`：问题文本
- `context`：相关上下文内容
- `answer`：参考答案（可选）

然后使用以下命令转换数据：

```bash
python -m src.data_processor --input your_data.csv --output data/custom_data
```

最后运行评估：

```bash
python main.py --dataset data/custom_data
```

## 常见问题解答

### 1. 为什么评估结果全是null值？

这通常表示API请求失败了。请检查：
- API密钥是否正确
- API服务是否正常
- 网络连接是否稳定

### 2. 为什么数据是.arrow格式？我无法直接打开它。

Arrow是一种高效的二进制列式存储格式，设计用于快速数据处理，而非直接查看。使用我们提供的`view_arrow_data.py`工具可以查看和导出数据。

### 3. 我想评估不同的指标，怎么做？

使用`--metrics`参数指定要评估的指标：

```bash
python main.py --metrics accuracy,completeness,relevance,coherence
```

支持的指标包括：accuracy, completeness, relevance, coherence, conciseness等。

### 4. 如何调整批处理大小？

使用`--batch-size`参数控制批处理大小：

```bash
python main.py --batch-size 5  # 较小的批次，适合API限制严格的情况
python main.py --batch-size 20  # 较大的批次，处理速度更快
```

### 5. 如何查看详细的评估报告？

评估结果会保存在`evaluation_results`目录下，包括文本报告和JSON格式数据。您还可以使用`--output`和`--json-output`参数指定输出文件路径：

```bash
python main.py --output my_results.txt --json-output my_results.json
```

## 评估指标说明

我们支持以下评估指标：

- **accuracy**：评估回答的准确性，验证是否与上下文一致且无虚假信息
- **completeness**：评估回答的完整性，验证是否涵盖所有关键信息
- **relevance**：评估回答与问题的相关性，验证是否直接解答问题
- **coherence**：评估回答的连贯性，验证是否逻辑清晰且结构良好
- **conciseness**：评估回答的简洁性，验证是否简明扼要，不含冗余信息

## 高级用法

如果您熟悉Python和RAG评估，可以尝试以下高级功能：

1. 自定义评估指标：修改`src/evaluate_rag.py`中的`create_metrics`函数
2. 集成其他数据集：参考现有下载器编写新的数据集下载器
3. 自定义API客户端：修改`src/custom_api_client.py`使用不同的LLM API服务

更详细的高级用法请参考`docs/`目录下的技术文档。 
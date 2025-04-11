# RAG评估工具包优化开发计划

## 主要目标

基于用户需求，本次开发计划旨在优化现有RAG评估工具包，解决代码重复、流程复杂等问题，提供更流畅的用户体验。优化后用户只需填入API密钥并运行main.py，即可完成从数据集准备到评估全过程。

## 需要修改的内容

### 1. 数据集管理优化

#### 修改文件：`src/dataset_manager.py`

1. 增强`check_dataset_exists`函数
   - 检测data目录中是否存在mldr和msmarco数据集
   - 返回更详细的数据集状态信息

2. 改进`ensure_dataset`函数
   - 如果数据集不存在，自动下载MLDR中文数据集和MS MARCO数据集
   - 下载完成后自动将数据集转换为JSON格式
   - 增加进度提示，提高用户体验

3. 扩展`export_dataset_to_json`函数
   - 确保导出后的JSON数据格式统一
   - 优化错误处理机制

### 2. API测试逻辑统一

#### 修改文件：`main.py`和`src/test_api.py`

1. 修改`main.py`中的`test_api_connection`函数
   - 移除重复逻辑
   - 直接调用`src/test_api.py`中的测试方法
   - 保留错误处理和用户友好提示

2. 确保`src/test_api.py`中的`ApiTester`类
   - 提供完整的API测试功能
   - 支持参数化配置测试条件

### 3. 参数解析系统统一

#### 修改文件：`main.py`和`src/evaluate_rag.py`

1. 整合参数解析逻辑
   - 移除`src/evaluate_rag.py`中的`parse_args`函数
   - 扩展`main.py`中的`parse_args`函数，包含所有需要的参数
   - 修改`evaluate_dataset`函数接口，接受完整的参数字典

2. 增加新参数
   - `--dataset-type`参数：指定要使用的数据集类型（mldr或msmarco）
   - `--force-download`参数：强制重新下载数据集

### 4. 评估入口统一

#### 修改文件：`main.py`、`run_mldr_eval.py`和`src/ragas_example.py`

1. 强化`main.py`作为唯一入口点
   - 将其他脚本中的核心功能整合到main.py
   - 保留其他脚本作为示例或模块，但标记为不推荐直接使用

2. 重构主流程
   - 简化`main.py`中的`main`函数
   - 确保流程清晰：环境检查->数据准备->API测试->执行评估->结果输出

### 5. 下载功能整合

#### 修改文件：`src/download_dataset.py`和`src/download_mldr.py`

1. 在保留现有功能的前提下整合接口
   - 不直接合并文件，避免破坏现有功能
   - 通过`dataset_manager.py`提供统一的下载入口
   - 根据数据集类型调用相应的下载函数

### 6. 目录创建逻辑优化

#### 修改文件：`main.py`

1. 增强`ensure_directories`函数
   - 确保评估结果目录存在
   - 确保数据导出目录存在
   - 添加更详细的目录创建日志

## 实现步骤

1. 首先修改`src/dataset_manager.py`，完善数据集检测和下载功能
2. 修改`main.py`中的API测试逻辑，使用`src/test_api.py`中的实现
3. 整合参数解析系统，确保参数一致性
4. 修改目录创建逻辑，确保必要目录存在
5. 强化`main.py`作为唯一入口点
6. 测试整个流程，确保从数据准备到评估结果输出的顺利进行

## 预期成果

1. 用户只需在`main.py`中填入API密钥
2. 运行`python main.py`即可自动完成：
   - 检测数据集是否存在，不存在则自动下载
   - 将数据集转换为可用格式
   - 测试API连接
   - 执行RAG评估
   - 输出并保存评估结果
3. 提供更清晰的错误提示和进度信息
4. 代码结构更加清晰，减少重复逻辑 
# 评论情感分析项目

## 项目简介

该项目实现了基于 BERT 和 LSTM 的评论情感分类模型，用于判断评论文本的情感倾向（积极或消极）。在开发过程中，充分利用了 Hugging Face 平台的资源和工具,需要准备好科学上网工具。

## 功能描述

- 数据预处理：包括文本清洗、分词、去除停用词等操作。
- BERT 模型：利用 Hugging Face 平台提供的预训练 BERT 模型进行文本特征提取，并添加全连接层进行情感分类。
- LSTM 模型：结合 Word2Vec 词向量训练，使用 LSTM 网络进行序列建模，实现情感分类。
- 模型训练和评估：在训练集上训练模型，并在测试集上评估模型性能。

## 技术栈

- Python
- PyTorch
- Hugging Face Transformers
- Gensim
- Jieba
- Pandas

## 使用方法

1. 克隆项目到本地：

2. 安装依赖库：

```bash
pip install torch pandas transformers gensim jieba
```

3. 准备数据集：
   - 将训练数据和测试数据分别保存为 `train.tsv` 和 `test.tsv` 文件。
   - 确保数据集的格式符合项目要求。

4. 运行项目：

```bash
python main.py
```

## 项目结构

- `BERT.py`：基于 BERT 的情感分类模型实现。
- `data_set.py`：数据预处理和加载模块。
- `main.py`：LSTM 模型实现和训练主程序。
- `test.py`：LSTM 模型预测脚本。
- `test_bert.py`：BERT 模型预测脚本。

## 示例运行结果

- 在测试集上，BERT 模型和 LSTM 模型分别达到了 82% 和 73% 的准确率。

## 项目亮点

1. **利用 Hugging Face 的预训练 BERT 模型**：通过 Hugging Face 的 Transformers 库，快速加载和使用预训练的 BERT 模型，简化了模型的实现过程。
2. **LSTM 模型的序列建模能力**：通过结合 Word2Vec 词向量，LSTM 模型能够有效捕捉文本序列信息，提升分类性能。
3. **数据预处理的完整性**：项目中实现了文本清洗、分词、去除停用词等多个预处理步骤，保证了数据质量。
4. **模型检查点保存机制**：在训练过程中，项目会保存最佳模型参数，便于后续加载和使用。

## Hugging Face 的使用

在该项目中，Hugging Face 的 Transformers 库发挥了重要作用：

1. **加载预训练模型**：使用 `BertTokenizer` 和 `BertModel` 从 Hugging Face 平台加载预训练的 BERT 模型。
2. **模型微调**：在预训练模型的基础上，添加全连接层进行微调，以适应特定的情感分类任务。
3. **利用 Hugging Face 的 API**：通过简洁易用的 API，快速实现模型的加载和使用，提高了开发效率。

## 贡献者

- <yako666666>

## 联系方式

如有任何问题或建议，请联系 <2929687470@qq.com>。

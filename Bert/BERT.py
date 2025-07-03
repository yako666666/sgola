import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import numpy as np
from data_set import load_tsv

# 数据读取
def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [[line.strip()] for line in f.readlines()]
        return data

train_x = load_txt('train.txt')
test_x = load_txt('test.txt')
train = train_x + test_x
X_all = [i for x in train for i in x]

_, train_y = load_tsv("C:/Users/29296/Desktop/train.tsv")
_, test_y = load_tsv("C:/Users/29296/Desktop/test.tsv")

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 将文本转换为BERT输入格式
def text_to_bert_input(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

X_train_input_ids, X_train_attention_mask = text_to_bert_input([line[0] for line in train_x])
X_test_input_ids, X_test_attention_mask = text_to_bert_input([line[0] for line in test_x])

# 使用DataLoader打包文件
train_dataset = TensorDataset(X_train_input_ids, X_train_attention_mask, torch.LongTensor(train_y))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TensorDataset(X_test_input_ids, X_test_attention_mask, torch.LongTensor(test_y))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 定义BERT模型
class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # 输出层

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

# 定义模型
model = BERTModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 检查是否有保存的检查点
checkpoint_path = 'model_checkpoint.pth'
start_epoch = 0
best_accuracy = 0.0

try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    print(f"Loaded checkpoint from epoch {start_epoch} with best accuracy {best_accuracy:.4f}")
except FileNotFoundError:
    print("No checkpoint found. Starting from scratch.")

if __name__ == "__main__":
    # 训练模型
    num_epochs = 3
    log_interval = 100
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch_idx, (input_ids, attention_mask, target) in enumerate(train_loader):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, batch_idx, len(train_loader), loss.item()))

        # 模型评估
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for input_ids, attention_mask, target in test_loader:
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            accuracy = correct / total
            print('Test Accuracy: {:.2%}'.format(accuracy))

            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy
                }, checkpoint_path)
                print(f"Saved checkpoint with accuracy {best_accuracy:.4f}")
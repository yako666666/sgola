import pandas as pd
import jieba
import re

# 数据读取
def load_tsv(file_path):
    data = pd.read_csv(file_path, sep='\t')
    data_x = data.iloc[:, -1]  # 假设评论在最后一列
    data_y = data.iloc[:, 1]  # 假设标签在第二列
    return data_x, data_y

# 清洗文本
def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除特殊字符和多余的空格
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 分词处理
def tokenize_text(text):
    return list(jieba.cut(text))

# 加载停用词
def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    return set(stopwords)

# 去除停用词
def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if token not in stopwords]

# 保存数据
def save_data(datax, path):
    with open(path, 'w', encoding="UTF8") as f:
        for lines in datax:
            for i, line in enumerate(lines):
                f.write(str(line))
                # 如果不是最后一行，就添加一个逗号
                if i != len(lines) - 1:
                    f.write(',')
            f.write('\n')

if __name__ == '__main__':
    # 加载数据
    train_x, train_y = load_tsv("C:/Users/29296/Desktop/train.tsv")
    test_x, test_y = load_tsv("C:/Users/29296/Desktop/test.tsv")

    # 加载停用词
    stop_words = load_stopwords('C:/Users/29296/Desktop/hit_stopwords.txt')

    # 数据预处理
    train_x = [clean_text(x) for x in train_x]
    test_x = [clean_text(x) for x in test_x]

    train_x = [tokenize_text(x) for x in train_x]
    test_x = [tokenize_text(x) for x in test_x]

    train_x = [remove_stopwords(x, stop_words) for x in train_x]
    test_x = [remove_stopwords(x, stop_words) for x in test_x]

    # 保存处理后的数据
    save_data(train_x, './train.txt')
    save_data(test_x, './test.txt')

    print('Successfully')
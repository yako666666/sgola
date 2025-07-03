import torch
import jieba
from torch import nn
from gensim.models import Word2Vec
import numpy as np
import re  # 导入正则表达式模块

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 取序列的最后一个输出
        return output

# 数据读取
def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f.readlines()]
        return data

# 去停用词
def drop_stopword(datas):
    with open('C:/Users/29296/Desktop/hit_stopwords.txt', 'r', encoding='UTF8') as f:
        stop_words = [word.strip() for word in f.readlines()]
    return [word for word in datas if word not in stop_words]

def preprocess_text(text):
    words = list(jieba.cut(text))
    return drop_stopword(words)

# 将文本转换为Word2Vec向量表示
def text_to_vector(text):
    # 加载训练好的Word2Vec模型
    word2vec_model = Word2Vec.load("word2vec.model")  # 确保路径正确
    vector = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
    return sum(vector) / len(vector) if vector else [0] * word2vec_model.vector_size

# 按标点符号分割文本
def split_text_by_punctuation(text):
    # 使用正则表达式按句号、问号、感叹号分割
    sentences = re.split(r'[。！？]', text)
    # 去除空字符串
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

if __name__ == '__main__':
    input_text = "普通人对于ChatGPT的使用，我觉得应该有三重境界。第一重是压根没有听说过ChatGPT的，这部分人不多，毕竟就连防沉迷系统都过不了的小孩都知道有这么个东西了。第二重是听说过但是没用过的，这部分应该占绝大多数，就算把各种套壳小程序等都算上，我觉得也没有多少人，因为这东西最直接的印象就是一个空空的对话框，特别是移动互联网时代，大多数人上网用的是手机，而ChatGPT明显是更适合电脑上的浏览器，所以使用的人就更少了。最后一重就是听说过，也使用过，我估计看这个回答的人大多数都没有机会去用真正的ChatGPT，这也是为什么「如何注册ChatGPT」「国内ChatGPT」「中文版ChatGPT」这类东西从2023年12月ChatGPT发布开始就一直也没停过。能成为攻略且一直都有新的攻略，只能证明这个东西有两个特点：1、有用，在多个领域都有很强的应用价值。不管是写代码、做报表、做汇报、学英语、扒文献等等，所以有需求、有受众。2、  难用，这个「难用」是指普通人很难才能用到，权限比较难获取。所以很多人其实都是卡在GPT权限这部分，其实有一个简单的方法，就是下面这个知乎知学堂官方推出的AI工具课，从GPT权限、实操技巧、一键数据分析到GPT赚钱一次性讲清楚，看完你就知道普通人如何最大化利用GPT了。不知道啥时候没，建议先预约一手，官方入口指路↓↓↓AI工具训练🔥零基础上手20+主流AI工具￥0免费领取听完别忘记领取他们整合的AI资源合集，尤其是第一个！！！ 1、  主流AI工具合集，附官网链接，点开直接用就可以； 2、  AI提示词设计指南，教你和GPT科学对话； 3、  工作汇报PPT模版20套，可以套壳直接用； 4、  提示词工程指南，程序员、工程师必备的；包括我在内，刚开始对着输入框，也只知道问一些比如「给我写首诗」，「按照***给我写一篇作文」这种问题。ChatGPT确实擅长这个，它很好的根据我们的要求写了出来。我也让它写过歌，写过绝句，写过律诗，但是问题来了：ChatGPT到底能干些什么？很多人，包括我在内，都在说ChatGPT除了不会生孩子，其他的都能干。但是这并不是一个具有可操作性的说法，这次我要做的就是把ChatGPT可以干什么，以及它可以帮助你干什么讲明白，讲细。ChatGPT的操作，一切都要从这个对话框开始，所以你在这个框里输入的东西，就代表了你将要用它来做什么。比如你要输入的是文字。文字可以是中文，也可以是英文，也可以是其他的语言。假设我们输入的是中文，我们可以随便输入一句话：这是我第一次用ChatGPT。然后接下来的是重点，我们想干什么？比如说最简单的，你可以什么都不输入，直接点击回车键看看ChatGPT能干什么。这个回答是单纯的这句话，ChatGPT会自动识别出你并没有什么具体的想法，所以它会主动的提示你，它可以帮助你回答任何问题，其实这也就是ChatGPT中Chat的核心含义。ChatGPT最主要的沟通方式就体现在Chat这个词上，也就是交流。所以即使你不会用，也可以顺着它的话继续说，比如你可以问它「请问你可以根据我提供的第一句话干什么？」你看它会告诉你有几种可能性，给出你选项你可能就知道下一步该问什么了。这个过程你应该掌握的是，ChatGPT是一个非常开放且自由的工具，它理论上会回答你提出的任何问题。所以结合ChatGPT的输入框+它的对话特性，我们就可以自己来分析到底用ChatGPT可以做什么。我先给你分享一个案例，一个完全没有任何编程经验的老哥，在ChatGPT出现后凭着直觉认为2023年的AI会爆火，因此他用ChatGPT写了一个AI的导航站。有些朋友可能不知道什么是导航站。就类似于这样的，就是总结了很多的网站，然后把他们分门别类的整理在一起，比如大模型软件，有ChatGPT，bard，文心一言等等，这样做的就是方便人们使用。这个大哥经历过以前的hao123时代，就觉得AI时代也需要，事实上就是这样的，他这个站点大概在2月份就上线了，在ChatGPT的帮助下基本上1晚上就搞出了这个站点的雏形，差不多10个月过去了，这个网站可以给他带来每个月差不多2000美元的收入。这也是我为什么总是说ChatGPT这样的AI工具，每个人都应该去尝试跟它结合。结合的方式我觉得也分几层。第一层的结合起码你得让它帮你减轻点学习或者工作的负担，比如让它把你今天背的单词写成一个短文，这样可以方便让你理解。比如我们要背这些单词，那么我们就可以直接复制粘贴到ChatGPT中。然后用这个Prompt：用这些单词写一篇短文方便我记忆，给我提供的单词加粗，并且提供中文翻译。你看，它非常的智能的给你用这些词汇写了一篇非常地道的英文短文，同时还会把你要背的单词的中文意思标注在后面。是不是非常的智能，但请你记住，这只是我的想法，你要做的是找到适合你的做法，而不是直接照抄我的。工作也一样，比如你的工作需要写一些材料性质的文档，你完全可以让ChatGPT帮你写一个初稿出来，然后在这个基础上修改，会非常的省事。第二层就是使用ChatGPT来为自己创造一份额外的收入，就跟刚才那个例子的老哥一样，其实每个人都有机会在ChatGPT的帮助下实现自己的想法。更多的是可能不知道怎么去着手，其实你只要知道你喜欢什么且ChatGPT能干什么之后，可能就比较清晰了。比如对我来说，我喜欢写文字，那我就是用的ChatGPT文字润色功能，以及图像输入和代码执行环境来帮助我写各种各样的教程或者演示视频。那对你来说，其实也是一样的逻辑，那就是根据ChatGPT可以接受的输入类型来确认你自己的方向。比如对于ChatGPT来说，它最基本的也是最强的就是文本输入。你看它的文本输入功能，有回答问题（契合知乎，百度知道等），提供信息（契合攻略类，问答型），撰写文本（契合各种自媒体，比如小红书，抖音等等各类平台的文案书写）。你可以看到，它最基本的文本输入功能，都可以做这么多的事情。它还有其他各种先进的功能，而你要做的其实就是提供一个想法。想法最重要，剩下的反而ChatGPT会教你一步步的怎么做，然后你要做的就是要去实践你的想法。下面就举一个我刚刚实现的案例，比如我想仿照刚刚提到的大哥做个AI导航站，这是我的想法。想法：AI导航站那么我就直接找ChatGPT来帮忙首先这是ChatGPT给我生成的代码，你不需要知道它是干什么的，按照ChatGPT的一步步做就可以。它会非常详细的告诉你怎么做我按照ChatGPT教的方法，花了大概10分钟成功的在Github上做了这个网站。这个网站打开后是这样的后续完全可以补上更多的AI工具，以及做的更精美一些。这个过程中可以一分钱不用花，ChatGPT用免费的GPT3.5，Github 完全免费的网站托管功能。并且这个过程中的任何问题都可以问ChatGPT，直到你完全搞好。这种情况，但凡是个人就已经烦了，但是ChatGPT不会，它会永远的非常有礼貌的不紧不慢的给你解决问题。"
    label = {1: "正面情绪", 0: "负面情绪"}

    # 创建模型实例
    model = LSTMModel(input_size=100, hidden_size=50, output_size=2)  # 假设input_size为100（Word2Vec向量的维度），hidden_size和output_size需要根据你的模型定义来调整

    # 加载模型状态字典
    model_state_dict = torch.load('model.pth')
    model.load_state_dict(model_state_dict)

    # 将模型设置为评估模式
    model.eval()

    # 按标点符号分割文本
    sentences = split_text_by_punctuation(input_text)

    for sentence in sentences:
        # 预处理输入数据
        input_data = preprocess_text(sentence)
        # 确保输入词向量与模型维度和数据类型相同
        input_vector = text_to_vector(input_data)  # 注意这里应该处理整个句子得到一个合适的输入形式，当前text_to_vector函数可能不适用，因为它期望一个单词列表
        # 由于LSTM需要序列输入，我们可能需要将单个向量重复或调整以适应模型的输入要求
        # 这里假设我们将单个向量作为序列长度为1的输入（\取决于你的具体实现）
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加batch维度

        # 将输入数据传入模型
        with torch.no_grad():
            output = model(input_tensor)

        # 获取预测的类别
        predicted_class = label[torch.argmax(output, dim=1).item()]
        print(f"Input sentence: {sentence}")
        print(f"模型预测的类别: {predicted_class}")
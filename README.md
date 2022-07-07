# CycleGAN网络实现文本风格迁移-华为昇腾AI创新训练营答辩项目

## 项目介绍

本项目旨在使用CycleGAN模型实现文本风格迁移（Text Style Transfer）任务。模型使用Transformer网络作为生成器，CNN作为判别器。文本风格迁移中的风格与图像风格迁移中的相比，其范围显然要宽泛许多，从词汇分布、组织结构到情感色彩、身份特征等方面的差异都可以认为是文本风格的一部分。因此，文本风格迁移在未来的应用形式和场景非常多样与充满想象力。另外，由于文本相比于图像是离散数据，文本风格迁移在梯度传递方面存在困难。目前，因为缺乏具有强有力数学基础的代表作，该领域发展前景十分广阔。

## 项目目的

+ 熟悉掌握使用MindSpore框架进行深度学习模型的构建和训练
+ 掌握Transformer模型的基本结构和编程方法
+ 掌握使用CycleGAN模型进行中文文本风格迁移
+ 掌握开源项目的发布流程

## 项目环境

+ MindSpore 1.7.0
+ Python 3.8
+ GPU RTX 3090
+ Ubuntu 20.04

## 实验步骤

### 数据准备

+ 选用鲁迅小说集作为目标域，从url下载txt格式文件
+ 选用网络作文作为假样本域
+ 去除下标、脚注和特殊符号等（不包括逗号和句号，作为分隔符）
+ 使用HanLP分词，将文本段落分割为词汇，以空格区分
+ 由于短文本风格不明显，以分隔符起始和结尾，截取符合长度要求的最长文本
+ 将文本按行保存为数据集

### 数据预处理

+ 按照8：2的比例分割为训练数据和测试数据
+ 使用北师大的CWV文学作品数据集将词汇转换为词向量

### 训练网络

训练网络代码梳理概况如下所示：

```
Class CycleGanModel
- Class TransformerGenerator
    - Class TransformerEncoder
        - Class EncoderCell
            - Class SelfAttention
            - Class FeedForward
        - Class DecoderCell
            - Class SelfAttention
            - Class Encoder-Decoder Attention
            - Class FeedForward
    - Class TransformerDecoder
- Class CnnDiscriminator
```

![Text Style Transfer](https://camo.githubusercontent.com/2a04777c752f76cc317eb2a258268f37565646fd990e6f6bcaaa44d06e996333/68747470733a2f2f692e696d6775722e636f6d2f55724c523971532e706e67)





## 数据来源

作文：https://github.com/Arslan-Z/zuowen-dataset-pt1
鲁迅:https://github.com/shengtudai2/LuXunRobot/tree/master/corpus

word2vec模型来源:https://github.com/Embedding/Chinese-Word-Vectors/blob/master/README_zh.md

### 删除特殊字符，

替换一些标点符号

**冒号：** 替换为**逗号，**。

**省略号……**    替换为  **逗号，**

**感叹号 ！ 问号 ？** 替换为 **句号 。**



### 数据过滤

去除特殊符号，替换特殊标点,句子拆分，滑动窗口生成长度小于40的句子集合，对句子集合拆分

### 分词工具

#### HanLP: Han Language Processing

https://github.com/hankcs/HanLP

面向生产环境的多语种自然语言处理工具包，基于Pytorch和TensorFlow 2.x双引擎，目标是普及落地最前沿的NLP技术。HanLP具备功能完善、精度准确、性能高效、语料时新、架构清晰、可自定义的特点。



### Word2Vec

#### Chinese Word Vectors 中文词向量 

为了将文本转换为计算机可以理解的模式，我们使用word2vec将中文词语转换为300维的词向量。

我们使用的word2vec模型是由8599篇现代文学作品训练而成,我们使用的该word2vec使用了N-Gram技术，因此充分考虑了上下文信息。

# 问题：

 ### 数据准备部分：

- 文本预处理时，原文章中的引文序号， 使用正则替换始终无法删除。最后发现该序号**(3)**为单个字符。并非由一对括号和数字表示： **⑶** **(3)**

  **解决办法**: 使用unicode字符表示序号，在原文中进行替换

- 因该使得，正样本（鲁迅文集）和负样本（作文）的大小 接近。

- 正样本与负样本在词汇分布上 应该接近，起初我们想负样本使用新闻，因为新闻格式工整，便于分词和分句，后来发现新闻中含有一些词汇的出现频率极高，例如“地方政府”，“高度评价”等。但选择风格较为接近的现代小说，虽然词汇分布相同，但最终效果可能不近理想，几经考虑之下，我们选择了作文作为负样本，词汇分布与正样本相同，同时文本风格又相差较大。 



### Word2Vec 部分

- 我们使用的word2vec使用的分词使HanLP分词，而我们在处理我们自己的文本的时候发现缺失值极大，后来发现，第一次分词的技术用的使jieba分词，分词后效果发现不好，于是我们查找原因发现使分词算法不同，于是我们选择了与word2vec模型相同的Hanlp。



### mindspore与Pytorch的不同

```
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
mindspore.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros')
```

**MindSpore** 在padding值大于0时，需要**设置**pad_mode='pad'

### 模型构建部分

- transformer模型的编写中，不熟悉的API太多，许多torch和mindspore之间的转换不熟悉。
- 这里我们遇到了两个小问题，一是mindspore无法像pytorch一样集中对神经网络进行参数初始化，所以要在每一层使用init进行初始化操作；而是init中输入shape与全连接层的shape是互为转置的。
- 生成器与编解码器。这里有一点很有趣，当使用特定的激活函数时，求解梯度时会报错，因此，原本打算在这里使用Gumbel Softmax的想法被放弃。另外，即使我重写了construct方法，依然无法通过out=Net(input)的形式搭建网络
- 我们使用了与DCGAN中类似的方式，将求解过程使用类进行了封装。由于前面提到的无法调用实例搭建网络的问题，这里我将nn中TrainOneStepCell类进行了继承和改写。
- 由于原方法中使用了*Attention Is All You Need* 中的 Normal动量优化算子求解梯度下降，我重写了NoamLR方法，将它传入Adam的lr中，实现了对原方法的优化。



### 团队协作

在团队合作中，我们需要不断向对方交付我们写好的代码或者处理好的文件，这时候我们发现Git是一个很好的团队协作工具。








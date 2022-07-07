import numpy as np


# 产生 word2vec 字典  打开文件 返回字典 传入的是 分词地址
# 返回值 是 字典{'词':Tensor}
def produce_word2vec(s):
    with open(s, encoding='utf-8') as file:
        word2vec_dict = {}
        for i in file:
            c = i.split()
            dict_keys = c[0]
            dict_values = np.array(c[1:-1], dtype=np.float32)
            word2vec_dict[dict_keys] = dict_values
    return word2vec_dict


# 返回句子向量
def sentence(path):
    with open(path, encoding='utf-8') as file:
        c = []
        for i in file:
            q = i.split()
            c.append(q)
    return c


# 返回句子向量的word2vec
# 传入 句子向量 和word2vec
# 返回二维列表[[  ]]
def sentence_word2vec(c, real_dict):
    real_c = []
    for i in c:
        real_i = []
        for j in i:
            if j not in real_dict.keys():
                continue
            real_i.append(real_dict.get(j))
        real_c.append(real_i)
    return real_c


def create_dict_iterator(size, batch_size, seq_len, word_size):
    real_dict = produce_word2vec('./dataset/sgns.literature.bigram')
    pos = sentence("./dataset/positive.txt")
    neg = sentence("./dataset/negetive.txt")
    pos_vec = sentence_word2vec(pos, real_dict)
    neg_vec = sentence_word2vec(neg, real_dict)
    len_pos = len(pos)
    len_neg = len(neg)

    for i in range(size):
        Data = []
        Label = np.ones(batch_size)
        for j in range(batch_size):
            idx = np.random.randint(0, len_pos + len_neg)
            if idx < len_pos:
                label = 1.0
                data = np.pad(np.array(pos_vec[idx]),
                              ((0, seq_len - len(pos_vec[idx])),
                               (0, word_size - len(pos_vec[idx][0]))),
                              'constant', constant_values=(0, 0))
            else:
                label = 0.0
                data = np.pad(np.array(neg_vec[idx - len_pos]),
                              ((0, seq_len - len(neg_vec[idx - len_pos])),
                               (0, word_size - len(neg_vec[idx - len_pos][0]))),
                              'constant', constant_values=(0, 0))
            Data.append(data)
            Label[j] = label
        yield {"data": np.array(Data), "label": Label}


"""
data_loader = []
    for i in range(size):
        data = ms.Tensor(np.random.randn(batch_size, seq_len, word_dim), dtype=ms.float32)
        label = ms.Tensor(np.random.randint(0, 2, (batch_size, 1, 1)), dtype=ms.float32)
        data_loader.append({"data":data, "label":label})
"""

if __name__ == '__main__':

    dataloader = create_dict_iterator(10, 10, 40, 300)
    for i in dataloader:
        print(i["data"].shape)

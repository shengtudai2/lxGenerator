import math
import copy
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore import nn
from mindspore.ops import functional as F
from utils import create_dict_iterator 

class Resblock(nn.Cell):
    def __init__(self, inner_dim, kernel_size):
        super(Resblock, self).__init__()
        self.inner_dim = inner_dim
        self.kernel_size = kernel_size
        self.relu = ops.ReLU()
        if kernel_size % 2 != 1:
            raise Exception("kernel size must be odd number!")
        self.conv_1 = nn.Conv1d(self.inner_dim, self.inner_dim, self.kernel_size, pad_mode="pad", padding=int((kernel_size-1)/2))
        self.conv_2 = nn.Conv1d(self.inner_dim, self.inner_dim, self.kernel_size, pad_mode="pad", padding=int((kernel_size-1)/2))

    def construct(self, inputs):
        output = self.relu(inputs)
        output = self.conv_1(output)
        output = self.relu(output)
        output = self.conv_2(output)
        return inputs + (0.3*output)

class Discriminator(nn.Cell):
    def __init__(self, word_dim, inner_dim, seq_len, kernel_size=3, two_out=False):
        super(Discriminator, self).__init__()
        self.word_dim = word_dim
        self.inner_dim = inner_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        if kernel_size % 2 != 1:
            raise Exception("kernel size must be odd number!")
        self.conv_1 = nn.Conv1d(self.word_dim, self.inner_dim, self.kernel_size, pad_mode="pad", padding=int((kernel_size-1)/2))
        self.resblock_1 = Resblock(inner_dim, kernel_size)
        self.resblock_2 = Resblock(inner_dim, kernel_size)
        self.resblock_3 = Resblock(inner_dim, kernel_size)
        self.resblock_4 = Resblock(inner_dim, kernel_size)
        W = seq_len*inner_dim
        self.fc_1 = nn.Dense(W, int(W/8))
        self.fc_2 = nn.Dense(int(W/8), int(W/32))
        self.fc_3 = nn.Dense(int(W/32), int(W/64))
        self.fc_4 = nn.Dense(int(W / 64), 2 if two_out else 1)
        self.relu = nn.LeakyReLU()

    def feed_fc(self, inputs):
        output = self.relu(self.fc_1(inputs))
        output = self.relu(self.fc_2(output))
        output = self.relu(self.fc_3(output))
        return self.fc_4(output)

    def construct(self, inputs):
        this_bs = inputs.shape[0]
        permute = ops.Transpose()
        inputs = ops.Cast()(permute(inputs, (0, 2, 1)), ms.float32)
        if inputs.shape[-1] != self.seq_len:
            # print("Warning: seq_len(%d) != fixed_seq_len(%d), auto-pad."%(inputs.shape[-1], self.seq_len))
            p1d = (0, self.seq_len - inputs.shape[-1])
            inputs = F.pad(inputs, p1d, "constant", 0)
            # print("after padding,", inputs.shape)
        output = self.conv_1(inputs)
        output = self.resblock_1(output)
        output = self.resblock_2(output)
        output = self.resblock_3(output)
        output = self.resblock_4(output)
        output = output.view(this_bs, -1)
        # print(output.shape)
        return self.feed_fc(output)

class TrainOneStepCell(nn.TrainOneStepCell):
    "for class function construct cannot be auto-invoked."
    def construct(self, *inputs):
        loss = self.network.construct(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class WithLossCellD(nn.Cell):
    "connect generator and loss."
    def __init__(self, netD, loss_fn):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.netD = netD
        self.loss_fn = loss_fn

    def construct(self, z, y):
        x = self.netD.construct(z)
        cast = ops.Cast()
        loss = self.loss_fn(x.view(-1), y.view(-1))
        return loss

class CLA(nn.Cell):
    "classifier"
    def __init__(self, myTrainOneStepCellForD):
        super(CLA, self).__init__(auto_prefix=True)
        self.myTrainOneStepCellForD = myTrainOneStepCellForD

    def construct(self, data, label):
        output_D = self.myTrainOneStepCellForD(data, label).view(-1)
        netD_loss = output_D.mean()
        return netD_loss

if __name__ == "__main__":

    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    
    data_root = "./datasets"  # 数据集根目录
    batch_size = 128          # 批量大小
    word_dim = 300            # 词向量大小
    inner_dim = 1024          # 隐藏层大小
    seq_len = 40              # 句子长度
    num_epochs = 10           # 训练周期数
    size = 500                # 数据集大小
    lr = 0.0002               # 学习率
    beta1 = 0.5               # Adam优化器的beta1超参数

    netD = Discriminator(word_dim, inner_dim, seq_len)
    loss = nn.BCELoss(reduction='mean')
    optimizerD = nn.Adam(netD.trainable_params(), learning_rate=lr, beta1=beta1)
    netD_with_criterion = WithLossCellD(netD, loss)
    myTrainOneStepCellForD = TrainOneStepCell(netD_with_criterion, optimizerD)
    
    cla = CLA(myTrainOneStepCellForD)
    cla.set_train()

    # 创建迭代器
    data_loader = create_dict_iterator(size*num_epochs, batch_size, seq_len, word_dim)
    # np.random.seed(1)
    # data_loader = []
    # for i in range(size):
    #     data = ms.Tensor(np.random.randn(batch_size, seq_len, word_dim), dtype=ms.float32)
    #     label = ms.Tensor(np.random.randint(0, 2, (batch_size, 1, 1)), dtype=ms.float32)
    #     data_loader.append({"data":data, "label":label})
    D_losses = []

    # 开始循环训练
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        # 为每轮训练读入数据
        for i in range(size):
            d = next(data_loader)
            data = Tensor(d["data"], ms.float32)
            label = Tensor(d["label"], ms.float32)
            netD_loss = cla.construct(data, label)
            if i % 50 == 0 or i == size - 1:
                # 输出训练记录
                print('[%2d/%d][%3d/%d]   Loss_D:%7.4f' % (
                    epoch + 1, num_epochs, i + 1, size, netD_loss.asnumpy()))
            D_losses.append(netD_loss.asnumpy())

    # 保存网络模型参数为ckpt文件
    ms.save_checkpoint(netD, "Discriminator.ckpt")

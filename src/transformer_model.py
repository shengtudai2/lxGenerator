import math
import copy
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore import nn
from mindspore.common.initializer import initializer, XavierUniform
from mindspore.ops import functional as F
from mindspore.ops import operations as P

# global device
# device = "GPU"
# print(ms.context.get_context("device_target"))

def swap(shape, pos1, pos2):
    list = [pos for pos in range(len(shape))]
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return Tensor.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    matmul = ops.BatchMatMul()
    softmax = ops.Softmax()
    d_k = query.shape[-1]

    scores = matmul(query, key.transpose(swap(key.shape, -2, -1))) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = softmax(scores)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.CellList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Cell):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        weight_init = initializer(XavierUniform(), [d_model, d_model], ms.float32)
        self.linears = clones(nn.Dense(d_model, d_model, weight_init), 4)
        self.attn = None
        self.dropout = nn.Dropout(keep_prob=dropout)
        
    def construct(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            expand_dims = ops.ExpandDims()
            mask = expand_dims(mask, 1)
        nbatches = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view((nbatches, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(swap(x.shape, 1, 2)) \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class LayerNorm(nn.Cell):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = ms.Parameter(Tensor(np.ones(features), ms.float32))
        self.b_2 = ms.Parameter(Tensor(np.zeros(features), ms.float32))
        self.eps = eps

    def construct(self, x):
        mean = x.mean(-1, keep_dims=True)
        std = x.std(-1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Cell):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def construct(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm.construct(x)))

class PositionwiseFeedForward(nn.Cell):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # weight_init for Dense: (out_channels, in_channels)
        weight_init_1 = initializer(XavierUniform(), [d_model, d_ff], ms.float32)
        weight_init_2 = initializer(XavierUniform(), [d_ff, d_model], ms.float32)
        self.w_1 = nn.Dense(d_model, d_ff, weight_init_2)
        self.w_2 = nn.Dense(d_ff, d_model, weight_init_1)
        self.dropout = nn.Dropout(dropout)

    def construct(self, x):
        relu = ops.ReLU()
        return self.w_2(self.dropout(relu(self.w_1(x))))

class EncoderLayer(nn.Cell):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def construct(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0].construct(x, lambda x: self.self_attn.construct(x, x, x, mask))
        return self.sublayer[1].construct(x, self.feed_forward.construct)

class DecoderLayer(nn.Cell):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def construct(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0].construct(x, lambda x: self.self_attn.construct(x, x, x, tgt_mask))
        x = self.sublayer[1].construct(x, lambda x: self.src_attn.construct(x, m, m, src_mask))
        return self.sublayer[2].construct(x, self.feed_forward.construct)
    
class Encoder(nn.Cell):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # layer = EncoderLayer()
        self.norm = LayerNorm(layer.size)
        
    def construct(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer.construct(x, mask)
        return self.norm.construct(x)

class Decoder(nn.Cell):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def construct(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer.construct(x, memory, src_mask, tgt_mask)
        return self.norm.construct(x)

class FullEncoder(nn.Cell):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, src_embed):
        super(FullEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed # Embedding function

    def construct(self, src, src_mask):
        return self.encoder.construct(self.src_embed(src), src_mask)

class Scale(nn.Cell):
    def __init__(self, d_model):
        super(Scale, self).__init__()
        self.d_model = d_model

    def construct(self, x):
        return x * math.sqrt(self.d_model)

class PositionalEncoding(nn.Cell):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings once in log space.
        pe = Tensor(np.zeros((max_len, d_model)), ms.float32)
        expand_dims = ops.ExpandDims()
        cast = ops.Cast()
        exp = ops.Exp()
        sin = ops.Sin()
        cos = ops.Cos()
        position = cast(expand_dims(ms.numpy.arange(0, max_len), 1), ms.float32)
        div_term = exp(cast(ms.numpy.arange(0, d_model, 2), ms.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        self.pe = expand_dims(pe, 0) # this is not trainable parameters
        
    def construct(self, x):
        cast = ops.Cast()
        x = cast(x, ms.float32) + self.pe[:, :x.shape[1]]
        return self.dropout(x)

class Generator(nn.Cell):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        weight_init = initializer(XavierUniform(), [vocab, d_model], ms.float32)
        self.proj = nn.Dense(d_model, vocab, weight_init)
        self.softmax = nn.Softmax()

    """RuntimeError: mindspore/ccsrc/plugin/device/gpu/kernel/nn/softmax_grad_gpu_kernel.h:124 Init] For 'LogSoftmaxGrad', the dimension of input should be equal to 2, but got 3"""
    def _construct(self, x):
        log_softmax = nn.LogSoftmax()
        return log_softmax(self.proj(x))
        
    def construct(self, x, scale=1.0):
        return self.softmax(self.proj(x)*scale)

class EncoderDecoder(nn.Cell):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # Embedding function
        self.tgt_embed = tgt_embed  # Embedding function
        self.generator = generator
        
    def construct(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder.construct(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder.construct(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def Transformer(N=6, d_model=1024, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.SequentialCell(Scale(d_model), c(position)),
        nn.SequentialCell(Scale(d_model), c(position)), None)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    # We do this in each Dense Layer.
    return model

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        expand_dims = ops.ExpandDims()
        self.src_mask = expand_dims((src != pad), -2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        expand_dims = ops.ExpandDims()
        tgt_mask = expand_dims((tgt != pad), -2)
        logical_and = ops.LogicalAnd()
        tgt_mask = logical_and(tgt_mask, Tensor(subsequent_mask(tgt.shape[-1]), tgt_mask.dtype))
        return tgt_mask

class TrainOneStepCell(nn.TrainOneStepCell):
    "for class function construct cannot be auto-invoked."
    def construct(self, *inputs):
        loss = self.network.construct(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class WithLossGenerator(nn.Cell):
    "connect generator and loss."
    def __init__(self, generator, criterion):
        super(WithLossGenerator, self).__init__(auto_prefix=True)
        self.netG = generator
        self.loss_fn = criterion

    def construct(self, z, y, norm):
        x = self.netG.construct(z)
        cast = ops.Cast()
        print(x.view(-1, x.shape[-1]), y.view(-1))
        loss = self.loss_fn(x.view(-1), y.view(-1)) \
                / cast(norm, ms.float32)
        return loss

class SimpleLossCompute(nn.Cell):
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, optimizer):
        super(SimpleLossCompute, self).__init__()
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer
        self.netG_with_criterion = WithLossGenerator(generator,
                                                     criterion)
        self.myTrainOneStepForG = TrainOneStepCell(
                                        self.netG_with_criterion,
                                        self.optimizer)
        
    def __call__(self, z, y, norm):
        output_G = self.myTrainOneStepForG(z, y, norm).view(-1)
        cast = ops.Cast()
        netG_loss = output_G * cast(norm, ms.float32)
        return netG_loss.mean()

def NoamLR(model_size, factor, warmup, total_step):
    lr = []
    for step in range(total_step):
        step += 1
        lr.append(factor * model_size ** (-0.5) * \
            min(step ** (-0.5), step * warmup ** (-1.5)))
    return lr

def greedy_decode(model, embed, src, src_mask, max_len, start_symbol=2):
    memory = model.encode(embed(src), src_mask)
    fill = ops.Fill()
    ys = fill(src.dtype, (src.shape[0], 1), start_symbol)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(embed(ys)), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
    return ys


"""
from mindspore.nn import learning_rate_schedule as lr_schedules

d_model = 32
vocab = 1024
netG = Generator(d_model, vocab)
criterion = nn.BCELoss(reduction='mean')
lr = 0.0002
beta1 = 0.5

lr = NoamLR(10, 1, 10, 100)
optG = nn.Adam(params=netG.trainable_params(), learning_rate=lr, beta1=beta1)
# optG = nn.Adam(netG.trainable_params(), learning_rate=lr, beta1=beta1)

sls = SimpleLossCompute(netG, criterion, optG)
sls.set_train()

z = Tensor(np.ones((1, 40, d_model)), ms.float32)
y = Tensor(np.zeros((1, 40, vocab)), ms.float32)
norm = Tensor([1], ms.float32)

loss = sls(z, y, norm)
print(loss)
"""
# batch = Batch(Tensor(np.ones((5, 6)), ms.float32), Tensor(np.ones((5, 6)), ms.float32))
# tf = Transformer()
"""
n_batch = 1
b_size = 1
n_head = 5
d_model = 20
d_ff = 32
size = (n_batch, b_size, n_head, d_model)
x = Tensor(np.ones(size), ms.float32)
mask = subsequent_mask(b_size)
attn = MultiHeadedAttention(n_head, d_model)
pffw = PositionwiseFeedForward(d_model, d_ff) 
encoder_layer = EncoderLayer(size, attn, pffw, 0.5)
encoder = Encoder(encoder_layer, 2)
print(encoder.construct(x, mask))
# print(encode.construct(x, mask))
# satn = MultiHeadedAttention(n_head, d_model)
# decode = DecoderLayer(size, attn, satn, pffw, 0.5)
# print(decode.construct(x, x, mask, mask))
"""

import torch
from torch import nn
from d2l import torch as d2l

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    # 参数列表中很好的初始化方法
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        # 回忆 nn.Embedding(vocab_size, num_hiddens) 就是可以容纳的词表数目 + 我要生成的词向量维度
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # 在segment中只有0或者1这两个映射 num_hiddens 不是序列的长度而是同每一个元素嵌入后的元素的维度一致
        # 这里就注意了这个就是陈铮老师所说的我们不是直接加的0或者是1而是直接加上了0或者1这两个编号的embedding
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        # 这个是多个 Transformer 的block，下面的代码就是将Transformer那边的代码搬过来
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        # max_len所有的输入都要加上这个，Parameter得意思就是这些初始化的位置编码也能够当做是参数来学习
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))
    # valid_lens 句子有多少是合法的 实际上也就是我们首先是定义了一个大网络然后在实际训练的
    # 时候我们只是输入了有效的句子
    '''
        一些解释
        这部分代码将X传递给一系列的“blocks”。在BERT的上下文中，一个“block”通常是一个Transformer编码器。
        每个块都会更新X。valid_lens可能用于指定序列中有效的长度，这在处理填充（padding）的时候尤其重要。
    '''
    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        # 加上Embedding这件事情实际上就是我们的原始数据是扰动的，我们加上embedding之后
        # 数据之间会有很明显的断层
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        # 将参数加入到block中，就是一层一层网上传最后forward的输出就是这个X
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X



# 给定一个词抽出一个 num_hiddens 维度的向量出来
# 这段代码的一些细节还是需要看 Transformer
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)


# 在这里随便跑一个forward path
# batch_size = 2  length = 8
tokens = torch.randint(0, vocab_size, (2, 8))
# 构造两个句子 每个句子长度为8
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
# 最后的输出结果是 batch_size * length * num_hiddens 相当于embedding每个词到768维
encoded_X = encoder(tokens, segments, None)
encoded_X.shape


# 上面只是定义了BertEncoder 而整个Bert是由多个模块组成 定义了任务
#@save
class MaskLM(nn.Module):
    # 你发现了没前面定义了一个forward path实际上已经是model了
    # 后面的训练加上损失函数的定义就构成了整个模型的训练
    """BERT的掩蔽语言模型任务"""
    # 在Encoder后面加了一个MLP
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))
    # X：Encoder已经输出的向量  pred_positions：我们需要预测的位置
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）也就是2个数字每一次重复3次
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        # masked_X 就是我们将对应的位置遮起来之后的输出
        masked_X = X[batch_idx, pred_positions]
        # 上面的案例中2 * 3 * dim就是我们得到的
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        # 将mask掉的向量输入mlp进行预测  最后的768正好与mlp的输入层对齐了
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat# 上面只是定义了BertEncoder 而整个Bert是由多个模块组成 定义了任务
#@save
class MaskLM(nn.Module):
    # 你发现了没前面定义了一个forward path实际上已经是model了
    # 后面的训练加上损失函数的定义就构成了整个模型的训练
    """BERT的掩蔽语言模型任务"""
    # 在Encoder后面加了一个MLP
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))
    # X：Encoder已经输出的向量  pred_positions：我们需要预测的位置
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）也就是2个数字每一次重复3次
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        # masked_X 就是我们将对应的位置遮起来之后的输出
        masked_X = X[batch_idx, pred_positions]
        # 上面的案例中2 * 3 * dim就是我们得到的
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        # 将mask掉的向量输入mlp进行预测  最后的768正好与mlp的输入层对齐了
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

batch_idx = torch.arange(0, 2)
# 假设batch_size=2，num_pred_positions=3
# 那么batch_idx是np.array（[0,0,0,1,1,1]）
batch_idx = torch.repeat_interleave(batch_idx, 3)
batch_idx

'''
    原始的代码是这样的encoded_X[[0,0,0,1,1,1],[[1, 5, 2], [6, 1, 5]]]
    上面有一个拍平的操作变成了 encoded_X[[0,0,0,1,1,1],[1, 5, 2, 6, 1, 5]]
    实际上这个和word2vec中的思想一样掩码的思想但实际上这一次又巧妙的使用了坐标的方法
    比如我们要抽取两个句子中的[[1, 5, 2], [6, 1, 5]]元素则[0,1][0,5]and so on就是我们想要的
'''
encoded_X[[0,0,0,1,1,1],[1, 5, 2, 6, 1, 5]]
'''
    原始的代码是这样的encoded_X[[0,0,0,1,1,1],[[1, 5, 2], [6, 1, 5]]]
    上面有一个拍平的操作变成了 encoded_X[[0,0,0,1,1,1],[1, 5, 2, 6, 1, 5]]
    实际上这个和word2vec中的思想一样掩码的思想但实际上这一次又巧妙的使用了坐标的方法
    比如我们要抽取两个句子中的[[1, 5, 2], [6, 1, 5]]元素则[0,1][0,5]and so on就是我们想要的
'''
encoded_X[[0,0,0,1,1,1],[1, 5, 2, 6, 1, 5]]


encoded_X.shape

mlm = MaskLM(vocab_size, num_hiddens)
# 两个句子 第一个mask掉[1, 5, 2] 第二个mask掉[6, 1, 5] 这些都是所谓的需要预测词
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
# 最后的输出 batch_size * 每个句子预测3个词 * 每个词的情况有10000种
# 也就是刚才的案例预测了6个词
mlm_Y_hat.shape

mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
# 6行 * 10000列  vs 6 这样的话就能将预测与label对齐了
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape

mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
mlm_Y.reshape(-1).shape


#@save
# 面向对象程序设计所有的Moudle都会被定义成一个类
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    # 这里仍然套用了softMax所以搞了两个输出而不是logistic的0或者1
    # 可以传入函数式参数与可变长参数
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        # woc  只有一个线性层玩个毛线？？
        # 这里的num_inputs就是一个句子向量的维度
        self.output = nn.Linear(num_inputs, 2)
    # 老是对batch感到陌生这里的batchsize可以理解为当前批的样本数量
    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)#@save
# 面向对象程序设计所有的Moudle都会被定义成一个类
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    # 这里仍然套用了softMax所以搞了两个输出而不是logistic的0或者1
    # 可以传入函数式参数与可变长参数
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        # woc  只有一个线性层玩个毛线？？
        # 这里的num_inputs就是一个句子向量的维度
        self.output = nn.Linear(num_inputs, 2)
    # 老是对batch感到陌生这里的batchsize可以理解为当前批的样本数量
    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)

# torch.flatten(encoded_X, start_dim=1) 从第1个维度之后进行拍平
# 2 * 8 * 768 => 2 * 6144 实际上就是直接将一个句子直接考虑一条向量
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape# torch.flatten(encoded_X, start_dim=1) 从第1个维度之后进行拍平
# 2 * 8 * 768 => 2 * 6144 实际上就是直接将一个句子直接考虑一条向量
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape


encoded_X = torch.flatten(encoded_X, start_dim=1)
encoded_X.shape

nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape

nsp_Y_hat,nsp_l

#@save
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat#@save
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


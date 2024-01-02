# the goal of this program to build a vocabulary
import re # regex
import yaml
import os
import collections
# 第一步是要读取配置
with open(r'../config.yml','r') as config_file:
    config = yaml.safe_load(config_file)


# 第一步：读取数据与数据清洗
def read_sentences(filePath):
    with open(filePath, 'r') as f:
        # 这个方法因该是可以一次读取很多行
        lines = f.readlines()
    # 善用推导表达式
    # 先缩减数据然后再去做统一变形，将所有的不在range的东西都替换成' '注意不是空字符
    return [re.sub('[^A-Za-z]',' ',line).strip().lower() for line in lines]

curPath = os.getcwd()
# join的路径是认识相对路径的
# filePath = os.path.join(curPath, config.get('file_path'))
# print(filePath)
# lines = read_sentences(filePath)
# print(lines[:10])
# print(len(lines))

# 第二部：tokenize 目标就是将句子变成二维数组
def tokenize(lines):
    # s.strip() 来检查是否剩余非空字符
    # 下面的这种写法是创的写法但不是标准的写法，下面的vocab中的token的写法是正统的写法
    return [[string for string in line.split() if string.strip()] for line in lines]

# print(tokenize(lines)[:10])


# 第三部构建词表
# 具体的特征与设计如下
# 模型需要的是数字而不是字符串所以词表需要构建一个编号到词词到编号的双向映射
# 并非所有的token都要参与训练，一般情况可以取频率高的部分训练
# 频率越高的越可能被频繁用到这个时候将频率高的数据放在一块就可以充分利用这个局部性原理
# 最后，词表是一个类

class Vocab:
    # 下面是构造器
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # 第零步鲁棒性检测
        if tokens is None:
            tokens = []
        # 这里面需要注意的是需要有一个保留的字符集合，比如未登录词，注意这里是reserved不是reverse
        if reserved_tokens is None:
            reserved_tokens = []
        # 第一步tokens计数并且倒序排列
        counter = count_corpus(tokens)
        # 直接定义property，_token_fres存放了{token : freq}
        self._token_fres =  sorted(counter.items(), key=lambda x : x[1], reverse=True)
        # 这里只是定义了这个数组还没有开始初始化
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 这里最后返回了一个字典，上面的 index_to_token 数组天然可以实现下标到token的索引
        self.token_to_index = {token : idx for idx, token in enumerate(self.idx_to_token)}
        # 依然没有读懂他的意思
        self.idx_to_token,self.token_to_index = [], dict()
        # 重建词表是因为要去除低频率的词汇
        for token,freq in self._token_fres:
            # 小于频次的直接cut
            if freq < min_freq:
                break
            # 为什么要使用self.token_to_index作为判别 是因为他是一个dict()
            if token not in self.token_to_index:
                self.idx_to_token.append(token)
                self.token_to_index[token] = len(self.idx_to_token) - 1
    # 类中的get方法都是使用魔法方法

    # 获得长度
    def __len__(self):
        return len(self.idx_to_token)

    # 注意这个方法  获得某一个元素使用递归的方法
    # 目标是根据tokens返回对应的idx，设置unk的作用是我们在手工输入token的时候难免会有别的未登录词
    def __getitem__(self, tokens):
        # 这个条件判断语句检查输入的 tokens 是否为列表或元组 这个才是真实的意思
        if not isinstance(tokens, (list,tuple)):
            # 这个因该是集合中的固有方法，先get(tokens)有的话得到下标没有的话返回0
            # 最后应该是把所有的获得的元素append了起来
            return self.token_to_index.get(tokens,self.unk)
        return [self.__getitem__(self, token) for token in tokens]

    # 还有一个就是根据多个下标返回对应的tokens
    def to_tokens(self, indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[idx] for idx in indices]

    # 定义未知词元
    @property
    def unk(self):
        return 0

    @property
    # 获得排序后的集合
    # @property 可以用于创建只读属性，这样可以在不改变类接口的情况下向类添加属性的访问器方法。
    def token_freq(self):
        return self._token_fres

# 统计词元，这是整个构建词表的第一步
def count_corpus(tokens):
    # 鲁棒性检测不能少
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 注意这里的数组可以是1 or 2维的
        tokens = [token for line in tokens for token in line]
    # 返回的是一个字典
    return collections.Counter(tokens)
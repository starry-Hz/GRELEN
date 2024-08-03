import torch.nn as nn  # 导入 PyTorch 的神经网络模块，提供了构建神经网络所需的功能
import sys  # 导入系统模块，用于操作系统相关功能
sys.path.append('..')  # 将上级目录添加到系统路径中，以便导入其他模块
from lib.utils import *  # 从 lib.utils 模块中导入所有功能，用于后续代码中的工具函数

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    # 定义一个具有两个全连接层、ELU 激活函数和批量归一化的多层感知机模型

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
        初始化 MLP 模型
        :param n_in: 输入层的节点数
        :param n_hid: 隐藏层的节点数
        :param n_out: 输出层的节点数
        :param do_prob: dropout 的概率，默认值为 0（即不使用 dropout）
        """
        super(MLP, self).__init__()  # 调用父类（nn.Module）的初始化函数
        self.fc1 = nn.Linear(n_in, n_hid)  # 定义第一个全连接层，输入维度为 n_in，输出维度为 n_hid
        self.fc2 = nn.Linear(n_hid, n_out)  # 定义第二个全连接层，输入维度为 n_hid，输出维度为 n_out
        self.bn = nn.BatchNorm1d(n_out)  # 定义批量归一化层，用于规范化输出层的输出
        self.dropout_prob = do_prob  # 保存 dropout 概率

        self.init_weights()  # 调用初始化权重的方法

    def init_weights(self):
        """
        初始化权重和偏置
        """
        for m in self.modules():  # 遍历模型中的所有子模块
            if isinstance(m, nn.Linear):  # 如果子模块是全连接层
                nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 正态分布初始化权重
                m.bias.data.fill_(0.1)  # 将偏置初始化为 0.1
            elif isinstance(m, nn.BatchNorm1d):  # 如果子模块是批量归一化层
                m.weight.data.fill_(1)  # 将权重初始化为 1
                m.bias.data.zero_()  # 将偏置初始化为 0

    def batch_norm(self, inputs):
        """
        批量归一化
        :param inputs: 输入数据
        :return: 批量归一化后的数据
        """
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)  # 将输入数据重塑为二维形式
        x = self.bn(x)  # 应用批量归一化
        return x.view(inputs.size(0), inputs.size(1), -1)  # 将数据重塑回原始的三维形式

    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据
        :return: 模型的输出
        """
        x = F.elu(self.fc1(inputs))  # 通过第一个全连接层，并使用 ELU 激活函数
        x = F.dropout(x, self.dropout_prob, training=self.training)  # 应用 dropout
        x = F.elu(self.fc2(x))  # 通过第二个全连接层，并使用 ELU 激活函数
        return self.batch_norm(x)  # 返回批量归一化后的输出

class Graph_learner(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, do_prob=0.):
        """
        初始化图学习模型
        :param n_in: 输入维度
        :param n_hid: 隐藏层维度
        :param n_head_dim: 每个头的维度
        :param head: 头的数量
        :param do_prob: dropout 的概率
        """
        super(Graph_learner, self).__init__()
        self.n_hid = n_hid  # 隐藏层的维度
        self.head = head  # 头的数量
        self.n_in = n_in  # 输入维度
        self.n_head_dim = n_head_dim  # 每个头的维度

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)  # 定义一个 MLP 模型，用于处理输入数据
        self.Wq = nn.Linear(n_hid, n_head_dim * head)  # 定义查询权重矩阵，维度为 n_hid 到 n_head_dim * head
        self.Wk = nn.Linear(n_hid, n_head_dim * head)  # 定义键权重矩阵，维度为 n_hid 到 n_head_dim * head
        for m in [self.Wq, self.Wk]:  # 对权重矩阵进行初始化
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 正态分布初始化权重
                m.bias.data.fill_(0.1)  # 将偏置初始化为 0.1

    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据，形状为 [B, N, T]（批量大小，节点数量，特征数量）
        :return: 注意力权重矩阵
        """
        X = self.mlp1(inputs)  # 通过 MLP 模型处理输入数据
        Xq = self.Wq(X)  # 计算查询向量
        Xk = self.Wk(X)  # 计算键向量
        B, N, n_hid = Xq.shape  # 获取批量大小、节点数量和隐藏层维度
        Xq = Xq.view(B, N, self.head, self.n_head_dim)  # 重塑查询向量，形状为 [B, N, head, n_head_dim]
        Xk = Xk.view(B, N, self.head, self.n_head_dim)  # 重塑键向量，形状为 [B, N, head, n_head_dim]
        Xq = Xq.permute(0, 2, 1, 3)  # 调整维度顺序，形状为 [B, head, N, n_head_dim]
        Xk = Xk.permute(0, 2, 1, 3)  # 调整维度顺序，形状为 [B, head, N, n_head_dim]
        probs = torch.matmul(Xq, Xk.transpose(-1, -2))  # 计算注意力权重矩阵
        return probs

class DCGRUCell_(torch.nn.Module):
    def __init__(self, device, num_units, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """
        初始化 DCGRU 单元
        :param device: 设备（CPU 或 GPU）
        :param num_units: 单元数
        :param max_diffusion_step: 最大扩散步数
        :param num_nodes: 节点数量
        :param nonlinearity: 非线性激活函数（'tanh' 或 'relu'）
        :param filter_type: 图卷积滤波器类型（'laplacian' 或其他）
        :param use_gc_for_ru: 是否使用图卷积来计算更新和重置门
        """
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu  # 根据 nonlinearity 选择激活函数
        self.device = device  # 设备
        self._num_nodes = num_nodes  # 节点数量
        self._num_units = num_units  # 单元数
        self._max_diffusion_step = max_diffusion_step  # 最大扩散步数
        self._supports = []  # 支持的图卷积
        self._use_gc_for_ru = use_gc_for_ru  # 是否使用图卷积计算更新和重置门

        # 定义用于图卷积的线性层
        self._gconv_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)
        self._gconv_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)
        self._gconv_c_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        self._gconv_c_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        for m in self.modules():  # 遍历模型中的所有子模块
            if isinstance(m, nn.Linear):  # 如果子模块是全连接层
                nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 正态分布初始化权重
                m.bias.data.fill_(0.1)  # 将偏置初始化为 0.1

    def forward(self, inputs, hx, adj):
        """
        前向传播
        :param inputs: 输入数据
        :param hx: 隐藏状态
        :param adj: 图的邻接矩阵
        :return: 更新后的隐藏状态
        """
        value = self._gconv_0(inputs)  # 通过图卷积计算值
        r = value[:, :self._num_units]  # 获取重置门
        u = value[:, self._num_units:]  # 获取更新门

        reset_hx = r * hx  # 计算重置后的隐藏状态
        if self._use_gc_for_ru:
            new_hx = self._activation(self._gconv_c_0(reset_hx, adj))  # 通过图卷积计算新的隐藏状态
        else:
            new_hx = self._activation(self._gconv_c_1(reset_hx))  # 通过全连接层计算新的隐藏状态
        hx = (1 - u) * hx + u * new_hx  # 更新隐藏状态

        return hx

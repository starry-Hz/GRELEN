import torch.nn as nn  # 导入 PyTorch 的神经网络模块，提供了构建神经网络所需的功能
import sys  # 导入系统模块，用于操作系统相关功能
sys.path.append('..')  # 将上级目录添加到系统路径中，以便导入其他模块
from lib.utils import *  # 从 lib.utils 模块中导入所有功能，用于后续代码中的工具函数

# 定义一个多层感知机模型（MLP）
"""
多层感知机(MLP,Multilayer Perceptron)也叫人工神经网络(ANN,Artificial Neural Network)
Figure3中的Feature Extraction(提取特征)
***输入时间序列数据通过MLP类进行特征提取***
多个时间序列S通过特征提取转换为隐含表示h

相关代码：
X = self.mlp1(inputs)  # 使用 MLP 提取特征Graph_learner类中的forward函数
"""
class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    # 定义一个具有两个全连接层、ELU(指数线性单元) 激活函数和批量归一化的多层感知机模型

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
        初始化 MLP 模型
        :param n_in: 输入层的节点数
        :param n_hid: 隐藏层的节点数
        :param n_out: 输出层的节点数
        :param do_prob: dropout 的概率，默认值为 0(即不使用 dropout)
        """
        super(MLP, self).__init__()  # 调用父类（nn.Module）的初始化函数
        self.fc1 = nn.Linear(n_in, n_hid)  # 定义第一个全连接层，输入维度为 n_in，输出维度为 n_hid
        self.fc2 = nn.Linear(n_hid, n_out)  # 定义第二个全连接层，输入维度为 n_hid，输出维度为 n_out
        self.bn = nn.BatchNorm1d(n_out)  # 定义批量归一化层，用于规范化输出层的输出
        self.dropout_prob = do_prob  # 保存 dropout 概率
        # dropout 是一种防止神经网络过拟合的正则化技术。基本思想：在每次训练过程中，随机丢弃一部分神经元，迫使神经网络不依赖某些特定的节点和路径，增强模型的泛化能力

        self.init_weights()  # 调用初始化权重的方法

    def init_weights(self):
        """
        初始化模型中所有全连接层和批量归一化层的权重和偏置
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

# 定义一个图学习模型
"""
Figure3中的Relation Inference(关系推断)部分,显示了如何从提取的特征h推断出节点之间的关系𝜃

在输入数据中学习节点之间的关系
1.特征提取:通过MLP模型对输入数据进行特征提取,将其转换为更高维的隐含表示。   代码中的 self.mlp1 部分
2.计算查询和键:通过'wq'和'wk'线性层计算查询和键向量。   代码中的 self.Wq(X) 和 self.Wk(X) 部分。
3.关系推断:通过查询向量和键向量的点积计算注意力权重矩阵。通过向量之间的点积来推断节点之间的关系。点积结果表示节点之间的相似度和关联程度,即注意力权重。
代码中的 torch.matmul(Xq, Xk.transpose(-1, -2)) 部分，对应图中的关系推断部分。
"""
class Graph_learner(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, do_prob=0.):  # n_in = T
        """
        初始化图学习模型
        :param n_in: 输入维度（特征数量 T)
        :param n_hid: 隐藏层维度（用于 MLP 的隐藏层大小）
        :param n_head_dim: 每个注意力头的维度
        :param head: 注意力头的数量
        :param do_prob: dropout 的概率（默认值为 0)
        """
        super(Graph_learner, self).__init__()
        self.n_hid = n_hid  # 隐藏层的维度
        self.head = head  # 头的数量
        self.n_in = n_in  # 输入维度
        self.n_head_dim = n_head_dim  # 每个头的维度

        # 定义一个多层感知机（MLP）用于特征提取
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)  # 定义一个 MLP 模型，用于处理输入数据
        
        # Wq 和 Wk 是用于计算查询（Query）和键（Key）的线性变换层
        # 输出维度 n_head_dim * head，即多头注意力机制中每个头的维度乘以头的数量
        self.Wq = nn.Linear(n_hid, n_head_dim * head)  # 定义查询权重矩阵，维度为 n_hid 到 n_head_dim * head
        self.Wk = nn.Linear(n_hid, n_head_dim * head)  # 定义键权重矩阵，维度为 n_hid 到 n_head_dim * head

        for m in [self.Wq, self.Wk]:  # 对权重矩阵进行初始化
            if isinstance(m, nn.Linear):  # 如果子模块是全连接层
                nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 正态分布初始化权重
                m.bias.data.fill_(0.1)  # 将偏置初始化为 0.1

    def forward(self, inputs):  # inputs: [B, N, T(features)]
        """
        前向传播
        :param inputs: 输入数据，形状为 [B, N, T]（批量大小，节点数量，特征数量）
        :return: 注意力权重矩阵
        """
        X = self.mlp1(inputs)  # 通过 MLP 模型处理输入数据
        Xq = self.Wq(X)  # 计算查询向量
        Xk = self.Wk(X)  # 计算键向量

        # 获取输入的维度信息
        B, N, n_hid = Xq.shape  # 获取批量大小 B，节点数量 N，隐藏层维度 n_hid

        # 调整查询和键向量的形状以适应多头注意力机制
        Xq = Xq.view(B, N, self.head, self.n_head_dim)  # 重塑查询向量，形状为 [B, N, head, head_dim]
        Xk = Xk.view(B, N, self.head, self.n_head_dim)  # 重塑键向量，形状为 [B, N, head, head_dim]

        # 调整维度顺序，便于后续的矩阵乘法操作
        Xq = Xq.permute(0, 2, 1, 3)  # 调整维度顺序，形状为 [B, head, N, head_dim]
        Xk = Xk.permute(0, 2, 1, 3)  # 调整维度顺序，形状为 [B, head, N, head_dim]

        # 计算注意力权重矩阵，使用矩阵乘法将查询向量和键向量相乘，并对最后两个维度进行转置
        probs = torch.matmul(Xq, Xk.transpose(-1, -2))  # 计算注意力权重矩阵    Relation Inference,Figure3中的关系推断
        return probs # 返回注意力权重矩阵

# 定义一个带有图卷积操作的 GRU 单元（DCGRU 单元）***Decoder***
"""
Figure3中的Decoder部分,展示了如何通过系列重建模块将潜在向量Z转换回时间序列数据,并使用学习到的图结重建数据
从时空数据中提取特征，计算更新和重置门，并更新隐藏状态，通过图卷积实现时空依赖关系的捕捉和建模。
"""
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
        super().__init__()  # 调用父类的初始化方法
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu  # 根据 nonlinearity 选择激活函数
        self.device = device  # 设置设备
        self._num_nodes = num_nodes  # 设置节点数量
        self._num_units = num_units  # 设置单元数量
        self._max_diffusion_step = max_diffusion_step  # 设置最大扩散步数
        self._supports = []  # 初始化支持的图卷积列表
        self._use_gc_for_ru = use_gc_for_ru  # 设置是否使用图卷积计算更新和重置门

        # 定义用于图卷积的线性层
        self._gconv_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)  # 定义第一个图卷积层
        self._gconv_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)  # 定义第二个图卷积层
        self._gconv_c_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)  # 定义第三个图卷积层，用于计算新的隐藏状态
        self._gconv_c_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)  # 定义第四个图卷积层，用于计算新的隐藏状态
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
        output_size = 2 * self._num_units  # 输出大小为单元数量的两倍
        if self._use_gc_for_ru:
            fn = self._gconv  # 如果使用图卷积计算更新和重置门，则使用图卷积函数
        else:
            fn = self._fc  # 否则使用全连接函数
        value = torch.sigmoid(fn(inputs, adj, hx, output_size, bias_start=1.0))  # 计算更新和重置门的值

        value = torch.reshape(value, (-1, self._num_nodes, output_size))  # 重塑值的形状
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)  # 分割更新和重置门的值
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))  # 重塑重置门的值
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))  # 重塑更新门的值

        c = self._gconv_c(inputs, adj, r * hx, self._num_units)  # 通过图卷积计算新的隐藏状态
        if self._activation is not None:
            c = self._activation(c)  # 应用激活函数

        new_state = u * hx + (1.0 - u) * c  # 计算新的隐藏状态
        return new_state

    @staticmethod
    def _build_sparse_matrix(L):
        """
        构建稀疏矩阵
        :param L: 输入矩阵
        :return: 构建后的稀疏矩阵
        """
        L = L.tocoo()  # 将矩阵转换为 COOrdinate 格式
        indices = np.column_stack((L.row, L.col))  # 获取矩阵的行列索引
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]  # 按行列排序索引
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)  # 构建稀疏张量
        return L

    def _calculate_random_walk_matrix(self, adj_mx):
        """
        计算随机游走矩阵
        :param adj_mx: 邻接矩阵
        :return: 随机游走矩阵
        """
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(self.device)  # 在邻接矩阵上加单位矩阵
        d = torch.sum(adj_mx, 1)  # 计算每个节点的度
        d_inv = 1. / d  # 计算度的倒数
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)  # 处理无穷大的情况
        d_mat_inv = torch.diag(d_inv)  # 构建度的倒数对角矩阵
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)  # 计算随机游走矩阵
        return random_walk_mx

    def _calculate_random_walk0(self, adj_mx, B):
        """
        计算随机游走矩阵，适用于批量操作
        :param adj_mx: 邻接矩阵
        :param B: 批量大小
        :return: 随机游走矩阵
        """
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).repeat(B, 1, 1).to(self.device)  # 在邻接矩阵上加单位矩阵，并扩展为批量大小
        d = torch.sum(adj_mx, 1)  # 计算每个节点的度
        d_inv = 1. / d  # 计算度的倒数
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)  # 处理无穷大的情况
        d_mat_inv = torch.diag_embed(d_inv)  # 构建度的倒数对角矩阵
        random_walk_mx = torch.matmul(d_mat_inv, adj_mx)  # 计算随机游走矩阵
        return random_walk_mx

    @staticmethod
    def _concat(x, x_):
        """
        连接两个张量
        :param x: 张量 x
        :param x_: 张量 x_
        :return: 连接后的张量
        """
        x_ = x_.unsqueeze(0)  # 在第一个维度上增加一个维度
        return torch.cat([x, x_], dim=0)  # 在第一个维度上连接两个张量

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        """
        全连接层操作
        :param inputs: 输入数据
        :param state: 隐藏状态
        :param output_size: 输出大小
        :param bias_start: 偏置的初始值
        :return: 计算后的值
        """
        batch_size = inputs.shape[0]  # 获取批量大小
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))  # 重塑输入数据的形状
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))  # 重塑隐藏状态的形状
        inputs_and_state = torch.cat([inputs, state], dim=-1)  # 连接输入数据和隐藏状态
        input_size = inputs_and_state.shape[-1]  # 获取输入大小
        weights = self._fc_params.get_weights((input_size, output_size))  # 获取全连接层的权重
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))  # 计算全连接层的输出，并应用 sigmoid 函数
        biases = self._fc_params.get_biases(output_size, bias_start)  # 获取全连接层的偏置
        value += biases  # 加上偏置
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        """
        图卷积操作
        :param inputs: 输入数据
        :param adj_mx: 邻接矩阵
        :param state: 隐藏状态
        :param output_size: 输出大小
        :param bias_start: 偏置的初始值
        :return: 计算后的值
        """
        B = inputs.shape[0]  # 获取批量大小
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)  # 计算随机游走矩阵
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)  # 计算转置后的随机游走矩阵

        batch_size = inputs.shape[0]  # 获取批量大小
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))  # 重塑输入数据的形状
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))  # 重塑隐藏状态的形状
        inputs_and_state = torch.cat([inputs, state], dim=2)  # 连接输入数据和隐藏状态
        input_size = inputs_and_state.size(2)  # 获取输入大小

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)  # 在第一个维度上增加一个维度
        x1_0 = torch.unsqueeze(x, 0)  # 在第一个维度上增加一个维度

        if self._max_diffusion_step == 0:  # 如果最大扩散步数为0
            pass  # 不进行扩散
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)  # 计算扩散步数为1的图卷积
            x1_1 = torch.matmul(adj_mx1, x1_0)  # 计算转置后的扩散步数为1的图卷积
            x0_0 = torch.cat([x0_0, x0_1], dim=0)  # 连接扩散步数为0和1的图卷积结果
            x1_0 = torch.cat([x1_0, x1_1], dim=0)  # 连接转置后的扩散步数为0和1的图卷积结果

            for k in range(2, self._max_diffusion_step + 1):  # 计算更大扩散步数的图卷积
                x0_2 = torch.matmul(adj_mx0, x0_1)  # 计算扩散步数为k的图卷积
                x1_2 = torch.matmul(adj_mx1, x1_1)  # 计算转置后的扩散步数为k的图卷积
                x0_0 = torch.cat([x0_0, x0_1], dim=0)  # 连接扩散步数为0到k的图卷积结果
                x1_0 = torch.cat([x1_0, x1_1], dim=0)  # 连接转置后的扩散步数为0到k的图卷积结果
                x0_1 = x0_2  # 更新扩散步数为k的图卷积结果
                x1_1 = x1_2  # 更新转置后的扩散步数为k的图卷积结果

        num_matrices = self._max_diffusion_step + 1  # 确定图卷积矩阵的数量
        x0_0 = x0_0.permute(1, 2, 3, 0)  # 调整图卷积结果的维度
        x1_0 = x1_0.permute(1, 2, 3, 0)  # 调整转置后的图卷积结果的维度
        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])  # 重塑图卷积结果的形状
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])  # 重塑转置后的图卷积结果的形状
        x0_0 = self._gconv_0(x0_0)  # 计算图卷积的输出
        x1_0 = self._gconv_1(x1_0)  # 计算转置后图卷积的输出

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])  # 返回图卷积的输出

    def _gconv_c(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        """
        图卷积操作，用于计算新的隐藏状态
        :param inputs: 输入数据
        :param adj_mx: 邻接矩阵
        :param state: 隐藏状态
        :param output_size: 输出大小
        :param bias_start: 偏置的初始值
        :return: 计算后的值
        """
        B = inputs.shape[0]  # 获取批量大小
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)  # 计算随机游走矩阵
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)  # 计算转置后的随机游走矩阵

        batch_size = inputs.shape[0]  # 获取批量大小
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))  # 重塑输入数据的形状
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))  # 重塑隐藏状态的形状
        inputs_and_state = torch.cat([inputs, state], dim=2)  # 连接输入数据和隐藏状态
        input_size = inputs_and_state.size(2)  # 获取输入大小

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)  # 在第一个维度上增加一个维度
        x1_0 = torch.unsqueeze(x, 0)  # 在第一个维度上增加一个维度

        if self._max_diffusion_step == 0:  # 如果最大扩散步数为0
            pass  # 不进行扩散
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)  # 计算扩散步数为1的图卷积
            x1_1 = torch.matmul(adj_mx1, x1_0)  # 计算转置后的扩散步数为1的图卷积
            x0_0 = torch.cat([x0_0, x0_1], dim=0)  # 连接扩散步数为0和1的图卷积结果
            x1_0 = torch.cat([x1_0, x1_1], dim=0)  # 连接转置后的扩散步数为0和1的图卷积结果

            for k in range(2, self._max_diffusion_step + 1):  # 计算更大扩散步数的图卷积
                x0_2 = torch.matmul(adj_mx0, x0_1)  # 计算扩散步数为k的图卷积
                x1_2 = torch.matmul(adj_mx1, x1_1)  # 计算转置后的扩散步数为k的图卷积
                x0_0 = torch.cat([x0_0, x0_1], dim=0)  # 连接扩散步数为0到k的图卷积结果
                x1_0 = torch.cat([x1_0, x1_1], dim=0)  # 连接转置后的扩散步数为0到k的图卷积结果
                x0_1 = x0_2  # 更新扩散步数为k的图卷积结果
                x1_1 = x1_2  # 更新转置后的扩散步数为k的图卷积结果

        num_matrices = self._max_diffusion_step + 1  # 确定图卷积矩阵的数量
        x0_0 = x0_0.permute(1, 2, 3, 0)  # 调整图卷积结果的维度
        x1_0 = x1_0.permute(1, 2, 3, 0)  # 调整转置后的图卷积结果的维度
        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])  # 重塑图卷积结果的形状
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])  # 重塑转置后的图卷积结果的形状
        x0_0 = self._gconv_c_0(x0_0)  # 计算图卷积的输出
        x1_0 = self._gconv_c_1(x1_0)  # 计算转置后图卷积的输出

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])  # 返回图卷积的输出


# 定义编码器模型
class EncoderModel(nn.Module):
    def __init__(self, device, n_dim, n_hid, max_diffusion_step, num_nodes, num_rnn_layers, filter_type):
        """
        初始化编码器模型
        :param device: 设备（CPU 或 GPU）
        :param n_dim: 输入维度
        :param n_hid: 隐藏层维度
        :param max_diffusion_step: 最大扩散步数
        :param num_nodes: 节点数量
        :param num_rnn_layers: RNN 层数
        :param filter_type: 滤波器类型
        """
        super(EncoderModel, self).__init__()
        self.device = device  # 设备
        self.input_dim = n_dim  # 输入维度
        self.rnn_units = n_hid  # 隐藏层维度
        self.max_diffusion_step = max_diffusion_step  # 最大扩散步数
        self.num_nodes = num_nodes  # 节点数量
        self.num_rnn_layers = num_rnn_layers  # RNN 层数
        self.filter_type = filter_type  # 滤波器类型
        self.hidden_state_size = self.num_nodes * self.rnn_units  # 隐藏状态大小
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell_(self.device, self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])  # 定义多个 DCGRU 单元

    def forward(self, inputs, adj, hidden_state=None):
        """
        前向传播
        :param inputs: 输入数据
        :param adj: 邻接矩阵
        :param hidden_state: 隐藏状态
        :return: 输出和隐藏状态
        """
        batch_size = inputs.shape[0]  # 获取批量大小
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(self.device)  # 初始化隐藏状态
        hidden_states = []  # 存储隐藏状态
        output = inputs  # 输入数据
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)  # 计算下一个隐藏状态
            hidden_states.append(next_hidden_state)  # 保存隐藏状态
            output = next_hidden_state  # 更新输出数据

        return output, torch.stack(hidden_states)  # 返回输出和隐藏状态

# 定义 GRELEN 模型
class Grelen(nn.Module):
    """
    GRELEN Model.
    """
    def __init__(self, device, T, target_T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head, temperature,
                 hard, GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type, do_prob=0.):
        """
        初始化 GRELEN 模型
        :param device: 设备（CPU 或 GPU）
        :param T: 输入序列长度
        :param target_T: 目标序列长度
        :param Graph_learner_n_hid: 图学习器隐藏层维度
        :param Graph_learner_n_head_dim: 图学习器头维度
        :param Graph_learner_head: 图学习器头数量
        :param temperature: Gumbel-softmax 温度参数
        :param hard: 是否使用硬 Gumbel-softmax
        :param GRU_n_dim: GRU 隐藏层维度
        :param max_diffusion_step: 最大扩散步数
        :param num_nodes: 节点数量
        :param num_rnn_layers: RNN 层数
        :param filter_type: 滤波器类型
        :param do_prob: dropout 概率
        """
        super(Grelen, self).__init__()
        self.device = device  # 设备
        self.len_sequence = T  # 输入序列长度
        self.target_T = target_T  # 目标序列长度
        self.graph_learner = Graph_learner(T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head,
                                           do_prob)  # 图学习器
        self.linear1 = nn.Linear(1, GRU_n_dim)  # 第一个线性层，用于投影
        nn.init.xavier_normal_(self.linear1.weight.data)  # 初始化权重
        self.linear1.bias.data.fill_(0.1)  # 初始化偏置

        self.temperature = temperature  # Gumbel-softmax 温度参数
        self.hard = hard  # 是否使用硬 Gumbel-softmax
        self.GRU_n_dim = GRU_n_dim  # GRU 隐藏层维度
        self.num_nodes = num_nodes  # 节点数量
        self.head = Graph_learner_head  # 图学习器头数量
        self.encoder_model = nn.ModuleList(
            [EncoderModel(self.device, GRU_n_dim, GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type) \
             for _ in range(self.head - 1)])  # 定义多个编码器模型
        self.linear_out = nn.Linear(GRU_n_dim, 1)  # 输出线性层
        nn.init.xavier_normal_(self.linear_out.weight.data)  # 初始化权重
        self.linear_out.bias.data.fill_(0.1)  # 初始化偏置

    def _compute_sampling_threshold(self, batches_seen):
        """
        计算采样阈值
        :param batches_seen: 已见批次
        :return: 采样阈值
        """
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj, head):
        """
        编码器前向传播
        :param inputs: 输入数据
        :param adj: 邻接矩阵
        :param head: 头编号
        :return: 编码后的隐藏状态
        """
        encoder_hidden_state = None
        encoder_hidden_state_tensor = torch.zeros(inputs.shape).to(self.device)
        for t in range(self.len_sequence):
            _, encoder_hidden_state = self.encoder_model[head](inputs[..., t], adj, encoder_hidden_state)
            encoder_hidden_state_tensor[..., t] = encoder_hidden_state[-1, ...].reshape(-1, self.num_nodes,
                                                                                        self.GRU_n_dim)
        return encoder_hidden_state_tensor

    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据
        :return: 概率和输出
        """
        B = inputs.shape[0]  # 获取批量大小
        input_projected = self.linear1(inputs.unsqueeze(-1))  # 通过线性层进行投影
        input_projected = input_projected.permute(0, 1, 3, 2)  # 调整维度顺序
        probs = self.graph_learner(inputs)  # 通过图学习器计算概率
        mask_loc = torch.eye(self.num_nodes, dtype=bool).to(self.device)
        probs_reshaped = probs.masked_select(~mask_loc).view(B, self.head, self.num_nodes * (self.num_nodes - 1)).to(self.device)
        probs_reshaped = probs_reshaped.permute(0, 2, 1)
        prob = F.softmax(probs_reshaped, -1)
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True).to(self.device)

        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes).to(self.device)
        mask = ~torch.eye(self.num_nodes, dtype=bool).unsqueeze(0).unsqueeze(0).to(self.device)
        mask = mask.repeat(self.head, B, 1, 1).to(self.device)
        adj_list[mask] = edges.permute(2, 0, 1).flatten()
        state_for_output = torch.zeros(input_projected.shape).to(self.device)
        state_for_output = (state_for_output.unsqueeze(0)).repeat(self.head - 1, 1, 1, 1, 1)

        for head in range(self.head - 1):
            state_for_output[head, ...] = self.encoder(input_projected, adj_list[head + 1, ...], head)

        state_for_output2 = torch.mean(state_for_output, 0).permute(0, 1, 3, 2)
        output = self.linear_out(state_for_output2).squeeze(-1)[..., -1 - self.target_T:-1]

        return prob, output

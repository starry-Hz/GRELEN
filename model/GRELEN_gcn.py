import torch.nn as nn  # 导入 PyTorch 的神经网络模块,提供了构建神经网络所需的功能
import sys  # 导入系统模块,用于操作系统相关功能
sys.path.append('..')  # 将上级目录添加到系统路径中,以便导入其他模块
from lib.utils import *  # 从 lib.utils 模块中导入所有功能,用于后续代码中的工具函数
# from pygcn.layers import GraphConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

# 引用主程序中的日志记录器
logger = logging.getLogger(__name__)


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
        :param do_prob: dropout 的概率,默认值为 0(即不使用 dropout)
        """
        super(MLP, self).__init__()  # 调用父类（nn.Module）的初始化函数
        self.fc1 = nn.Linear(n_in, n_hid)  # 定义第一个全连接层,输入维度为 n_in,输出维度为 n_hid
        self.fc2 = nn.Linear(n_hid, n_out)  # 定义第二个全连接层,输入维度为 n_hid,输出维度为 n_out
        # 批量归一化层可以对每个小批量的数据进行归一化处理,数据的均值调整为0,方差调整为1
        self.bn = nn.BatchNorm1d(n_out)  # 定义批量归一化层,用于规范化输出层的输出
        self.dropout_prob = do_prob  # 保存 dropout 概率
        # dropout 是一种防止神经网络过拟合的正则化技术。
        # 基本思想：在每次训练过程中,随机丢弃一部分神经元,迫使神经网络不依赖某些特定的节点和路径,增强模型的泛化能力

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
        :param inputs: 输入数据[B,N,Fin]
        :return: 模型的输出
        """
        x = F.elu(self.fc1(inputs))  # 通过第一个全连接层,并使用 ELU 激活函数
        x = F.dropout(x, self.dropout_prob, training=self.training)  # 应用 dropout,随机将部分神经元的输出置0,以防止过拟合
        x = F.elu(self.fc2(x))  # 通过第二个全连接层,并使用 ELU 激活函数
        return self.batch_norm(x)  # 返回批量归一化后的输出 [B,N,Fout]

# 定义一个图学习模型
"""
Figure3中的Relation Inference(关系推断)部分,显示了如何从提取的特征h推断出节点之间的关系𝜃

在输入数据中学习节点之间的关系
1.特征提取:通过MLP模型对输入数据进行特征提取,将其转换为更高维的隐含表示。   代码中的 self.mlp1 部分
2.计算查询和键:通过'wq'和'wk'线性层计算查询和键向量。   代码中的 self.Wq(X) 和 self.Wk(X) 部分。
3.关系推断:通过查询向量和键向量的点积计算注意力权重矩阵。
通过向量之间的点积来推断节点之间的关系。点积结果表示节点之间的相似度和关联程度,即注意力权重。
代码中的 torch.matmul(Xq, Xk.transpose(-1, -2)) 部分,对应图中的relation inference关系推断部分。
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
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)  # 定义一个 MLP 模型,用于处理输入数据
        
        # Wq 和 Wk 是用于计算查询（Query）和键（Key）的线性变换层
        # 输出维度 n_head_dim * head,即多头注意力机制中每个头的维度乘以头的数量
        self.Wq = nn.Linear(n_hid, n_head_dim * head)  # 定义查询权重矩阵,维度为 n_hid 到 n_head_dim * head
        self.Wk = nn.Linear(n_hid, n_head_dim * head)  # 定义键权重矩阵,维度为 n_hid 到 n_head_dim * head

        for m in [self.Wq, self.Wk]:  # 对权重矩阵进行初始化
            if isinstance(m, nn.Linear):  # 如果子模块是全连接层
                nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 正态分布初始化权重
                m.bias.data.fill_(0.1)  # 将偏置初始化为 0.1

    def forward(self, inputs):  # inputs: [B, N, T(features)]
        """
        前向传播
        :param inputs: 输入数据,形状为 [B, N, T]（批量大小,节点数量,特征数量）
        :return: 注意力权重矩阵
        """
        X = self.mlp1(inputs)  # 通过 MLP 模型处理输入数据
        Xq = self.Wq(X)  # 计算查询向量
        Xk = self.Wk(X)  # 计算键向量

        # 获取输入的维度信息
        B, N, n_hid = Xq.shape  # 获取批量大小 B,节点数量 N,隐藏层维度 n_hid

        # 多头注意力机制
        # 调整查询和键向量的形状以适应多头注意力机制
        # 将查询和键矩阵重塑,以便将头的维度分离出来
        Xq = Xq.view(B, N, self.head, self.n_head_dim)  # 重塑查询向量,形状为 [B, N, head, head_dim]
        Xk = Xk.view(B, N, self.head, self.n_head_dim)  # 重塑键向量,形状为 [B, N, head, head_dim]

        # 调整维度顺序,便于后续的矩阵乘法操作
        Xq = Xq.permute(0, 2, 1, 3)  # 调整维度顺序,形状为 [B, head, N, head_dim]
        Xk = Xk.permute(0, 2, 1, 3)  # 调整维度顺序,形状为 [B, head, N, head_dim]

        # 计算注意力权重矩阵,使用矩阵乘法将查询向量和键向量相乘,并对最后两个维度进行转置
        probs = torch.matmul(Xq, Xk.transpose(-1, -2))  # 计算注意力权重矩阵    Relation Inference,Figure3中的关系推断
        return probs # 返回注意力权重矩阵,每个头上,各节点之间的相关性矩阵[B,H,N,N]
        # probs表示节点之间的相关性,可以用于生成邻接矩阵

# # GCN实现1
# class GCNCell(torch.nn.Module):
#     def __init__(self, device, num_units, max_diffusion_step, num_nodes, 
#                 filter_type="laplacian",nonlinearity='tanh'):
#         super().__init__()
#         self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
#         self.device = device
#         self._num_nodes = num_nodes
#         self._num_units = num_units

#         self._gconv = nn.Linear(self._num_units, self._num_units)

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#                 m.bias.data.fill_(0.1)

#     def _calculate_random_walk_matrix(self, adj_mx):
#         adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(self.device)
#         d = torch.sum(adj_mx, 1)
#         d_inv = 1. / d
#         d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)
#         d_mat_inv = torch.diag(d_inv)
#         random_walk_mx = torch.mm(d_mat_inv, adj_mx)
#         return random_walk_mx

#     def forward(self, inputs, hx,adj):
#         logging.info(f"输入inputs形状为{inputs.shape}")  # [128, 51, 64]
#         logging.info(f"邻接矩阵adj形状为{adj.shape}")  # [128, 51, 51]

#         # 输入重塑
#         B = inputs.shape[0]
#         inputs = inputs.reshape(B, self._num_nodes, -1)  # [128, 51, 64]

#         # 计算随机游走矩阵
#         random_walk_mx = self._calculate_random_walk_matrix(adj[0])  # 只使用第一个批次的邻接矩阵
#         random_walk_mx = random_walk_mx.unsqueeze(0).repeat(B, 1, 1)  # [128, 51, 51]

#         # 图卷积计算
#         gconv_output = self._gconv(inputs)  # [128, 51, 64]
#         if self._activation is not None:
#             gconv_output = self._activation(gconv_output)

#         return gconv_output.reshape(B, -1)  # [128, 3264]


class GCNCell(torch.nn.Module):
    def __init__(self, device, num_units, max_diffusion_step, num_nodes, 
                filter_type="laplacian",nonlinearity='tanh'):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self.device = device
        self._num_nodes = num_nodes
        self._num_units = num_units

        self._gconv = nn.Linear(self._num_units, self._num_units)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _calculate_random_walk_matrix(self, adj_mx):
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(self.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hj,adj):
        # logging.info(f"GCNCell输入inputs形状为{inputs.shape}")  # [128, 51, 64]
        # logging.info(f"邻接矩阵adj形状为{adj.shape}")  # [128, 51, 51]

        B = inputs.shape[0]
        inputs = inputs.reshape(B, self._num_nodes, -1)  # [128, 51, 64]

        # 计算随机游走矩阵
        random_walk_mx = self._calculate_random_walk_matrix(adj[0])  # 只使用第一个批次的邻接矩阵
        random_walk_mx = random_walk_mx.unsqueeze(0).repeat(B, 1, 1)  # [128, 51, 51]

        # 使用图卷积进行信息传播
        gconv_output = torch.bmm(random_walk_mx, inputs)  # [128, 51, 64]
        
        # 应用激活函数
        if self._activation is not None:
            gconv_output = self._activation(gconv_output)

        return gconv_output.reshape(B, -1)  # [128, 3264]



# 定义编码器模型
"""
实现了一个基于多层扩散卷积门控循环单元(DCGRU)的编码器模型,主要用于处理时间序列数据和图结构数据。
将输入数据与图的邻接矩阵相结合,学习节点之间的关系,通过循环网络层进行时间序列建模。
通过堆叠多层DCGRUCell_,实现了对图结构时序数据的编码
"""
class EncoderModel(nn.Module):
    def __init__(self, device, n_dim, n_hid, max_diffusion_step, num_nodes, num_rnn_layers, filter_type):
        """
        初始化编码器模型
        :param device: 设备(CPU 或 GPU)
        :param n_dim: 输入维度
        :param n_hid: 隐藏层维度,每个DCGRU的隐藏层单元数量Fhid
        :param max_diffusion_step: 最大扩散步数
        :param num_nodes: 节点数量
        :param num_rnn_layers: RNN 层数
        :param filter_type: 滤波器类型
        """
        super(EncoderModel, self).__init__()
        self.device = device  # 设备
        self.input_dim = n_dim  # 输入维度
        self.rnn_units = n_hid  # 隐藏层维度,即每个DCGRU的隐藏数量单元Fhid
        self.max_diffusion_step = max_diffusion_step  # 最大扩散步数
        self.num_nodes = num_nodes  # 节点数量
        self.num_rnn_layers = num_rnn_layers  # DCGRU层的数量
        self.filter_type = filter_type  # 滤波器类型
        # # 定义每层的隐藏状态大小：节点数 * 隐藏单元数
        self.hidden_state_size = self.num_nodes * self.rnn_units  # 隐藏状态大小
        # 创建了num_rnn_layers个DCGRU单元,每层用于处理输入数据和邻接矩阵,并逐层递归更新隐藏状态
        self.dcgru_layers = nn.ModuleList(
            [GCNCell(self.device, self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])  # 定义多个 DCGRU 单元

    def forward(self, inputs, adj, hidden_state=None):
        """
        前向传播
        :param inputs: 输入数据[B,N*Fin]
        :param adj: 邻接矩阵[B,N,N]
        :param hidden_state: 隐藏状态
        :return: 输出和隐藏状态
        """
        batch_size = inputs.shape[0]  # 获取批量大小
        if hidden_state is None:
            # 如果没有提供隐藏状态,则初始化为全零
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(self.device)  # 初始化隐藏状态
        hidden_states = []  # 用于存储每一层的隐藏状态
        output = inputs  # 输入数据作为初始的输出

        # 逐层处理数据
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            # 调用每一层 DCGRU,计算输出和更新隐藏状态
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)  # 计算下一个隐藏状态
            hidden_states.append(next_hidden_state)  # 将当前层的隐藏状态保存到hidden_staes列表中
            output = next_hidden_state  # 将当前层的输出作为下一层的输入

        # 返回最后一层的输出和所有隐藏状态
        # output[B,N*Fhid] hidden_states [L,B,N*Fhid]
        return output, torch.stack(hidden_states)

# 定义 GRELEN 模型
class Grelen(nn.Module):
    """
    GRELEN Model.
    """
    def __init__(self, device, T, target_T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head, temperature,
                 hard, GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type, do_prob=0.):
        """
        初始化 GRELEN 模型
        :param device: 设备(CPU 或 GPU)
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
        super(Grelen, self).__init__()  # 调用父类的初始化方法
        self.device = device  # 设置模型运行的设备
        self.len_sequence = T  # 设置输入序列长度
        self.target_T = target_T  # 设置预测的目标序列长度

        # 初始化图学习器,负责学习节点之间的关系（Relation Inference 部分）
        # 通过计算每个节点的q和k确定节点之间的关系,从而生成图结构的概率。
        self.graph_learner = Graph_learner(T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head, do_prob)
        
        # 用于输入时序数据投影的线性层,投影维度为 GRU 的隐藏维度
        self.linear1 = nn.Linear(1, GRU_n_dim)
        
        # 初始化线性层的权重,使用 Xavier 标准正态分布
        nn.init.xavier_normal_(self.linear1.weight.data)
        
        # 将偏置初始化为 0.1
        self.linear1.bias.data.fill_(0.1)

        self.temperature = temperature  # Gumbel-softmax 的温度参数,用于控制采样的“随机性”
        self.hard = hard  # 是否使用硬 Gumbel-softmax
        self.GRU_n_dim = GRU_n_dim  # GRU 隐藏层的维度
        self.num_nodes = num_nodes  # 图的节点数量
        self.head = Graph_learner_head  # 图学习器的头数量

        # 定义多个 EncoderModel 模型,用于对输入数据进行编码,每个图学习头都对应一个 EncoderModel
        # 编码器从输入时序数据中提取特征,由多个EncoderModel模块组成,处理多头注意力的图结构。
        self.encoder_model = nn.ModuleList(
            [EncoderModel(self.device, GRU_n_dim, GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type)
             for _ in range(self.head - 1)]
        )

        # 定义输出层,用于将编码结果输出为时间序列预测结果
        self.linear_out = nn.Linear(GRU_n_dim, 1)
        
        # 初始化输出层的权重,使用 Xavier 标准正态分布
        nn.init.xavier_normal_(self.linear_out.weight.data)
        
        # 初始化输出层的偏置为 0.1
        self.linear_out.bias.data.fill_(0.1)

    def _compute_sampling_threshold(self, batches_seen):
        """
        动态计算采样阈值,随着训练进行动态调整采样策略
        :param batches_seen: 已见批次
        :return: 采样阈值
        """
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj, head):
        """
        编码器的前向传播
        通过遍历时间步和邻接矩阵,逐步编码输入的时间序列数据,并返回所有时间步的隐藏状态
        :param inputs: 输入数据（形状为 B x N x T)
        :param adj: 邻接矩阵（表示图结构）
        :param head: 图学习器的头编号
        :return: 编码后的隐藏状态张量
        """
        encoder_hidden_state = None  # 初始化隐藏状态
        encoder_hidden_state_tensor = torch.zeros(inputs.shape).to(self.device)  # 初始化存储隐藏状态的张量

        # 对每一个时间步执行编码操作
        for t in range(self.len_sequence):
            # 调用对应头编号的编码器模型进行编码,更新隐藏状态
            # 隐藏状态h_t由当前输入x_t和上一时间步的隐藏状态h_(t-1)共同决定
            _, encoder_hidden_state = self.encoder_model[head](inputs[..., t], adj, encoder_hidden_state)
            # 将编码后的隐藏状态保存到张量中
            encoder_hidden_state_tensor[..., t] = encoder_hidden_state[-1, ...].reshape(-1, self.num_nodes, self.GRU_n_dim)

        return encoder_hidden_state_tensor  # 返回编码后的隐藏状态张量

    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据 [B,N,T]
        :return: 概率和输出
        """
        B = inputs.shape[0]  # 获取输入批量大小
        input_projected = self.linear1(inputs.unsqueeze(-1))  # 通过线性层对输入进行投影    [B, N, T, GRU_n_dim]
        input_projected = input_projected.permute(0, 1, 3, 2)  # 调整维度顺序以适应模型的输入格式   [B, N, GRU_n_dim, T]

        # 通过图学习器计算节点之间的关系概率（图中的 Relation Inference）
        probs = self.graph_learner(inputs)    # [B, head, N, N]

        # 构建掩码矩阵,用于去除图中节点与自己的连接
        # 生成一个对角线为True,其他部分为false的掩码矩阵mask_loc,用于忽略自己连接(节点与自己的连接)
        mask_loc = torch.eye(self.num_nodes, dtype=bool).to(self.device)
        # 去除对角线的元素,获取节点之间的连接概率
        probs_reshaped = probs.masked_select(~mask_loc).view(B, self.head, self.num_nodes * (self.num_nodes - 1)).to(self.device)
        probs_reshaped = probs_reshaped.permute(0, 2, 1)

        # 对连接概率应用 softmax,将权重归一化为概率分布,确保每个节点的连接概率之和为1
        prob = F.softmax(probs_reshaped, -1)

        # 通过 Gumbel-softmax 进行采样,确定最终的图结构
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True).to(self.device)
        # 计算出的图关系通过Gumbel-softmax进行采样,采样后的结构变为潜在变量z,用以确定节点之间的连接
        # **对应图中的sampling**

        # 初始化邻接矩阵列表,用于存储每个头的邻接矩阵
        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes).to(self.device)
        # 构建掩码,用于忽略对角元素（节点与自身的连接）
        mask = ~torch.eye(self.num_nodes, dtype=bool).unsqueeze(0).unsqueeze(0).to(self.device)
        mask = mask.repeat(self.head, B, 1, 1).to(self.device)

        # 将采样得到的边填充到邻接矩阵中,即学习到的图结构(节点之间的连接)填充到邻接矩阵中
        adj_list[mask] = edges.permute(2, 0, 1).flatten()
        # print(adj_list.shape)
        logging.info(f"GRELEN_gcn邻接矩阵的形状为{adj_list.shape}")

        # 初始化输出状态张量,用于存储编码结果
        state_for_output = torch.zeros(input_projected.shape).to(self.device)
        state_for_output = (state_for_output.unsqueeze(0)).repeat(self.head - 1, 1, 1, 1, 1)

        # 对每个头进行编码,在头部维度上循环,处理不同的图结构
        for head in range(self.head - 1):
            # 调用编码器进行前向传播,从输入数据中提取时序特征,并在每个head生成对应的隐藏状态h,并将编码结果存储到 state_for_output 中
            state_for_output[head, ...] = self.encoder(input_projected, adj_list[head + 1, ...], head)

        # state_for_output2和output对应图中的Decoder和Series Reconstruction部分
        # 将编码后的时序特征通过线性层进行重构,生成预测输出 output。此过程对应图中的解码器和序列重构部分,它负责将编码后的特征（或隐藏状态）还原为预测的时间序列。
        # 对所有头的编码结果取平均值,并调整维度
        state_for_output2 = torch.mean(state_for_output, 0).permute(0, 1, 3, 2)
        # 通过线性层linear_out生成最终的时间序列预测结果
        output = self.linear_out(state_for_output2).squeeze(-1)[..., -1 - self.target_T:-1]
        # 通过对多头编码器的输出取平均,模型可以综合不同头部捕获的特征,提升表示的多样性和完整性,减少单个头部可能的误差,增强模型的鲁棒性

        # prob表示图结构的学习结果(各节点的关系),output是基于图结构进行预测的时间序列
        # prob:节点间的相关性概率矩阵,形状为[B,N(N-1),H];output:模型的预测输出,形状为[B,N,target_T]
        return prob, output  # 返回图结构的概率和预测的时间序列结果

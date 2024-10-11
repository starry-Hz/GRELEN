#-*- coding : utf-8-*-
# coding:unicode_escape
import torch.nn as nn
import sys
sys.path.append('..')
from lib.utils import *
from .encoders2 import SoftPoolingGcnEncoder


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class Graph_learner(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, do_prob=0.):  # n_in = T
        super(Graph_learner, self).__init__()
        self.n_hid = n_hid
        self.head = head
        self.n_in = n_in
        self.n_head_dim = n_head_dim

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.Wq = nn.Linear(n_hid, n_head_dim * head)
        self.Wk = nn.Linear(n_hid, n_head_dim * head)
        for m in [self.Wq, self.Wk]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):  # inputs: [B, N, T(features)]
        X = self.mlp1(inputs)
        Xq = self.Wq(X)  # [B, N, n_hid_subspace]
        Xk = self.Wk(X)
        B, N, n_hid = Xq.shape
        Xq = Xq.view(B, N, self.head, self.n_head_dim)  # [B, N, head, head_dim]
        Xk = Xk.view(B, N, self.head, self.n_head_dim)
        Xq = Xq.permute(0, 2, 1, 3)  # [B, head, N, head_dim]
        Xk = Xk.permute(0, 2, 1, 3)
        probs = torch.matmul(Xq, Xk.transpose(-1, -2))  # [B, head, N, N]

        return probs


class EncoderModel(nn.Module):
    def __init__(self, device, n_dim, n_hid, max_diffusion_step, num_nodes, num_rnn_layers, filter_type):
        """
        初始化编码器模型
        :param device: 设备(CPU �? GPU)
        :param n_dim: 输入维度
        :param n_hid: 隐藏层维�?,每个DCGRU的隐藏层单元数量Fhid
        :param max_diffusion_step: 最大扩散步�?
        :param num_nodes: 节点数量
        :param num_rnn_layers: RNN 层数
        :param filter_type: 滤波器类�?
        """
        super(EncoderModel, self).__init__()
        self.device = device
        self.input_dim = n_dim
        self.hidden_dim = n_hid
        self.rnn_units = n_hid
        self.max_diffusion_step = max_diffusion_step
        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.filter_type = filter_type
        self.hidden_state_size = self.num_nodes * self.rnn_units

        # 使用SoftPoolingGcnEncoder
        self.gcn_layers = nn.ModuleList(
            [SoftPoolingGcnEncoder(self.num_nodes, self.input_dim, self.hidden_dim, self.hidden_dim, 
                                   label_dim=2, num_layers=3, assign_hidden_dim=self.hidden_dim)
             for _ in range(self.num_rnn_layers)])

        # 添加一个线性层将输出调整到合适的形状
        self.projection_layer = nn.Linear(2, self.num_nodes * self.rnn_units)


    def forward(self, inputs, adj, hidden_state=None):
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(self.device)
        hidden_states = []
        output = inputs
        for layer_num, gcn_layer in enumerate(self.gcn_layers):
            output = gcn_layer(output, adj, batch_num_nodes=None)
            # 打印 gcn_layer 的输出形�?
            # print(f"Layer {layer_num} - SoftPoolingGcnEncoder output shape: {output.shape}")
            # 使用线性层将输出转换为期望的形�?
            output = self.projection_layer(output)  # [batch_size, num_nodes * rnn_units]
            output = output.view(batch_size, self.num_nodes, self.rnn_units)  # [batch_size, num_nodes, rnn_units]
            hidden_states.append(output)
        return output, torch.stack(hidden_states)



class Grelen(nn.Module):
    """
    GRELEN Model.
    """

    def __init__(self, device, T, target_T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head, temperature,
                 hard, \
                 GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type, do_prob=0.):  # n_in = T
        super(Grelen, self).__init__()
        self.device = device
        self.len_sequence = T
        self.target_T = target_T
        self.graph_learner = Graph_learner(T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head,
                                           do_prob)
        self.linear1 = nn.Linear(1, GRU_n_dim)  # First layer of projection
        nn.init.xavier_normal_(self.linear1.weight.data)
        self.linear1.bias.data.fill_(0.1)

        self.temperature = temperature
        self.hard = hard
        self.GRU_n_dim = GRU_n_dim
        self.num_nodes = num_nodes
        self.head = Graph_learner_head
        self.encoder_model = nn.ModuleList(
            [EncoderModel(self.device, GRU_n_dim, GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type)
             for _ in range(self.head - 1)])
        self.linear_out = nn.Linear(GRU_n_dim, 1)
        nn.init.xavier_normal_(self.linear_out.weight.data)
        self.linear_out.bias.data.fill_(0.1)

        # 添加 SoftPoolingGcnEncoder 用于产生 self.assign_tensor
        self.soft_pooling_encoder = SoftPoolingGcnEncoder(num_nodes, GRU_n_dim, GRU_n_dim, GRU_n_dim, label_dim=2,
                                                          num_layers=3, assign_hidden_dim=GRU_n_dim, dropout=do_prob)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj, head):
        """
        Encoder forward pass
        """

        encoder_hidden_state = None
        encoder_hidden_state_tensor = torch.zeros(inputs.shape).to(self.device)
        for t in range(self.len_sequence):
            _, encoder_hidden_state = self.encoder_model[head](inputs[..., t], adj, encoder_hidden_state)
            # 打印 encoder_hidden_state 的形�?
            # print(f"Step {t} - encoder_hidden_state shape: {encoder_hidden_state.shape}")
            # print(f"inputs.size(0): {inputs.size(0)}, self.num_nodes: {self.num_nodes}, self.GRU_n_dim: {self.GRU_n_dim}")
            # 尝试 reshape 之前确保形状是匹配的
            encoder_hidden_state_tensor[..., t] = encoder_hidden_state[-1, ...].view(inputs.size(0), self.num_nodes, self.GRU_n_dim)
        return encoder_hidden_state_tensor

    def forward(self, inputs):

        B = inputs.shape[0]
        input_projected = self.linear1(inputs.unsqueeze(-1))  # [B, N, T, GRU_n_dim]
        input_projected = input_projected.permute(0, 1, 3, 2)  # [B, N, GRU_n_dim, T]
        probs = self.graph_learner(inputs)  # [B, head, N, N]
        mask_loc = torch.eye(self.num_nodes, dtype=bool).to(self.device)
        probs_reshaped = probs.masked_select(~mask_loc).view(B, self.head, self.num_nodes * (self.num_nodes - 1)).to(self.device)
        probs_reshaped = probs_reshaped.permute(0, 2, 1)
        prob = F.softmax(probs_reshaped, -1)
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True).to(self.device)

        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes).to(self.device)
        mask = ~torch.eye(self.num_nodes, dtype=bool).unsqueeze(0).unsqueeze(0).to(self.device)
        mask = mask.repeat(self.head, B, 1, 1).to(self.device)
        adj_list[mask] = edges.permute(2, 0, 1).flatten()


        adj_mean = torch.mean(adj_list, dim=0)

        # print(f"adj_list:{adj_list.shape}")
        # print(f"edges.permute(2, 0, 1):{edges.permute(2, 0, 1).shape}")
        # print(f"adj_out:{adj_out.shape}")
        state_for_output = torch.zeros(input_projected.shape).to(self.device)
        state_for_output = (state_for_output.unsqueeze(0)).repeat(self.head - 1, 1, 1, 1, 1)

        for head in range(self.head - 1):
            state_for_output[head, ...] = self.encoder(input_projected, adj_list[head + 1, ...], head)

        state_for_output2 = torch.mean(state_for_output, 0).permute(0, 1, 3, 2)
        output = self.linear_out(state_for_output2).squeeze(-1)[..., -1 - self.target_T:-1]

        # assign_tensor = SoftPoolingGcnEncoder.assign_tensor_(input_projected,)

        return prob, output, adj_mean

    # def loss(self, pred, target, assign_matrices):
    #     """
    #     Loss function derived from SoftPoolingGcnEncoder.
    #     :param pred: Predicted output
    #     :param target: Ground truth labels
    #     :param assign_matrices: Assignment matrices from the pooling layer
    #     :return: Computed loss value
    #     """
    #     mse_loss = F.mse_loss(pred, target)
    #     # Regularization term for assignment matrices to encourage sparsity
    #     reg_loss = 0
    #     for assign_matrix in assign_matrices:
    #         reg_loss += torch.mean(torch.sum(assign_matrix ** 2, dim=1))
    #     total_loss = mse_loss + 1e-4 * reg_loss
    #     return total_loss

    # # 损失函数
    # def loss(self,pred,label,type="softmax",adj=None, batch_num_nodes=None, adj_hop=1,linkpred=True):
    #     # SoftPoolingGcnEncoder 中产�? self.assign_tensor 的过�?
    #     pred = pred.view(pred.size(0), -1, pred.size(-1))  # 调整 pred 的形状为 [B, N, F]，以匹配邻接矩阵

    #     self.assign_tensor = self.soft_pooling_encoder.gcn_forward(pred, adj, self.soft_pooling_encoder.conv_first,
    #                                                                self.soft_pooling_encoder.conv_block,
    #                                                                self.soft_pooling_encoder.conv_last)
        
    #     eps = 1e-7
    #     label = label.view(-1)  # 展平标签以匹配交叉熵损失的输入要求，但保持批次大小不�?
    #     label = label.squeeze(-1)  # 移除多余的维度，调整�? [B]

    #     # softmax + CE
    #     if type == 'softmax':
    #         # 交叉熵损失函�? (F.cross_entropy) 计算损失
    #         loss1 =  F.cross_entropy(pred, label, reduction='mean')
    #         # loss1 = F.cross_entropy(pred, label.argmax(dim=-1), reduction='mean')  # 使用交叉熵损失，标签取最大值索�?
    #     elif type == 'margin':
    #         # 多标签边缘损�? (margin)：用于多标签分类任务,鼓励正确类别的预测分数比其他类别高�?
    #         batch_size = pred.size()[0]
    #         label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()    # [batch_size, label_dim]
    #         # long() 函数将张量元素转换为 torch.int64 类型
    #         # 使用 scatter_ 方法�? label 中的类别信息转换为独热编�? (one-hot encoding)
    #         label_onehot.scatter_(1, label.view(-1,1), 1)
    #         loss2 =  torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

    #     eps = 1e-7  # 设置小的数值eps,用于计算链路预测损失�?,防止log函数中出现零值导致数值问�?
    #     if linkpred:
    #         print(adj.shape)
    #         max_num_nodes = adj.size()[1]
    #         # pred_adj0 : [batch_size, num_nodes, num_nodes],表示池化后的节点分配回原始节点时的关系矩�?,@表示矩阵乘法
    #         pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
    #         tmp = pred_adj0
    #         pred_adj = pred_adj0
    #         for adj_pow in range(adj_hop-1):
    #             tmp = tmp @ pred_adj0   # 表示多跳邻接关系的近�?,每次相乘表示多跳邻接关系的连接权�?
    #             pred_adj = pred_adj + tmp   # 累加每次计算的多跳邻接矩�?,得到整体的预测邻接矩�?
    #         # torch.min() 是逐元素比较函�?,它会�? pred_adj �? torch.ones(1) 的对应位置上取最小�?
    #         pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
    #         #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
    #         #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
    #         #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)

    #         # 计算链路预测损失 self.link_loss
    #         # 真实连接：对于连接存在的边（adj = 1�?,损失�? -adj * torch.log(pred_adj + eps)�?
    #         # 非连接：对于不存在的边（adj = 0�?,损失�? -(1 - adj) * torch.log(1 - pred_adj + eps)
    #         self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)

    #         # 表示所有图的节点数量相�?,直接使用最大节点数计算邻接矩阵的总元素数�? num_entries
    #         if batch_num_nodes is None:
    #             num_entries = max_num_nodes * max_num_nodes * adj.size()[0] # 总的邻接关系数量
    #             print('Warning: calculating link pred loss without masking')
    #             logging.info('Warning: calculating link pred loss without masking')
    #         else:
    #             # 如果提供了每个图的节点数量batch_num_nodes,生成掩码,屏蔽掉多余的计算,减少无效节点对链路预测损失的影响
    #             num_entries = np.sum(batch_num_nodes * batch_num_nodes)
    #             embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
    #             adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
    #             self.link_loss[(1-adj_mask).bool()] = 0.0

    #         self.link_loss = torch.sum(self.link_loss) / float(num_entries) # 对链路预测损失进行平�?,得到归一化的链路预测损失�?
    #         #print('linkloss: ', self.link_loss)
    #         # logging.info
    #         return loss1 + self.link_loss    # 返回总损失值（预测损失 + 链路预测损失�?
    #     return loss1 # 如果没有启用链路预测损失,则仅返回预测损失

    # def loss(self, pred, label, type="softmax", adj=None, batch_num_nodes=None, adj_hop=1, linkpred=True):
    #     # print(f"pred shape :{pred.shape},label.shape{label.shape}")   # pred shape :torch.Size([128, 51, 29]),label.shapetorch.Size([128, 51, 29])
    #     pred = pred.view(pred.size(0), -1, pred.size(-1))  # 调整 pred 的形状为 [B, N, F]，以匹配邻接矩阵
    #     # print("##############################################")
    #     # print(f"pred shape :{pred.shape}")  # pred shape :torch.Size([128, 51, 29])
    #     if pred.dim() == 4:
    #         print(f"Skipping soft_pooling_encoder as pred is 4D: pred.shape={pred.shape}")
    #     else:
    #         self.assign_tensor = self.soft_pooling_encoder.gcn_forward(pred, adj, self.soft_pooling_encoder.conv_first,
    #                                                                    self.soft_pooling_encoder.conv_block,
    #                                                                    self.soft_pooling_encoder.conv_last)
        
    #     eps = 1e-7
    #     pred = pred.reshape(pred.size(0), -1) 
    #     label = label.reshape(-1)
        
    #     # softmax + CE
    #     if type == 'softmax':
    #         loss1 = F.cross_entropy(pred.view(-1, pred.size(-1)), label, reduction='mean')  # 计算交叉熵损�?
    #     elif type == 'margin':
    #         batch_size = pred.size()[0]
    #         label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
    #         label_onehot.scatter_(1, label.view(-1, 1), 1)
    #         loss1 = torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

    #     if linkpred:
    #         max_num_nodes = adj.size()[1]
    #         pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
    #         tmp = pred_adj0
    #         pred_adj = pred_adj0
    #         for adj_pow in range(adj_hop - 1):
    #             tmp = tmp @ pred_adj0
    #             pred_adj = pred_adj + tmp
    #         pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
    #         self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)

    #         if batch_num_nodes is None:
    #             num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
    #             print('Warning: calculating link pred loss without masking')
    #             logging.info('Warning: calculating link pred loss without masking')
    #         else:
    #             num_entries = np.sum(batch_num_nodes * batch_num_nodes)
    #             embedding_mask = self.soft_pooling_encoder.construct_mask(max_num_nodes, batch_num_nodes)
    #             adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
    #             self.link_loss[(1 - adj_mask).bool()] = 0.0

    #         self.link_loss = torch.sum(self.link_loss) / float(num_entries)
    #         return loss1 + self.link_loss

    #     return loss1
    def loss(self, pred, label, type="softmax", adj=None, batch_num_nodes=None, adj_hop=1, linkpred=True):
        print(f"pred shape:{pred.shape},label shape:{label.shape}")   # pred shape:torch.Size([128, 51, 29]),label shape:torch.Size([128, 51, 29])
        # print(f"adj shape:{adj.shape}") # adj shape:torch.Size([128, 51, 51])
        
        self.assign_tensor = self.soft_pooling_encoder.gcn_forward(
            pred, adj, 
            self.soft_pooling_encoder.conv_first,
            self.soft_pooling_encoder.conv_block,
            self.soft_pooling_encoder.conv_last
        )
        print("################################################################")
        print(f"Adjusted pred shape: {pred.shape}, label shape: {label.shape}")

        if type == 'softmax':
            try:
                loss1 = F.cross_entropy(pred, label, reduction='mean')
            except ValueError as e:
                print(f"Error in cross_entropy: {e}")
                return None
        elif type == 'margin':
            label_onehot = torch.zeros(batch_size * num_nodes, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            loss1 = torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        if torch.isnan(loss1):
            print("Warning: NaN detected in primary loss computation!")
            return None

        eps = 1e-7
        total_loss = loss1


        if linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            batch_num_nodes = adj.size()[-1]
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
                logging.info('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.soft_pooling_encoder.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            total_loss += self.link_loss

        if torch.isnan(total_loss):
            print("Warning: NaN detected in total loss computation!")
            return None

        print(total_loss)
        return total_loss
    

        


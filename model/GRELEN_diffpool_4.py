#-*- coding : utf-8-*-
# coding:unicode_escape
import torch.nn as nn
import sys
sys.path.append('..')
from lib.utils import *
# from .encoders3 import SoftPoolingGcnEncoder

import torch
from torch.nn import init
import torch.nn.functional as F

import numpy as np


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


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True, device='cuda:0'):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.device = device
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(self.device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(self.device))
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None, device='cuda:0'):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.device = device

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, device=self.device).to(self.device)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias, device=self.device).to(self.device) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, device=self.device).to(self.device)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim).to(self.device)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim).to(self.device))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim).to(self.device))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes): 
        packed_masks = [torch.ones(int(num), device=self.device) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes, device=self.device)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2)

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, type='softmax'):
        if type == 'softmax':
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim, device=self.device).long()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, device, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None):

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args, device=device)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        self.device = device

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        x_a = kwargs['assign_x'] if 'assign_x' in kwargs else x

        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask)
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x
        
            embedding_tensor = self.gcn_forward(x, adj, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        output = torch.cat(out_all, dim=1) if self.concat else out
        # print(f"SoftPoolingGcnEncoder output shape:{output.shape}")   # output shape:torch.Size([1280, 384])
        ypred = self.pred_model(output)
        # print(f"ypred:{ypred.shape}")   # ypred:torch.Size([1280, 2])
        # print(ypred)

        return ypred
        # return output

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype, device=self.device))
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
                logging.info('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            return loss + self.link_loss
        return loss


class EncoderModel(nn.Module):
    def __init__(self, device, n_dim, n_hid, max_diffusion_step, num_nodes, num_rnn_layers, filter_type):
        """
        初始化编码器模型
        :param device: 设备(CPU GPU)
        :param n_dim: 输入维度
        :param n_hid: 隐藏层维数,每个DCGRU的隐藏层单元数量Fhid
        :param max_diffusion_step: 最大扩散步数
        :param num_nodes: 节点数量
        :param num_rnn_layers: RNN 层数
        :param filter_type: 滤波器类型
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
            [SoftPoolingGcnEncoder(self.device,self.num_nodes, self.input_dim, self.hidden_dim, self.hidden_dim, 
                                   label_dim=2, num_layers=3, assign_hidden_dim=self.hidden_dim)
             for _ in range(self.num_rnn_layers)])

        # 添加一个线性层将输出调整到合适的形状
        # self.projection_layer = nn.Linear(2, self.num_nodes * self.rnn_units)


    def forward(self, inputs, adj, hidden_state=None):
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(self.device)
        hidden_states = []
        output = inputs
        # print(f"EncoderModel output shape : {output.shape}") # output shape : torch.Size([1280, 51, 64])
        for layer_num, gcn_layer in enumerate(self.gcn_layers):
            output = gcn_layer(output, adj, batch_num_nodes=None)
            # print(f"EncoderModel Layer {layer_num} - SoftPoolingGcnEncoder output shape: {output.shape}")
            # output = self.projection_layer(output)  # [batch_size, num_nodes * rnn_units]
            # output = output.view(batch_size, self.num_nodes, self.rnn_units)  # [batch_size, num_nodes, rnn_units]
            hidden_states.append(output)
        # EncoderModel output shape : torch.Size([128, 2]),torch.stack(hidden_states):torch.Size([1, 128, 2])
        # print(f"EncoderModel output shape : {output.shape},torch.stack(hidden_states):{torch.stack(hidden_states).shape}")
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
        # self.linear_out = nn.Linear(GRU_n_dim, 1)
        self.linear_out = nn.Linear(2, 2)
        nn.init.xavier_normal_(self.linear_out.weight.data)
        self.linear_out.bias.data.fill_(0.1)

        # Attention weights for combining multiple heads
        self.attention_weights = nn.Parameter(torch.ones(self.head - 1), requires_grad=True)

        # 添加 SoftPoolingGcnEncoder 用于产生 self.assign_tensor
        self.soft_pooling_encoder = SoftPoolingGcnEncoder(self.device,num_nodes, GRU_n_dim, GRU_n_dim, GRU_n_dim, label_dim=2,
                                                          num_layers=3, assign_hidden_dim=GRU_n_dim, dropout=do_prob)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj, head):
        """
        Encoder forward pass with aggregation
        """
        encoder_hidden_state = None
        output = inputs

        # 存储每个时间步的隐藏状态
        hidden_states = []

        for t in range(self.len_sequence):
            ypred, encoder_hidden_state = self.encoder_model[head](output[..., t], adj, encoder_hidden_state)
            hidden_states.append(ypred)

        # 将所有时间步的隐藏状态拼接成 tensor
        hidden_states = torch.stack(hidden_states, dim=0)

        # 对时间维度进行聚合（这里用平均池化）
        aggregated_state = torch.mean(hidden_states, dim=0)

        return aggregated_state

    def forward(self, inputs):
        B = inputs.shape[0]
        input_projected = self.linear1(inputs.unsqueeze(-1))  # [B, N, T, GRU_n_dim]
        input_projected = input_projected.permute(0, 1, 3, 2)  # [B, N, GRU_n_dim, T]
        probs = self.graph_learner(inputs)  # [B, head, N, N]

        # 获得每个头的邻接矩阵
        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes).to(self.device)

        # 保存所有头的输出
        state_for_output = []

        for head in range(self.head - 1):
            encoder_output = self.encoder(input_projected, adj_list[head + 1, ...], head)
            state_for_output.append(encoder_output)

        # 将多个头的输出堆叠并加权求和
        state_for_output = torch.stack(state_for_output, dim=0)  # [head-1, B, 2]
        attention_weights = F.softmax(self.attention_weights, dim=0)  # 对注意力权重进行 softmax
        state_for_output = torch.sum(attention_weights.view(-1, 1, 1) * state_for_output, dim=0)  # 加权求和

        # 预测输出
        output = self.linear_out(state_for_output)
        print(f"GRELEN output shape : {output.shape}")  # GRELEN output shape : torch.Size([128, 2])
        # print(f"GRELEN output {output}")

        return probs, output, torch.mean(adj_list, dim=0)

    # # 计算预测值与真实标签之间的损失,用于模型的训练和优化
    # def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
    #     eps = 1e-7
    #     loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
    #     if self.linkpred:
    #         max_num_nodes = adj.size()[1]
    #         pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
    #         tmp = pred_adj0
    #         pred_adj = pred_adj0
    #         for adj_pow in range(adj_hop - 1):
    #             tmp = tmp @ pred_adj0
    #             pred_adj = pred_adj + tmp
    #         pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype, device=self.device))
    #         self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
    #         if batch_num_nodes is None:
    #             num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
    #             print('Warning: calculating link pred loss without masking')
    #             logging.info('Warning: calculating link pred loss without masking')
    #         else:
    #             num_entries = np.sum(batch_num_nodes * batch_num_nodes)
    #             embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
    #             adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
    #             self.link_loss[(1 - adj_mask).bool()] = 0.0

    #         self.link_loss = torch.sum(self.link_loss) / float(num_entries)
    #         return loss + self.link_loss
    #     return loss
    # 计算预测值与真实标签之间的损失,用于模型的训练和优化
    def loss(self, pred, label,adj, type='softmax'):
        '''
        pred : 预测值,形状为 [batch_size, label_dim],表示每个样本的预测输出
        label : 真实标签,形状为 [batch_size],表示每个样本的真实类别
        type : 损失类型,默认为 'softmax',可选为 'softmax' 或 'margin'
        '''
        # softmax + CE
        if type == 'softmax':
            # 交叉熵损失函数 (F.cross_entropy) 计算损失
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            # 多标签边缘损失 (margin)：用于多标签分类任务,鼓励正确类别的预测分数比其他类别高。
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()    # [batch_size, label_dim]
            # long() 函数将张量元素转换为 torch.int64 类型
            # 使用 scatter_ 方法将 label 中的类别信息转换为独热编码 (one-hot encoding)
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)
            
        #return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())
    

        


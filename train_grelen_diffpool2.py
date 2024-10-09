#-*- coding : utf-8-*-
# coding:unicode_escape

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import logging
from time import time
# import time
# 从tensorboardX库中导入SummaryWriter模块
from tensorboardX import SummaryWriter


from lib.utils import *
# from model.encoders import SoftPoolingGcnEncoder

# 从model.GRELEN库中导入所有模块
#####################################################################修改模型
from model.GRELEN_diffpool_2 import *

# import logging
# # 日志配置
# logging.basicConfig(filename='train_grelen.log', level=logging.INFO,
#                     format='%(asctime)s:%(levelname)s:%(message)s')
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='./log/train_grelen_diffpool_2.log',
                    filemode='a')

# 训练一个 epoch 的函数
def train_epoch(net, train_loader, optimizer, sw, epoch, config):
    B, N, T, target_T = config.B, config.N, config.T, config.target_T

    net.train()  # 设置为训练模式
    total_loss_train = []

    for batch_index, batch_data in enumerate(train_loader):
        encoder_inputs, labels, true_adj = batch_data  # 获取输入数据、标签和邻接矩阵标签
        encoder_inputs = encoder_inputs[:, :, 0, :]
        labels = labels[:, :, 0, T - target_T:]

        optimizer.zero_grad()
        prob, output = net(encoder_inputs)

        # 计算总损失，使用 SoftPoolingGcnEncoder 的 loss 函数
        total_loss = net.loss(prob, labels, true_adj)
        total_loss_train.append(total_loss.item())

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        # 记录到 TensorBoard
        global_step = epoch * len(train_loader) + batch_index
        sw.add_scalar('training_loss', total_loss.item(), global_step)

    # 打印并记录平均损失
    print(f'Epoch {epoch}, Total Loss: {np.mean(total_loss_train)}')
    logging.info(f'Epoch {epoch}, Total Loss: {np.mean(total_loss_train)}')


# 主程序入口
if __name__ == '__main__':
    from config_files.SWAT_config_gcn import Config
    config = Config()

    device = config.device
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    train_filename = config.train_filename

    train_loader, train_target_tensor, val_loader, val_target_tensor, _mean, _std = load_data_train(train_filename, device, batch_size)
    logging.info(f"Training file: {train_filename}")

    B, N, T, target_T = config.B, config.N, config.T, config.target_T
    n_in = config.n_in
    n_hid = config.n_hid
    do_prob = config.do_prob
    Graph_learner_n_hid = config.Graph_learner_n_hid
    Graph_learner_n_head_dim = config.Graph_learner_n_head_dim
    Graph_learner_head = config.Graph_learner_head
    prior = config.prior
    temperature = config.temperature
    GRU_n_dim = config.GRU_n_dim
    max_diffusion_step = config.max_diffusion_step
    num_rnn_layers = config.num_rnn_layers

    num_nodes = N
    hard = 'True'
    filter_type = 'random'

    net = Grelen(
        config.device, T, target_T, Graph_learner_n_hid,
        Graph_learner_n_head_dim, Graph_learner_head,
        temperature, hard, GRU_n_dim,
        max_diffusion_step, num_nodes,
        num_rnn_layers, filter_type, do_prob=0.
    ).to(config.device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=config.save_dir, flush_secs=5)

    best_val_loss = np.inf
    for epoch in range(1):
        train_epoch(net, train_loader, optimizer, sw, epoch, config)
        val_loss = val_epoch(net, val_loader, sw, epoch, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(config.save_dir, f'best_epoch_{epoch}.params'))
            logging.info(f'Saved best model at epoch {epoch} with validation loss {val_loss:.4f}')

    sw.close()
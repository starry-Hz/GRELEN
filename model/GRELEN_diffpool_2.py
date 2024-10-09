import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import logging
from tensorboardX import SummaryWriter
from lib.utils import *
from model.GRELEN_diffpool_2 import *
from model.encoders import SoftPoolingGcnEncoder

# 配置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='./log/train_grelen_diffpool_2.log',
                    filemode='a')

# 计算链路预测损失的函数
def link_prediction_loss(pred_adj, true_adj):
    """
    计算链路预测损失
    :param pred_adj: 预测的邻接矩阵
    :param true_adj: 实际的邻接矩阵
    :return: 损失值
    """
    loss_fn = nn.BCELoss()  # 使用二分类交叉熵损失
    return loss_fn(pred_adj, true_adj)

# 训练一个 epoch 的函数
def train_epoch(net, train_loader, optimizer, sw, epoch, config):
    B, N, T, target_T = config.B, config.N, config.T, config.target_T
    prior = config.prior
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior).to(config.device)
    variation = 1

    net.train()  # 设置为训练模式
    kl_train = []
    nll_train = []
    link_loss_train = []
    softpool_loss_train = []

    for batch_index, batch_data in enumerate(train_loader):
        encoder_inputs, labels, true_adj = batch_data  # 获取输入数据、标签和邻接矩阵标签
        encoder_inputs = encoder_inputs[:, :, 0, :]
        labels = labels[:, :, 0, T - target_T:]

        optimizer.zero_grad()
        prob, output = net(encoder_inputs)

        # 计算 KL 散度损失和负对数似然损失
        loss_kl = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(config.device)
        loss_nll = nll_gaussian(output, labels, variation).to(config.device)
        total_loss = loss_kl + loss_nll

        # 计算链路预测损失
        pred_adj = torch.mean(prob, dim=1)  # 使用预测的概率矩阵
        loss_link = link_prediction_loss(pred_adj, true_adj)
        total_loss += loss_link

        # 计算 SoftPoolingGcnEncoder 的损失
        if hasattr(net, 'gcn_layers') and isinstance(net.gcn_layers[0], SoftPoolingGcnEncoder):
            softpool_loss = sum(layer.loss(prob, labels, true_adj) for layer in net.gcn_layers if isinstance(layer, SoftPoolingGcnEncoder))
            total_loss += softpool_loss
            softpool_loss_train.append(softpool_loss.item())

        # 记录损失
        kl_train.append(loss_kl.item())
        nll_train.append(loss_nll.item())
        link_loss_train.append(loss_link.item())

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        # 记录到 TensorBoard
        global_step = epoch * len(train_loader) + batch_index
        sw.add_scalar('training_loss', total_loss.item(), global_step)
        sw.add_scalar('kl_loss', loss_kl.item(), global_step)
        sw.add_scalar('nll_loss', loss_nll.item(), global_step)
        sw.add_scalar('link_loss', loss_link.item(), global_step)
        if softpool_loss_train:
            sw.add_scalar('softpool_loss', softpool_loss.item(), global_step)

    # 打印并记录平均损失
    print(f'Epoch {epoch}, KL Loss: {np.mean(kl_train)}, NLL Loss: {np.mean(nll_train)}, Link Loss: {np.mean(link_loss_train)}, SoftPool Loss: {np.mean(softpool_loss_train) if softpool_loss_train else 0}')
    logging.info(f'Epoch {epoch}, KL Loss: {np.mean(kl_train)}, NLL Loss: {np.mean(nll_train)}, Link Loss: {np.mean(link_loss_train)}, SoftPool Loss: {np.mean(softpool_loss_train) if softpool_loss_train else 0}')


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
    variation = 1

    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior).to(device)

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

        # 检测并保存最佳模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(config.save_dir, f'best_epoch_{epoch}.params'))
            logging.info(f'Saved best model at epoch {epoch} with validation loss {val_loss:.4f}')

    sw.close()
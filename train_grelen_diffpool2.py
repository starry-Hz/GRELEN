# 导入utils库中的所有模块
from lib.utils import *

from time import time
# import time
# 从tensorboardX库中导入SummaryWriter模块
from tensorboardX import SummaryWriter

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
                    filename='./log/train_grelen_diffpool2.log',
                    filemode='a')

# 验证一个epoch的函数
# 通过验证集计算模型的损失,并记录每个epoch的验证损失
def val_epoch(net, val_loader, sw, epoch, config):
    # 从配置中获取参数
    B, N, T, target_T = config.B, config.N, config.T, config.target_T     # B:批次大小, N:节点数, T:输入序列长度, target_T:目标序列长度
    prior = config.prior
    log_prior = torch.FloatTensor(np.log(prior))  # 将prior转换为log_prior
    log_prior = torch.unsqueeze(log_prior, 0)  # 添加一个维度
    log_prior = torch.unsqueeze(log_prior, 0)  # 再添加一个维度
    log_prior = Variable(log_prior)  # 转换为变量
    log_prior = log_prior.cuda()  # 将变量转移到GPU

    # 设置为评估模式
    # 禁用 dropout 和 batch normalization 中的随机行为，确保模型在验证集上进行评估时具有稳定的表现。
    net.train(False)

    # 不计算梯度
    with torch.no_grad():   # 验证过程中不需要反向传播
        tmp = []  # 临时存储每个batch的损失
        # 遍历验证数据加载器
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data  # 获取输入数据和标签
            encoder_inputs = encoder_inputs[:, :, 0, :]  # 处理输入数据
            labels = labels[:, :, 0, T-target_T:]  # 处理标签数据

            # 前向传播,获取预测输出
            prob, output,adj_out = net(encoder_inputs)

            # # 计算KL散度损失和负对数似然损失
            loss_kl = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(device)  # 计算KL散度
            loss_nll = nll_gaussian(output, labels, variation).to(device)  # 计算负对数似然损失
            loss = loss_kl + loss_nll  # 总损失
            # tmp.append(loss.item())  # 保存损失
            print(f"output:{output.shape}") # [128, 51, 29]
            print(f"labels:{labels.shape}") # [128, 51, 29]

            output_reshaped = output.view(-1, output.shape[-1])  # [128 * 51, 29]

            # label形状有问题 想办法变为[6528],搞清楚label的形状问题###################################
            labels_reshaped = labels.view(-1, labels.shape[-1])  # [128 * 51, 29]
            labels_reshaped = labels_reshaped.reshape(-1)
            print(f"output_reshaped:{output_reshaped.shape},labels_reshaped[0] shape :{labels_reshaped.shape},labels_reshaped[0] shape:{labels_reshaped.shape}")

            loss_pool = net.loss(output_reshaped,labels_reshaped,adj=adj_out)
            tmp.append(loss_pool.item())  # 保存损失

        # # 计算平均验证损失
        validation_loss = sum(tmp) / len(tmp)

        # 打印并记录验证损失
        print('epoch: %s, loss_pool: %.2f' % (epoch, validation_loss))
        logging.info('epoch: %s, loss_pool: %.2f' % (epoch, validation_loss))

        # 将验证损失添加到SummaryWriter中
        sw.add_scalar('loss_pool', validation_loss, epoch)

    # 返回验证损失
    return loss_pool

# 主程序入口
if __name__ == '__main__':
    # 导入配置
    from config_files.SWAT_config_gcn import Config
    config = Config()

    # 从配置中获取参数
    device = config.device  # 设备
    batch_size = config.batch_size  # 批处理大小
    learning_rate = config.learning_rate  # 学习率
    epochs = config.epochs  # 训练周期数
    train_filename = config.train_filename  # 训练数据文件名

    # 加载训练和验证数据
    train_loader, train_target_tensor, val_loader, val_target_tensor, _mean, _std = load_data_train(train_filename, device, batch_size)
    logging.info(f"训练集文件{train_filename}")

    # 从配置中获取参数
    B, N, T, target_T = config.B, config.N, config.T, config.target_T   # B:批次大小, N:节点数, T:输入序列长度, target_T:目标序列长度

    # 从配置中获取模型参数
    n_in = config.n_in  # 输入特征数
    n_hid = config.n_hid  # 隐藏层维度
    do_prob = config.do_prob # dropout概率,随机将一部分神经元暂时“关闭”,防止过拟合

    Graph_learner_n_hid = config.Graph_learner_n_hid  # 图学习器隐藏层维度
    Graph_learner_n_head_dim = config.Graph_learner_n_head_dim  # 图学习器头部维度
    Graph_learner_head = config.Graph_learner_head  # 图学习器头数
    prior = config.prior  # 先验概率
    temperature = config.temperature  # 温度参数
    GRU_n_dim = config.GRU_n_dim  # GRU维度
    max_diffusion_step = config.max_diffusion_step  # 最大扩散步数
    num_rnn_layers = config.num_rnn_layers  # RNN层数

    # 从配置中获取训练参数
    start_epoch = config.start_epoch  # 起始训练周期
    params_path = config.save_dir  # 保存目录

    # 设置训练参数
    num_nodes = N
    hard = 'True'
    filter_type = 'random'
    variation = 1

    # 将prior转换为log_prior，并将其转移到GPU
    log_prior = torch.FloatTensor(np.log(prior))  # 将prior转换为log_prior
    log_prior = torch.unsqueeze(log_prior, 0)  # 添加一个维度,形状从(4)变为(1,4)
    log_prior = torch.unsqueeze(log_prior, 0)  # 再添加一个维度,形状从(1,4)变为(1,1,4)
    log_prior = Variable(log_prior)  # 转换为变量
    log_prior = log_prior.cuda()  # 将变量转移到GPU

    # 初始化Grelen模型并将其转移到GPU
    net = Grelen(
        config.device, T, target_T, Graph_learner_n_hid,
        Graph_learner_n_head_dim, Graph_learner_head,
        temperature, hard, GRU_n_dim,
        max_diffusion_step, num_nodes,
        num_rnn_layers, filter_type, do_prob=0.
    ).to(config.device)

    # 初始化优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 初始化SummaryWriter
    # 创建文件,并保存数据到文件,利用这种方法可以在训练循环过程中,将数据写入文件中,并不会延缓训练的速度
    sw = SummaryWriter(logdir=params_path, flush_secs=5) # flush_secs=5 表示每隔5秒将日志缓冲区的数据刷新到磁盘

    # 打印并记录模型信息
    ##########################################################################################################
    print(net)
    logging.info(net)
    ##########################################################################################################

    # 初始化训练过程中的变量
    best_epoch = 0
    global_step = 0
    best_val_loss = np.inf  # 初始化为正无穷
    start_time = time()  # 记录开始时间

    # 如果start_epoch大于0,则加载之前的模型参数。实现断点续训功能,在之前的训练中保存了模型,可以从指定的epoch加载保存的参数
    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch)
        logging.info('start epoch:', start_epoch)
        print('load weight from: ', params_filename)
        logging.info('load weight from: ', params_filename)

    total_time = 0
    # 开始训练循环
    for epoch in range(start_epoch, 100):
        
        begin_time = time()
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        # 验证一个epoch并获取验证损失
        val_loss = val_epoch(net, val_loader, sw, epoch, config)

        # 如果验证损失是最好的，则保存模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)
            logging.info('save parameters to file: %s' % params_filename)

        # 设置为训练模式,模型会进行反向传播和参数更新
        # dropout:随机丢弃神经元,batchnorm:使用当前batch的统计信息,并更新内部的均值和方差
        net.train()

        # kl_train = []
        # nll_train = []
        loss_pool_train = []
        # 遍历训练数据加载器
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data  # 获取输入数据和标签
            encoder_inputs = encoder_inputs[:, :, 0, :]  # 处理输入数据
            labels = labels[:, :, 0, T - target_T:]  # 处理标签数据

            optimizer.zero_grad()  # 梯度清零
            # 调用了Grelen类的forward方法。在类的实例化对象后加上括号，相当于调用__call__方法，
            # __call__方法中调用了forward函数，params实现了前向传播
            # prob是图结构的概率分布,output是时间序列预测的输出
            prob, output, adj_out = net(encoder_inputs)

            # 计算KL散度损失和负对数似然损失
            # loss_kl = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(device)
            # loss_nll = nll_gaussian(output, labels, variation).to(device)
            # loss = loss_kl + loss_nll  # 总损失
            # nll_train.append(loss_nll)  # 记录负对数似然损失
            # kl_train.append(loss_kl)  # 记录KL散度损失

            print("main_loss#############################################")
            output_reshaped = output.view(-1, output.shape[-1])  # [128 * 51, 29]
            labels_reshaped = labels.view(-1, labels.shape[-1])  # [128 * 51, 29]

            loss_pool = net.loss(output_reshaped,labels_reshaped[0],adj=adj_out)
            # loss_pool = net.loss(output,labels,adj=adj_out)
            loss_pool_train.append(loss_pool)
            loss = loss_pool_train
            print(f"loss:{loss}")
            # 反向传播和优化
            loss.to(device)
            loss.backward()  # 反向传播
            optimizer.step()  # 优化一步

            # 记录训练损失
            training_loss = loss.item()
            global_step += 1
            # sw.add_scalar('training_loss', training_loss, global_step)  # 记录训练损失
            # sw.add_scalar('kl_loss', loss_kl.item(), global_step)  # 记录KL损失
            # sw.add_scalar('nll_loss', loss_nll.item(), global_step)  # 记录负对数似然损失
            sw.add_scalar('loss_pool', loss_pool_train.item(), global_step)  # 记录负对数似然损失

        # 打印并记录每个epoch的平均KL损失和负对数似然损失
        # nll_train_ = torch.tensor(nll_train)
        # kl_train_ = torch.tensor(kl_train)
        loss_pool_train_ = torch.tensor(loss_pool_train)
        # print('epoch: %s, kl loss: %.2f, nll loss: %.2f' % (epoch, kl_train_.mean(), nll_train_.mean()))
        # logging.info('epoch: %s, kl loss: %.2f, nll loss: %.2f' % (epoch, kl_train_.mean(), nll_train_.mean()))
        print('epoch: %s, loss_pool_train_: %.2f' % (epoch, loss_pool_train_.mean()))
        logging.info('epoch: %s, loss_pool_train_: %.2f' % (epoch, loss_pool_train_.mean()))

        # 调整学习率
        if epoch == 30:
            adjust_learning_rate(optimizer, 0.0002)
        if epoch == 100:
            adjust_learning_rate(optimizer, 0.0001)

        elapsed = time() - begin_time
        total_time += elapsed
        print(f"start epoch:{start_epoch},elapsed time:{elapsed},total time:{total_time}")
        logging.info(f"start epoch:{start_epoch},elapsed time:{elapsed},total time:{total_time}")
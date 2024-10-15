from os import path  # 导入os模块中的path模块,用于文件路径操作
import torch  # 导入PyTorch库
import pandas as pd  # 导入Pandas库,用于数据处理
import numpy as np  # 导入NumPy库,用于数值计算
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score  # 导入评价指标

from lib.utils import *  # 导入自定义工具函数
from model.GRELEN import *  # 导入GRELEN模型

# import logging
# # 日志配置
# logging.basicConfig(filename='test_grelen.log', level=logging.INFO,
#                     format='%(asctime)s:%(levelname)s:%(message)s')
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='./log/test_grelen.log',
                    filemode='a')

if __name__ == '__main__':
    from config_files.SWAT_config import Config  # 导入配置文件
    from torch.utils.tensorboard import SummaryWriter
    # 创建 TensorBoard 的 SummaryWriter 对象
    writer = SummaryWriter(log_dir='loss')

    config = Config()  # 加载配置

    device = config.device  # 获取设备（CPU或GPU）
    batch_size = config.batch_size  # 获取批量大小
    learning_rate = config.learning_rate  # 获取学习率
    epochs = config.epochs  # 获取训练轮数
    test_filename = config.test_filename  # 获取测试文件名
    # batch_size*10是为了在测试阶段加载更多的数据批次,希望一次性处理更多的数据
    test_loader, test_target_tensor, _mean, _std = load_data_test(test_filename, device, batch_size*10)  # 加载测试数据

    # 获取配置中的参数
    B, N, T, target_T = config.B, config.N, config.T, config.target_T  # B:批次大小, N:节点数, T:输入序列长度, target_T:目标序列长度
    n_in = config.n_in # 输入特征数
    n_hid = config.n_hid # 隐藏层维度
    do_prob = config.do_prob  # dropout概率

    Graph_learner_n_hid = config.Graph_learner_n_hid  # 图学习器隐藏层维度
    Graph_learner_n_head_dim = config.Graph_learner_n_head_dim  # 图学习器头部维度
    Graph_learner_head = config.Graph_learner_head  # 图学习器头数
    prior = config.prior  # 先验概率
    temperature = config.temperature  # 温度参数
    GRU_n_dim = config.GRU_n_dim  # GRU维度
    max_diffusion_step = config.max_diffusion_step  # 最大扩散步数
    num_rnn_layers = config.num_rnn_layers  # RNN层数

    start_epoch = config.start_epoch  # 起始训练周期
    params_path = config.save_dir  # 保存目录

    num_nodes = N
    hard = 'True'
    filter_type = 'random'
    variation = 1

    # 转换并加载先验概率
    # 对先验概率prior进行对数转换,添加维度使其形状适合后续
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior).cuda()

    # 初始化模型
    net = Grelen(config.device, T, target_T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head,
                 temperature, hard, GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type, do_prob=0.).to(config.device)

    # 加载模型参数 self.param_file = "experiments/swat_test/test_model.params"
    param_file = config.param_file
    net.load_state_dict(torch.load(param_file))
    logging.info(f"param file: {param_file}")

    print('Model loaded...')
    logging.info('Model loaded...')

    target_tensor = torch.zeros((0, N, target_T))   # 用于存储目标输出（真实值）
    reconstructed_tensor = torch.zeros((0, N, target_T))    # 用于存储模型的重构输出（预测值）
    # 初始化张量用于存储测试结果,N*(N-1)表示图中所有节点对(去除自连接)的数量
    prob_tensor = torch.zeros((0, N * (N - 1), Graph_learner_head)) # 用于存储图学习器输出的概率（节点关系的概率矩阵）
    # print(f"prob tensor shape:{prob_tensor.shape}") # prob tensor shape:torch.Size([0, 2550, 4])
    # 测试模型
    with torch.no_grad():   # 禁用梯度计算
        # 加载测试数据,每次加载一个批次,batch_index当前批次的索引,batch_data当前批次数据
        for batch_index, batch_data in enumerate(test_loader):
            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs[:, :, 0, :]
            labels = labels[:, :, 0, 1:]
            # prob:模型学习到的节点之间的关系概率矩阵;output:模型的预测输出
            prob, output = net(encoder_inputs)
            # print(f"prob shape:{prob.shape},output shape:{output.shape}")   # prob shape:torch.Size([1280, 2550, 4]),output shape:torch.Size([1280, 51, 29])
            # 将当前批次的prob拼接到prob_tensor中
            prob_tensor = torch.cat([prob_tensor, prob.cpu()], dim=0)

    # 将累积的概率张量prob_tensor从GPU移动到CPU,去除计算图,并转换为Numpy数据
    prob_result = prob_tensor.cpu().detach().numpy()
    # 调整边的形状，得到测试图矩阵，表示节点之间的关系矩阵
    mat_test = reshape_edges(prob_tensor, N).cpu()

    # 保存结果,将节点间的关系矩阵mat_test保存为.npy文件
    save_path = path.dirname(param_file)
    if config.save_result:
        np.save(save_path + '/' + os.path.basename(param_file).split('.')[0] + '.npy', np.array(mat_test))
    print(f'Graph saved: {save_path}/{os.path.basename(param_file).split(".")[0]}.npy')
    logging.info(f'Graph saved: {save_path}/{os.path.basename(param_file).split(".")[0]}.npy')

    ### 评估模型
    print('Evaluation...')
    logging.info('Evaluation...')
    w = config.moving_window_  # 从配置中获取移动窗口大小,用于计算移动平均值
    anomaly_time = pd.read_csv(config.anomaly_file)  # 从指定的异常时间文件中读取数据,文件包含异常发生的起始和结束时间
    anomaly_time = np.array(anomaly_time.iloc[:, :2])  # 提取异常时间数据的前两列,将其转换为NumPy数组
    anomaly_start = anomaly_time[:, 0]  # 提取异常的起始时间
    anomaly_end = anomaly_time[:, 1]  # 提取异常的结束时间

    # 计算移动平均并过滤     # 计算每个节点的出度,对出度进行移动平均滤波,减少噪声
    # moving average filtering是一种信号滤波算法,用于减小信号中的噪声或去除高频成分,从而平滑信号
    total_out_degree_move_filtered = np.zeros((mat_test.shape[1] - w + 1, N))  # 初始化存储移动平均滤波后的结果,数组形状(时间步数,节点数)
    for fe in range(N):  # 遍历每个节点
        y = torch.mean(mat_test[0, :, :, fe], -1)  # 计算节点 `fe` 的出度,取图结构矩阵的均值
        xx = moving_average(y, w)  # 计算 `y` 的移动平均,窗口大小为 `w`
        total_out_degree_move_filtered[:, fe] = y[w-1:] - xx  # 计算出度变化,将 `y` 减去移动平均值后存储

    ############ 对于每个节点计算其在不同时间步上的出度变化,利用移动平均滤波后的结果减去原始的出度,得到出度的变化,对所有节点和时间步计算平均值,得到整体的loss
    """
    loss 反映了节点之间关系（出度）在时间步上的变化。
    通过计算每个时间步上节点之间连通性的变化情况，并与平滑的移动平均值作差，得到每个时间步的“异常程度”。
    这可以用于检测那些明显偏离正常连接模式的时间步，进而标记为异常。
    """
    loss = np.mean(total_out_degree_move_filtered, 1)  # 计算每个时间步的平均损失
    # print(f"loss shape:{loss.size}")   # loss shape:(7469,)

    # 将损失写入 TensorBoard
    for i, l in enumerate(loss):
        writer.add_scalar('Loss', l, i)
    # 关闭 writer
    writer.close()

    f1 = np.zeros((200, 200))  # 初始化F1分数矩阵,用于存储不同阈值组合下的F1得分
    for i in range(200):  # 遍历第一个阈值范围
        for j in range(200):  # 遍历第二个阈值范围
            thr1 = 0.0005 * i  # 根据循环计算第一个阈值
            thr2 = -0.0005 * j  # 根据循环计算第二个阈值
            # 调整预测的异常点并计算F1得分,将结果与真实值比较
            anomaly, ground_truth = point_adjust_eval(anomaly_start, anomaly_end, config.downsampling_fre, loss, thr1, thr2)
            f1[i, j] = f1_score(anomaly, ground_truth)  # 计算并存储当前阈值组合下的F1得分

    pos = np.unravel_index(np.argmax(f1), f1.shape)  # 找到F1得分最高的阈值组合位置
    logging.info(f"最高的阈值组合为{pos}")
    # 使用找到的最佳阈值组合,重新评估异常检测效果,应用点调整策略
    anomaly, ground_truth = point_adjust_eval(anomaly_start, anomaly_end, config.downsampling_fre, loss, pos[0] * 0.0005, -pos[1] * 0.0005)
    # print(f"ground_truth异常标签数量{np.sum(ground_truth==1)},正常标签数量{np.sum(ground_truth==0)}")
    # print(f"anomaly异常标签数量{np.sum(anomaly==1)},正常标签数量{np.sum(anomaly==0)}")

    # 输出并记录评估结果
    print('F1 score: ', f1_score(anomaly, ground_truth))
    print('Precision score: ', precision_score(anomaly, ground_truth))
    print('Recall score: ', recall_score(anomaly, ground_truth))
    print('Confusion matrix: ', classification_report(anomaly, ground_truth))
    # print(f"classification_report后异常标签数量{np.sum(ground_truth==1)},正常标签数量{np.sum(ground_truth==0)}")

    logging.info(f"F1 score: {f1_score(anomaly, ground_truth)}")
    logging.info(f"Precision score: {precision_score(anomaly, ground_truth)}")
    logging.info(f"Recall score: {recall_score(anomaly, ground_truth)}")
    logging.info(f"Confusion matrix: \n{classification_report(anomaly, ground_truth)}")
    
    """
    test_grelen_annotate判断最终的异常
    根据grelen模型输出节点之间的关系概率矩阵,然后拼接得到prob_tensor,调整边的形状生成节点之间的关系矩阵mat_test。
    计算每个节点的出度,对出度进行移动平均滤波,然后对所有节点和时间步计算平均值,得到整体的loss,loss的每个元素代表每个时间步的损失值。
    最后设定阈值thr1和thr2进行判断该时间步是否存在异常,大于thr1或小于thr2视作异常。
    """
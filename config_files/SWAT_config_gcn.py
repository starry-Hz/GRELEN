import numpy as np

# 定义配置类
class Config(object):
    def __init__(self):
        # 数据配置
        self.downsampling_fre = 60  # 数据下采样频率
        self.target_len = 30  # 目标长度

        # 模型配置
        self.device = 'cuda:0'  # 设备
        self.batch_size = 128  # 批处理大小,决定模型在训练过程中一次处理的数据量大小
        self.learning_rate = 0.001  # 学习率
        self.epochs = 200  # 训练周期数
        # self.train_filename = 'data/SWAT/train_swat.npz'  # 训练数据文件名
        # self.test_filename = 'data/SWAT/test_swat.npz'  # 测试数据文件名
        self.train_filename = 'data/SWAT/train_swat_gcn.npz'  # 训练数据文件名
        self.test_filename = 'data/SWAT/test_swat_gcn.npz'  # 测试数据文件名

        # 其他模型相关配置
        self.B, self.N, self.T, self.target_T = 32, 51, 30, 29  # B:批次大小, N:节点数, T:输入序列长度, target_T:目标序列长度
        self.n_in = 12  # 输入特征数
        self.n_hid = 64  # 隐藏层维度
        self.do_prob = 0.  # dropout概率

        # 图学习器相关配置
        self.Graph_learner_n_hid = 64  # 图学习器隐藏层维度
        self.Graph_learner_n_head_dim = 32  # 图学习器头部维度
        self.Graph_learner_head = 4  # 图学习器头数
        self.prior = np.array([0.91, 0.03, 0.03, 0.03])  # 先验概率
        self.temperature = 0.5  # 温度参数
        self.GRU_n_dim = 64  # GRU维度
        self.max_diffusion_step = 2  # 最大扩散步数
        self.num_rnn_layers = 1  # RNN层数

        # 保存相关配置
        self.save_dir = 'experiments/swat_test'  # 保存目录
        self.start_epoch = 0  # 起始训练周期
        self.param_file = "experiments/swat_test/epoch_9.params"  # 模型参数文件
        self.save_result = True  # 是否保存结果
        self.moving_window_ = 2  # 移动窗口大小
        self.anomaly_file = 'data/SWAT/SWAT_Time.csv'  # 异常数据文件

import logging  # 导入日志模块
import numpy as np  # 导入NumPy库，用于科学计算
import os  # 导入OS模块，用于文件操作
import pickle  # 导入Pickle模块，用于序列化
import scipy.sparse as sp  # 导入SciPy稀疏矩阵模块
import sys  # 导入Sys模块，用于系统操作
import tensorflow as tf  # 导入TensorFlow库
import torch  # 导入PyTorch库
from scipy.sparse import linalg  # 从SciPy稀疏矩阵模块中导入线性代数模块
from torch.autograd import Variable  # 从PyTorch自动求导模块中导入变量
import torch.nn.functional as F  # 导入PyTorch中的功能模块

# 设置设备为GPU（如果可用），否则为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import logging
# # 日志配置
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s:%(levelname)s:%(message)s')

import logging

# 引用主程序中的日志记录器
logger = logging.getLogger(__name__)

def sample_gumbel(shape, eps=1e-10):
    """
    从Gumbel(0, 1)分布中采样

    基于
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ，
    (MIT license)
    """
    U = torch.rand(shape).float().to(device)  # 从均匀分布中采样
    return -torch.log(eps - torch.log(U + eps))  # 返回Gumbel分布样本

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    从Gumbel-Softmax分布中采样

    基于
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(device)  # 生成Gumbel噪声
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()  # 将噪声转移到GPU
    y = logits + Variable(gumbel_noise)  # 添加Gumbel噪声
    return F.softmax(y / tau, dim=-1)  # 返回Gumbel-Softmax样本

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    从Gumbel-Softmax分布中采样并可选地离散化
    3.5 Sampling 使用Gumbel-softmax分类重参数技巧

    基于
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ，
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)  # 生成Gumbel-Softmax软样本
    if hard:
        shape = logits.size()  # 获取logits的尺寸
        _, k = y_soft.data.max(-1)  # 获取最大值的索引
        y_hard = torch.zeros(*shape).to(device)  # 创建全零Tensor
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()  # 将y_hard转移到GPU
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0).to(device)  # 将最大值的位置设为1
        y = Variable(y_hard - y_soft.data) + y_soft  # 返回一热编码和软样本的组合
    else:
        y = y_soft  # 返回软样本
    return y  # 返回样本

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """
    计算KL散度
    ********KL散度损失,3.7中的L2********

    preds: 预测分布
    log_prior: 先验分布的对数
    num_atoms: 原子数量
    eps: 小常数，防止数值问题
    """
    kl_div = preds * (torch.log(preds + eps) - log_prior)  # 计算KL散度
    return kl_div.sum() / (num_atoms * preds.size(0))  # 返回归一化的KL散度

def nll_gaussian(preds, target, variance, add_const=False):
    """
    计算高斯负对数似然
    ********总重建损失,3.7中的L1********
    评估预测值与真实值之间的误差

    preds: 预测值 (tensor)
    target: 目标值 (tensor)
    variance: 方差 (float)
    add_const: 是否添加常数项 (bool)
    """
    # 计算负对数似然
    neg_log_p = ((preds - target) ** 2 / (2 * variance))  # 计算平方误差并除以2倍方差

    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)  # 计算常数项 0.5 * log(2 * pi * variance)
        neg_log_p += const  # 添加常数项到负对数似然中

    # 返回归一化的负对数似然
    return neg_log_p.sum() / (target.size(0) * target.size(1))  # 求和并除以目标值的元素数量（批量大小和序列长度的乘积）

def adjust_learning_rate(optimizer, lr):
    """
    调整学习率

    optimizer: 优化器
    lr: 新的学习率
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 设置新的学习率

def reshape_edges(edges, N):
    """
    重塑边的张量

    edges: 边的张量 (B, N(N-1), type)
    N: 节点数量
    """
    edges = torch.tensor(edges)  # 转换为张量
    mat = torch.zeros(edges.shape[-1], edges.shape[0], N, N) + 1  # 初始化矩阵
    mask = ~torch.eye(N, dtype=bool).unsqueeze(0).unsqueeze(0)  # 创建掩码
    mask = mask.repeat(edges.shape[-1], edges.shape[0], 1, 1)  # 扩展掩码
    mat[mask] = edges.permute(2, 0, 1).flatten()  # 将边填充到矩阵中
    return mat  # 返回重塑后的矩阵

def load_data_train(filename, DEVICE, batch_size, shuffle=True):
    """
    加载训练数据

    filename: 数据文件名
    DEVICE: 设备
    batch_size: 批量大小
    shuffle: 是否打乱数据
    """
    file_data = np.load(filename)  # 加载文件数据
    train_x = file_data['train_x']  # 获取训练输入数据
    train_x = train_x[:, :, 0:1, :]  # 选择输入数据的特定通道
    train_target = file_data['train_target']  # 获取训练目标数据

    val_x = file_data['val_x']  # 获取验证输入数据
    val_x = val_x[:, :, 0:1, :]  # 选择输入数据的特定通道
    val_target = file_data['val_target']  # 获取验证目标数据

    mean = file_data['mean'][:, :, 0:1, :]  # 获取均值
    std = file_data['std'][:, :, 0:1, :]  # 获取标准差

    # ------- 训练数据加载器 -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # 将训练输入数据转换为Tensor并转移到设备
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # 将训练目标数据转换为Tensor并转移到设备
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)  # 创建训练数据集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)  # 创建训练数据加载器

    # 创建PyTorch的TensorDataset，将输入和目标数据组合为一个数据集，再通过DataLoader将数据集包装起来，以便分批次迭代使用

    # ------- 验证数据加载器 -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # 将验证输入数据转换为Tensor并转移到设备
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # 将验证目标数据转换为Tensor并转移到设备
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)  # 创建验证数据集
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 创建验证数据加载器

    # 打印训练和验证数据的尺寸
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    logging.info(f'train: {train_x_tensor.size()}, {train_target_tensor.size()}')
    logging.info(f'val: {val_x_tensor.size()}, {val_target_tensor.size()}')
    """
    2024-09-09 10:34:23,543 - root - INFO - train: torch.Size([4195, 51, 1, 30]), torch.Size([4195, 51, 1, 30])
    2024-09-09 10:34:23,544 - root - INFO - val: torch.Size([1027, 51, 1, 30]), torch.Size([1027, 51, 1, 30])
    """

    # 返回训练数据加载器、训练目标数据、验证数据加载器、验证目标数据、均值和标准差
    return train_loader, train_target_tensor, val_loader, val_target_tensor, mean, std

def load_data_test(filename, DEVICE, batch_size, shuffle=False):
    """
    加载测试数据

    filename: 数据文件名
    DEVICE: 设备
    batch_size: 批量大小
    shuffle: 是否打乱数据
    """
    file_data = np.load(filename)  # 加载文件数据
    train_x = file_data['test_x']  # 获取测试输入数据
    train_target = file_data['test_target']  # 获取测试目标数据
    train_x = train_x[:, :, 0:1, :]  # 选择输入数据的特定通道

    mean = file_data['mean'][:, :, 0:1, :]  # 获取均值
    std = file_data['std'][:, :, 0:1, :]  # 获取标准差

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # 将测试输入数据转换为Tensor并转移到设备
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # 将测试目标数据转换为Tensor并转移到设备
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)  # 创建测试数据集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)  # 创建测试数据加载器

    # 返回测试数据加载器、测试目标数据、均值和标准差
    return train_loader, train_target_tensor, mean, std

def moving_average(x, w):
    """
    计算移动平均值

    x: 输入数据 (array-like)
    w: 窗口大小 (int)
    """
    # 使用卷积计算移动平均值
    return np.convolve(x, np.ones(w), 'valid') / w  # 返回移动平均值

def point_adjust_eval(anomaly_start, anomaly_end, down, loss, thr1, thr2):
    """
    调整点评估

    anomaly_start: 异常开始点
    anomaly_end: 异常结束点
    down: 下采样率
    loss: 损失
    thr1: 阈值1
    thr2: 阈值2
    """
    len_ = int(anomaly_end[-1]/down)+1  # 计算长度
    anomaly = np.zeros((len_))  # 初始化异常数组
    loss = loss[:len_]  # 裁剪损失数组
    anomaly[np.where(loss>thr1)] = 1  # 标记大于阈值1的点
    anomaly[np.where(loss<thr2)] = 1  # 标记小于阈值2的点
    ground_truth = np.zeros((len_))  # 初始化真实值数组
    for i in range(len(anomaly_start)):
        ground_truth[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)+1] = 1  # 标记异常区域
        if np.sum(anomaly[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)])>0:
            anomaly[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)] = 1  # 调整异常区域
        anomaly[int(anomaly_start[i]/down)] = ground_truth[int(anomaly_start[i]/down)]  # 调整开始点
        anomaly[int(anomaly_end[i]/down)] = ground_truth[int(anomaly_end[i]/down)]  # 调整结束点

    # anomaly调整后的异常检测结果;  ground_truth实际的异常标记
    return anomaly, ground_truth  # 返回调整后的异常和真实值

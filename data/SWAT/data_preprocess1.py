import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于数值计算
import copy  # 导入copy库，用于深拷贝操作
import sys  # 导入sys库，用于系统相关操作

def downsampling(mat, interval):
    """
    对矩阵进行降采样
    :param mat: 输入矩阵
    :param interval: 采样间隔
    :return: 降采样后的矩阵
    """
    num_row, num_col = mat.shape  # 获取矩阵的行数和列数
    res = num_row % interval  # 计算行数对采样间隔的余数
    if res != 0:  # 如果余数不为0，需要补充一些行使其可以整除
        add_num = interval - res  # 计算需要补充的行数
        add_mat = np.zeros((add_num, num_col))  # 创建全零矩阵进行补充
        mat = np.concatenate((mat, add_mat))  # 将补充后的矩阵与原矩阵连接
    num_row, num_col = mat.shape  # 重新获取补充后的矩阵的行数和列数
    mat_tmp = np.zeros((interval, int(num_row / interval), num_col))  # 创建临时矩阵
    for i in range(interval):  # 循环间隔次数
        mat_tmp[i, ...] = mat[i::interval, :]  # 对每个间隔进行采样
    return np.mean(mat_tmp, 0)  # 返回降采样后的矩阵

def max_min_norm(mat, max_=None, min_=None):
    """
    对矩阵进行最大-最小归一化
    :param mat: 输入矩阵
    :param max_: 最大值
    :param min_: 最小值
    :return: 归一化后的矩阵，最大值，最小值
    """
    if max_ is None:  # 如果没有提供最大值
        max_ = np.max(mat, 0)  # 计算每列的最大值
    if min_ is None:  # 如果没有提供最小值
        min_ = np.min(mat, 0)  # 计算每列的最小值
    nrow, ncol = mat.shape  # 获取矩阵的行数和列数
    for i in range(ncol):  # 对每一列进行归一化处理
        if max_[i] == min_[i]:  # 如果最大值等于最小值
            mat[:, i] = mat[:, i] - min_[i]  # 将该列的值减去最小值
        else:
            mat[:, i] = (mat[:, i] - min_[i]) / (max_[i] - min_[i])  # 归一化处理
    return mat, max_, min_  # 返回归一化后的矩阵，最大值，最小值

def swat_generate(xx, split, length, filename=None, max_=None, min_=None):
    """
    生成SWAT数据集
    :param xx: 输入数据
    :param split: 训练集与验证集的划分比例
    :param length: 序列长度
    :param filename: 输出文件名
    :param max_: 最大值
    :param min_: 最小值
    :return: 最大值，最小值，生成的数据
    """
    mat_, max_, min_ = max_min_norm(xx, max_, min_)  # 对输入数据进行归一化处理
    nrow, ncol = xx.shape  # 获取输入数据的行数和列数

    xx1 = xx[:int(nrow * split), :]  # 按照划分比例获取训练集数据
    xx2 = xx[int(nrow * split):, :]  # 按照划分比例获取验证集数据
    train_x = np.zeros((xx1.shape[0] - length + 1, length, ncol))  # 创建训练集样本矩阵
    for i in range(train_x.shape[0]):  # 生成训练集样本
        train_x[i, ...] = xx1[i:i + length, :]
    valid_x = np.zeros((xx2.shape[0] - length + 1, length, ncol))  # 创建验证集样本矩阵
    for i in range(valid_x.shape[0]):  # 生成验证集样本
        valid_x[i, ...] = xx2[i:i + length, :]
    all_data = {  # 整理所有生成的数据
        'train': {
            'x': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 训练集输入数据
            'target': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 训练集目标数据
        },
        'val': {
            'x': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),  # 验证集输入数据
            'target': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),  # 验证集目标数据
        },
        'stats': {
            '_mean': np.zeros((1, 1, 3, 1)),  # 统计信息：均值
            '_std': np.zeros((1, 1, 3, 1)),  # 统计信息：标准差
        }
    }
    if filename is None:  # 如果没有提供文件名
        return max_, min_, all_data  # 返回生成的数据
    np.savez_compressed(filename,
                        train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                        val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                        )  # 将生成的数据保存到压缩文件中

    return max_, min_, all_data  # 返回最大值，最小值和生成的数据

def swat_generate_test(xx, length, filename=None, max_=None, min_=None):
    """
    生成SWAT测试数据集
    :param xx: 输入数据
    :param length: 序列长度
    :param filename: 输出文件名
    :param max_: 最大值
    :param min_: 最小值
    :return: 最大值，最小值，生成的数据
    """
    mat_, max_, min_ = max_min_norm(xx, max_, min_)  # 对输入数据进行归一化处理
    nrow, ncol = xx.shape  # 获取输入数据的行数和列数

    train_x = np.zeros((xx.shape[0] - length + 1, length, ncol))  # 创建测试集样本矩阵
    for i in range(train_x.shape[0]):  # 生成测试集样本
        train_x[i, ...] = xx[i:i + length, :]
    all_data = {  # 整理所有生成的数据
        'test': {
            'x': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 测试集输入数据
            'target': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 测试集目标数据
        },
        'stats': {
            '_mean': np.zeros((1, 1, 3, 1)),  # 统计信息：均值
            '_std': np.zeros((1, 1, 3, 1)),  # 统计信息：标准差
        }
    }
    if filename is None:  # 如果没有提供文件名
        return max_, min_, all_data  # 返回生成的数据
    np.savez_compressed(filename,
                        test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                        )  # 将生成的数据保存到压缩文件中

    return max_, min_, all_data  # 返回最大值，最小值和生成的数据

if __name__ == '__main__':
    sys.path.append("../..")  # 添加项目根目录到系统路径
    from config_files.SWAT_config import Config  # 从配置文件导入配置

    config = Config()  # 初始化配置
    swat_normal = pd.read_csv('SWaT_Dataset_Normal_v0.csv')  # 读取正常的SWAT数据集
    swat_abnormal = pd.read_csv('SWaT_Dataset_Attack_v0.csv')  # 读取异常的SWAT数据集
    swat_normal_np = np.array(swat_normal.iloc[:, 1: -1])  # 将正常数据集转换为numpy数组
    swat_abnormal_np = np.array(swat_abnormal.iloc[:, 1: -1])  # 将异常数据集转换为numpy数组

    train_x = downsampling(swat_normal_np, config.downsampling_fre)[3000:, :]  # 对正常数据进行降采样和预处理
    split = 0.8  # 训练集和验证集划分比例
    length = config.target_len  # 序列长度
    max_, min_, all_data = swat_generate(copy.copy(train_x), split, length, 'train_swat.npz')  # 生成训练和验证数据集

    test_x = downsampling(swat_abnormal_np, config.downsampling_fre)  # 对异常数据进行降采样和预处理
    max_, min_, all_data_test = swat_generate_test(copy.copy(test_x), length, 'test_swat.npz', copy.copy(max_), copy.copy(min_))  # 生成测试数据集

    test_gr = all_data_test['test']['x']  # 加载生成的测试数据
    test_old = np.load("/home/zwq/Test/test_60_complete (1).npz")['train_x']  # 加载旧的测试数据

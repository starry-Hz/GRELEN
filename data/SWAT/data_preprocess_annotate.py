import pandas as pd  # 导入pandas库,用于数据处理（如读取CSV文件,数据表格操作等）
import numpy as np  # 导入numpy库,用于高效的数组和矩阵运算
import copy  # 导入copy库,用于深拷贝对象,确保修改副本时不会影响原对象
import sys  # 导入sys库,用于与Python解释器进行交互（如修改路径、获取命令行参数）

def downsampling(mat, interval):
    """
    对矩阵进行降采样,根据指定的间隔,将矩阵的行数缩减为每interval取一行。
    :param mat: 输入的二维矩阵
    :param interval: 采样间隔
    :return: 降采样后的矩阵
    """
    num_row, num_col = mat.shape  # 获取矩阵的行数和列数
    res = num_row % interval  # 计算矩阵行数对采样间隔的余数
    if res != 0:  # 如果行数不能被采样间隔整除
        add_num = interval - res  # 计算需要补充的行数
        add_mat = np.zeros((add_num, num_col))  # 创建一个全零的矩阵进行补充
        mat = np.concatenate((mat, add_mat))  # 将原矩阵与补充的全零矩阵拼接在一起
    num_row, num_col = mat.shape  # 获取补充后的矩阵的行数和列数
    mat_tmp = np.zeros((interval, int(num_row / interval), num_col))  # 创建一个临时矩阵用于存储降采样数据
    for i in range(interval):  # 遍历每个采样间隔
        mat_tmp[i, ...] = mat[i::interval, :]  # 每隔interval行采样一次,存储到临时矩阵中
    return np.mean(mat_tmp, 0)  # 对每个采样间隔的数据取均值,返回降采样后的矩阵

def max_min_norm(mat, max_=None, min_=None):
    """
    对矩阵进行最大-最小归一化,将数据缩放到[0,1]区间。
    :param mat: 输入矩阵
    :param max_: 每列的最大值（可选）
    :param min_: 每列的最小值（可选）
    :return: 归一化后的矩阵,以及最大值和最小值
    """
    if max_ is None:  # 如果没有提供最大值
        max_ = np.max(mat, 0)  # 计算每列的最大值
    if min_ is None:  # 如果没有提供最小值
        min_ = np.min(mat, 0)  # 计算每列的最小值
    nrow, ncol = mat.shape  # 获取矩阵的行数和列数
    for i in range(ncol):  # 遍历每一列
        if max_[i] == min_[i]:  # 如果最大值等于最小值
            mat[:, i] = mat[:, i] - min_[i]  # 该列的所有值减去最小值
        else:
            mat[:, i] = (mat[:, i] - min_[i]) / (max_[i] - min_[i])  # 归一化处理：减去最小值并除以范围
    return mat, max_, min_  # 返回归一化后的矩阵,最大值和最小值

def swat_generate(xx, split, length, filename=None, max_=None, min_=None):
    """
    生成SWAT数据集的训练集和验证集。
    :param xx: 输入数据矩阵
    :param split: 训练集和验证集划分的比例
    :param length: 序列长度
    :param filename: 保存文件名（可选）
    :param max_: 最大值（可选）
    :param min_: 最小值（可选）
    :return: 归一化后的最大值、最小值和生成的数据集
    """
    mat_, max_, min_ = max_min_norm(xx, max_, min_)  # 对输入数据进行最大-最小归一化处理
    nrow, ncol = xx.shape  # 获取矩阵的行数和列数

    xx1 = xx[:int(nrow * split), :]  # 按照划分比例获取训练集数据
    xx2 = xx[int(nrow * split):, :]  # 按照划分比例获取验证集数据
    
    # train_x存储训练集样本数据
    """
    第一维样本数量xx1.shape[0] - length + 1,即总行数-序列长度+1,训练集中包含多少个时间序列样本,多少个窗口;
    第二维序列长度length,每个样本包含多少个时间点;
    第三维是列数col,每个样本的特征维度
    train_x.shape = (样本数量,时间步长,特征数)  许多深度学习模型(如RNN,LSTM,CNN)中,通常需要具有特定的维度,
    """
    train_x = np.zeros((xx1.shape[0] - length + 1, length, ncol))  # 初始化训练集样本矩阵
    for i in range(train_x.shape[0]):  # 生成训练集样本
        train_x[i, ...] = xx1[i:i + length, :]
    valid_x = np.zeros((xx2.shape[0] - length + 1, length, ncol))  # 初始化验证集样本矩阵
    for i in range(valid_x.shape[0]):  # 生成验证集样本
        valid_x[i, ...] = xx2[i:i + length, :]
    
    # np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2)中
    # np.transpose是进行维度转置
    # np.expand_dims为每个样本添加一个额外的维度,确保数据结构复合深度学习模型的格式
    """
    数据形状变化过程
    train_x形状(samples,timesteps,features)
    ——通过np.transpose(train_x, (0, 2, 1))之后变为(samples,features,timesteps)
    ——通过np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2)之后变为(samples,features,1,timesteps)
    """
    all_data = {  # 生成包含训练集、验证集和统计信息的数据字典
        'train': {
            'x': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 训练集输入数据
            'target': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 训练集目标数据
        },
        'val': {
            'x': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),  # 验证集输入数据
            'target': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),  # 验证集目标数据
        },
        'stats': {
            '_mean': np.zeros((1, 1, 3, 1)),  # 初始化均值（可以后续用于标准化）
            '_std': np.zeros((1, 1, 3, 1)),  # 初始化标准差（可以后续用于标准化）
        }
    }
    if filename is None:  # 如果没有提供文件名
        return max_, min_, all_data  # 返回最大值、最小值和生成的数据集
    # 存储为.npz格式的文件
    np.savez_compressed(filename,
                        train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                        val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                        )  # 将生成的数据保存到压缩文件中

    return max_, min_, all_data  # 返回最大值、最小值和生成的数据集

def swat_generate_test(xx, length, filename=None, max_=None, min_=None):
    """
    生成SWAT测试数据集。
    :param xx: 输入数据矩阵
    :param length: 序列长度
    :param filename: 保存文件名（可选）
    :param max_: 最大值（可选）
    :param min_: 最小值（可选）
    :return: 归一化后的最大值、最小值和生成的测试数据集
    """
    mat_, max_, min_ = max_min_norm(xx, max_, min_)  # 对输入数据进行最大-最小归一化处理
    nrow, ncol = xx.shape  # 获取矩阵的行数和列数

    train_x = np.zeros((xx.shape[0] - length + 1, length, ncol))  # 初始化测试集样本矩阵
    for i in range(train_x.shape[0]):  # 生成测试集样本
        train_x[i, ...] = xx[i:i + length, :]
    all_data = {  # 生成包含测试集和统计信息的数据字典
        'test': {
            'x': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 测试集输入数据
            'target': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 测试集目标数据
        },
        'stats': {
            '_mean': np.zeros((1, 1, 3, 1)),  # 初始化均值
            '_std': np.zeros((1, 1, 3, 1)),  # 初始化标准差
        }
    }
    if filename is None:  # 如果没有提供文件名
        return max_, min_, all_data  # 返回最大值、最小值和生成的测试数据集
    np.savez_compressed(filename,
                        test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                        )  # 将生成的测试数据保存到压缩文件中

    return max_, min_, all_data  # 返回最大值、最小值和生成的测试数据集

if __name__ == '__main__':
    sys.path.append("../..")  # 将项目上两级目录添加到系统路径中，方便导入模块
    from config_files.SWAT_config import Config  # 从配置文件中导入Config类
    config = Config()  # 初始化配置对象，读取配置参数
    swat_normal = pd.read_csv('SWaT_Dataset_Normal_v0.csv')  # 读取正常的SWaT数据集
    swat_abnormal = pd.read_csv('SWaT_Dataset_Attack_v0.csv')  # 读取异常的SWaT数据集
    
    swat_normal_np = np.array(swat_normal.iloc[:, 1: -1])  # 将正常数据集的第2列到倒数第2列转换为NumPy数组
    swat_abnormal_np = np.array(swat_abnormal.iloc[:, 1: -1])  # 将异常数据集的第2列到倒数第2列转换为NumPy数组

    # 对正常数据集进行降采样处理，并丢弃前3000行数据
    train_x = downsampling(swat_normal_np, config.downsampling_fre)[3000:, :]
    
    split = 0.8  # 训练集和验证集的划分比例，80%数据用于训练
    length = config.target_len  # 配置文件中指定的序列长度
    
    # 生成训练和验证数据，并将其保存到 'train_swat.npz' 文件中
    # 同时返回标准化所需的最大值和最小值，以及生成的所有数据
    max_, min_, all_data = swat_generate(copy.copy(train_x), split, length, 'train_swat.npz')
    
    # 对异常数据集进行降采样处理
    test_x = downsampling(swat_abnormal_np, config.downsampling_fre)
    
    # 生成测试数据，使用训练数据集的最大值和最小值进行标准化，并保存到 'test_swat.npz' 文件中
    max_, min_, all_data_test = swat_generate_test(copy.copy(test_x), length, 'test_swat.npz', copy.copy(max_), copy.copy(min_))

    # 从生成的测试数据中提取测试集输入数据
    test_gr = all_data_test['test']['x']
    
    # 从已有的测试数据文件中加载训练集数据
    test_old = np.load("/home/zwq/Test/test_60_complete (1).npz")['train_x']


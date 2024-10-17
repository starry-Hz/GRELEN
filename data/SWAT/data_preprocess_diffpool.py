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

def swat_generate_data(xx, train_split, val_split, length, filename=None, max_=None, min_=None):
    # 对输入数据进行最大-最小归一化处理
    mat_, max_, min_ = max_min_norm(xx, max_, min_)
    nrow, ncol = xx.shape  # 获取数据的行数和列数
    
    # 按照划分比例分别获取训练集、验证集和测试集
    train_end = int(nrow * train_split)
    val_end = int(nrow * (train_split + val_split))
    
    xx_train = xx[:train_end, :]  # 训练集数据
    xx_val = xx[train_end:val_end, :]  # 验证集数据
    xx_test = xx[val_end:, :]  # 测试集数据

    # 生成训练集样本
    train_x = np.zeros((xx_train.shape[0] - length + 1, length, ncol))
    for i in range(train_x.shape[0]):
        train_x[i, ...] = xx_train[i:i + length, :]

    # 生成验证集样本
    valid_x = np.zeros((xx_val.shape[0] - length + 1, length, ncol))
    for i in range(valid_x.shape[0]):
        valid_x[i, ...] = xx_val[i:i + length, :]

    # 生成测试集样本
    test_x = np.zeros((xx_test.shape[0] - length + 1, length, ncol))
    for i in range(test_x.shape[0]):
        test_x[i, ...] = xx_test[i:i + length, :]

    # 构建数据字典
    all_data = {
        'train': {
            'x': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 训练集输入数据
            'target': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # 训练集目标数据
        },
        'val': {
            'x': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),  # 验证集输入数据
            'target': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),  # 验证集目标数据
        },
        'test': {
            'x': np.expand_dims(np.transpose(test_x, (0, 2, 1)), 2),  # 测试集输入数据
            'target': np.expand_dims(np.transpose(test_x, (0, 2, 1)), 2),  # 测试集目标数据
        },
        'stats': {
            '_mean': np.zeros((1, 1, 3, 1)),  # 初始化均值
            '_std': np.zeros((1, 1, 3, 1)),  # 初始化标准差
        }
    }

    # 如果没有提供文件名，就直接返回数据
    if filename is None:
        return max_, min_, all_data

    # 将数据保存为.npz文件
    np.savez_compressed(filename,
                        train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                        val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                        test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std'])

    return max_, min_, all_data

if __name__ == '__main__':
    import sys
    import os
    import pandas as pd
    import numpy as np
    import copy

    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 获取 config_files 目录的绝对路径
    config_files_dir = os.path.abspath(os.path.join(current_dir, '../../'))

    # 将 config_files 路径添加到 sys.path 中
    sys.path.append(config_files_dir)

    # 从 SWAT_config 文件中导入 Config 类
    from config_files.SWAT_config import Config

    # 初始化配置对象，读取配置参数
    config = Config()



    # 尝试读取正常和异常的数据集
    try:
        swat_normal = pd.read_csv(os.path.join(current_dir, 'SWaT_Dataset_Normal_v1.csv'))  # 读取正常的SWaT数据集
        swat_abnormal = pd.read_csv(os.path.join(current_dir, 'SWaT_Dataset_Attack_v0.csv'))  # 读取异常的SWaT数据集
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)



    # 读取异常的SWaT数据集
    swat_abnormal = pd.read_csv("data/SWAT/SWaT_Dataset_Attack_v0.csv")

    # 新增一列 'Attack_Flag'，如果是 'Attack' 或 'A ttack' 则为 1，否则为 0
    swat_abnormal['Attack_Flag'] = swat_abnormal['Normal/Attack'].apply(lambda x: 1 if x in ['Attack', 'A ttack'] else 0)

    # 获取 'Normal/Attack' 列的索引
    col_index = swat_abnormal.columns.get_loc('Normal/Attack')

    # 将 'Attack_Flag' 插入到 'Normal/Attack' 列之前
    swat_abnormal.insert(col_index, 'Attack_Flag', swat_abnormal.pop('Attack_Flag'))
    swat_abnormal_np = np.array(swat_abnormal.iloc[:, 1: -1])

    split = 0.8  # 训练集和验证集的划分比例，80%数据用于训练
    length = config.target_len  # 配置文件中指定的序列长度

    # 设置输出文件的路径
    data_file = "data/SWAT/"
    data_path = os.path.join(data_file, 'all_data_diffpool.npz')

    data = downsampling(swat_abnormal_np, config.downsampling_fre)
    max_, min_, all_data = swat_generate_data(copy.copy(data), train_split=0.6, val_split=0.2, length=length, filename=data_path)



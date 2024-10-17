import logging  # ������־ģ��
import numpy as np  # ����NumPy�⣬���ڿ�ѧ����
import os  # ����OSģ�飬�����ļ�����
import pickle  # ����Pickleģ�飬�������л�
import scipy.sparse as sp  # ����SciPyϡ�����ģ��
import sys  # ����Sysģ�飬����ϵͳ����
import tensorflow as tf  # ����TensorFlow��
import torch  # ����PyTorch��
from scipy.sparse import linalg  # ��SciPyϡ�����ģ���е������Դ���ģ��
from torch.autograd import Variable  # ��PyTorch�Զ���ģ���е������
import torch.nn.functional as F  # ����PyTorch�еĹ���ģ��

# �����豸ΪGPU��������ã�������ΪCPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import logging
# # ��־����
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s:%(levelname)s:%(message)s')

# import logging

# # �����������е���־��¼��
# logger = logging.getLogger(__name__)

def sample_gumbel(shape, eps=1e-10):
    """
    ��Gumbel(0, 1)�ֲ��в���

    ����
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ��
    (MIT license)
    """
    U = torch.rand(shape).float().to(device)  # �Ӿ��ȷֲ��в���
    return -torch.log(eps - torch.log(U + eps))  # ����Gumbel�ֲ�����

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    ��Gumbel-Softmax�ֲ��в���

    ����
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(device)  # ����Gumbel����
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()  # ������ת�Ƶ�GPU
    y = logits + Variable(gumbel_noise)  # ���Gumbel����
    return F.softmax(y / tau, dim=-1)  # ����Gumbel-Softmax����

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    ��Gumbel-Softmax�ֲ��в�������ѡ����ɢ��
    3.5 Sampling ʹ��Gumbel-softmax�����ز�������

    ����
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ��
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)  # ����Gumbel-Softmax������
    if hard:
        shape = logits.size()  # ��ȡlogits�ĳߴ�
        _, k = y_soft.data.max(-1)  # ��ȡ���ֵ������
        y_hard = torch.zeros(*shape).to(device)  # ����ȫ��Tensor
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()  # ��y_hardת�Ƶ�GPU
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0).to(device)  # �����ֵ��λ����Ϊ1
        y = Variable(y_hard - y_soft.data) + y_soft  # ����һ�ȱ���������������
    else:
        y = y_soft  # ����������
    return y  # ��������

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """
    ����KLɢ��
    ********KLɢ����ʧ,3.7�е�L2********

    preds: Ԥ��ֲ�
    log_prior: ����ֲ��Ķ���
    num_atoms: 
    eps: С��������ֹ��ֵ����
    """
    kl_div = preds * (torch.log(preds + eps) - log_prior)  # ����KLɢ��
    return kl_div.sum() / (num_atoms * preds.size(0))  # ���ع�һ����KLɢ��

def nll_gaussian(preds, target, variance, add_const=False):
    """
    �����˹��������Ȼ
    ********���ؽ���ʧ,3.7�е�L1********
    ����Ԥ��ֵ����ʵֵ֮������

    preds: Ԥ��ֵ (tensor)
    target: Ŀ��ֵ (tensor)
    variance: ���� (float)
    add_const: �Ƿ���ӳ����� (bool)
    """
    # ���㸺������Ȼ
    neg_log_p = ((preds - target) ** 2 / (2 * variance))  # ����ƽ��������2������

    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)  # ���㳣���� 0.5 * log(2 * pi * variance)
        neg_log_p += const  # ��ӳ������������Ȼ��

    # ���ع�һ���ĸ�������Ȼ
    return neg_log_p.sum() / (target.size(0) * target.size(1))  # ��Ͳ�����Ŀ��ֵ��Ԫ��������������С�����г��ȵĳ˻���

def adjust_learning_rate(optimizer, lr):
    """
    ����ѧϰ��

    optimizer: �Ż���
    lr: �µ�ѧϰ��
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # �����µ�ѧϰ��

def reshape_edges(edges, N):
    """
    ���ܱߵ�����

    edges: �ߵ����� (B, N(N-1), type)
    N: �ڵ�����
    """
    edges = torch.tensor(edges)  # ת��Ϊ����
    mat = torch.zeros(edges.shape[-1], edges.shape[0], N, N) + 1  # ��ʼ������
    mask = ~torch.eye(N, dtype=bool).unsqueeze(0).unsqueeze(0)  # ��������
    mask = mask.repeat(edges.shape[-1], edges.shape[0], 1, 1)  # ��չ����
    mat[mask] = edges.permute(2, 0, 1).flatten()  # ������䵽������
    return mat  # �������ܺ�ľ���

def load_data_train(filename, DEVICE, batch_size, shuffle=True):
    """
    ����ѵ������

    filename: �����ļ���
    DEVICE: �豸
    batch_size: ������С
    shuffle: �Ƿ��������
    """
    file_data = np.load(filename)  # �����ļ�����
    train_x = file_data['train_x']  # ��ȡѵ����������
    train_x = train_x[:, :, 0:1, :]  # ѡ���������ݵ��ض�ͨ��
    train_target = file_data['train_target']  # ��ȡѵ��Ŀ������

    val_x = file_data['val_x']  # ��ȡ��֤��������
    val_x = val_x[:, :, 0:1, :]  # ѡ���������ݵ��ض�ͨ��
    val_target = file_data['val_target']  # ��ȡ��֤Ŀ������

    mean = file_data['mean'][:, :, 0:1, :]  # ��ȡ��ֵ
    std = file_data['std'][:, :, 0:1, :]  # ��ȡ��׼��

    # ------- ѵ�����ݼ����� -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # ��ѵ����������ת��ΪTensor��ת�Ƶ��豸
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # ��ѵ��Ŀ������ת��ΪTensor��ת�Ƶ��豸
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)  # ����ѵ�����ݼ� 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)  # ����ѵ�����ݼ�����

    # tensorDataset��Ҫ�����ǽ��������(ͨ�����������ݺ�Ŀ������)��ϳ�һ�����ݼ�,����ÿ������ͨ�����������е������ĵ�һ��ά������ȡ
    # ����PyTorch��TensorDataset���������Ŀ���������Ϊһ�����ݼ�����ͨ��DataLoader�����ݼ���װ����,�Ա�����ε���ʹ��,shuffle=True�����������

    # ------- ��֤���ݼ����� -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # ����֤��������ת��ΪTensor��ת�Ƶ��豸
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # ����֤Ŀ������ת��ΪTensor��ת�Ƶ��豸
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)  # ������֤���ݼ�
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # ������֤���ݼ�����

    # ��ӡѵ������֤���ݵĳߴ�
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    logging.info(f'train: {train_x_tensor.size()}, {train_target_tensor.size()}')
    logging.info(f'val: {val_x_tensor.size()}, {val_target_tensor.size()}')
    """
    2024-09-09 10:34:23,543 - root - INFO - train: torch.Size([4195, 51, 1, 30]), torch.Size([4195, 51, 1, 30])
    2024-09-09 10:34:23,544 - root - INFO - val: torch.Size([1027, 51, 1, 30]), torch.Size([1027, 51, 1, 30])
    """

    # ����ѵ�����ݼ�������ѵ��Ŀ�����ݡ���֤���ݼ���������֤Ŀ�����ݡ���ֵ�ͱ�׼��
    return train_loader, train_target_tensor, val_loader, val_target_tensor, mean, std

def load_data_test(filename, DEVICE, batch_size, shuffle=False):
    """
    ���ز�������

    filename: �����ļ���
    DEVICE: �豸
    batch_size: ������С
    shuffle: �Ƿ��������
    """
    file_data = np.load(filename)  # �����ļ�����
    train_x = file_data['test_x']  # ��ȡ������������
    train_target = file_data['test_target']  # ��ȡ����Ŀ������
    train_x = train_x[:, :, 0:1, :]  # ѡ���������ݵ��ض�ͨ��

    mean = file_data['mean'][:, :, 0:1, :]  # ��ȡ��ֵ
    std = file_data['std'][:, :, 0:1, :]  # ��ȡ��׼��

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # ��������������ת��ΪTensor��ת�Ƶ��豸
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # ������Ŀ������ת��ΪTensor��ת�Ƶ��豸
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)  # �����������ݼ�
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)  # �����������ݼ�����

    # ���ز������ݼ�����������Ŀ�����ݡ���ֵ�ͱ�׼��
    return train_loader, train_target_tensor, mean, std

# �����µ�npz�ļ�
def load_data_train(filename, DEVICE, batch_size, shuffle=True):
    # ����npz�ļ�
    file_data = np.load(filename)

    # ����ѵ�����ݺ�Ŀ��
    train_x = file_data['train_x']  # ѵ��������
    train_x = train_x[:, :, 0:1, :]  # (B, N, F, T)
    train_target = file_data['train_target']  # ѵ����Ŀ��

    # ������֤���ݺ�Ŀ��
    val_x = file_data['val_x']  # ��֤������
    val_x = val_x[:, :, 0:1, :]  # (B, N, F, T)
    val_target = file_data['val_target']  # ��֤��Ŀ��

    # ���ز������ݺ�Ŀ��
    test_x = file_data['test_x']  # ���Լ�����
    test_x = test_x[:, :, 0:1, :]  # (B, N, F, T)
    test_target = file_data['test_target']  # ���Լ�Ŀ��

    # ��ȡ��ֵ�ͱ�׼��
    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # -------- ����ѵ������ǩ --------
    train_label = train_target[:, :, -1]  # ��ȡ���һ����Ϊ��ǩ
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor, train_label_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # -------- ������֤����ǩ --------
    val_label = val_target[:, :, -1]  # ��ȡ���һ����Ϊ��ǩ
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)
    val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor).to(DEVICE)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor, val_label_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------- ������Լ���ǩ --------
    test_label = test_target[:, :, -1]  # ��ȡ���һ����Ϊ��ǩ
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor).to(DEVICE)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor, test_label_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ��ӡ����ά����Ϣ
    print('train:', train_x_tensor.size(), train_target_tensor.size(), train_label_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size(), val_label_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size(), test_label_tensor.size())

    logging.info(f'train: {train_x_tensor.size()}, {train_target_tensor.size()}, {train_label_tensor.size()}')
    logging.info(f'val: {val_x_tensor.size()}, {val_target_tensor.size()}, {val_label_tensor.size()}')
    logging.info(f'test: {test_x_tensor.size()}, {test_target_tensor.size()}, {test_label_tensor.size()}')

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std





def moving_average(x, w):
    """
    �����ƶ�ƽ��ֵ

    x: �������� (array-like)
    w: ���ڴ�С (int)
    """
    # ʹ�þ�������ƶ�ƽ��ֵ
    return np.convolve(x, np.ones(w), 'valid') / w  # �����ƶ�ƽ��ֵ

def point_adjust_eval(anomaly_start, anomaly_end, down, loss, thr1, thr2):
    """
    ����������

    anomaly_start: �쳣��ʼ��
    anomaly_end: �쳣������
    down: �²�����
    loss: ��ʧ
    thr1: ��ֵ1
    thr2: ��ֵ2
    """
    len_ = int(anomaly_end[-1]/down)+1  # ���㳤��
    anomaly = np.zeros((len_))  # ��ʼ���쳣����
    loss = loss[:len_]  # �ü���ʧ����
    anomaly[np.where(loss>thr1)] = 1  # ��Ǵ�����ֵ1�ĵ�
    anomaly[np.where(loss<thr2)] = 1  # ���С����ֵ2�ĵ�
    ground_truth = np.zeros((len_))  # ��ʼ����ʵֵ����
    for i in range(len(anomaly_start)):
        # �쳣��ʼ��"anomaly_start[i]" �� �쳣������"anomaly_end[i]" �²������������Ϊ1,��ʾ���������������쳣����
        # anomaly_start[i]/down �� anomaly_end[i]/down+1 ��ԭʼ��ʱ������������С���²����������
        ground_truth[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)+1] = 1  # ����쳣����
        # �������������һ���㱻���Ϊ�쳣���������򶼱���Ϊ�쳣
        if np.sum(anomaly[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)])>0:
            anomaly[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)] = 1  # �����쳣����
        anomaly[int(anomaly_start[i]/down)] = ground_truth[int(anomaly_start[i]/down)]  # ������ʼ��
        anomaly[int(anomaly_end[i]/down)] = ground_truth[int(anomaly_end[i]/down)]  # ����������

    # anomaly��ʾԤ����쳣���;  ground_truth��ʾʵ�ʵ��쳣���
    return anomaly, ground_truth  # ���ص�������쳣����ʵֵ

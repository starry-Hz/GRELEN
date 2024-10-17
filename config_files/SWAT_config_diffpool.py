import numpy as np

# ����������
class Config(object):
    def __init__(self):
        # ��������
        self.downsampling_fre = 60  # �����²���Ƶ��
        self.target_len = 30  # Ŀ�곤��

        # ģ������
        self.device = 'cuda:0'  # �豸
        self.batch_size = 128  # �������С,����ģ����ѵ��������һ�δ������������С
        self.learning_rate = 0.001  # ѧϰ��
        self.epochs = 200  # ѵ��������
        # self.train_filename = 'data/SWAT/train_swat.npz'  # ѵ�������ļ���
        # self.test_filename = 'data/SWAT/test_swat.npz'  # ���������ļ���
        self.train_filename = 'data/SWAT/train_swat_gcn.npz'  # ѵ�������ļ���
        self.test_filename = 'data/SWAT/test_swat_gcn.npz'  # ���������ļ���

        # ����ģ���������
        self.B, self.N, self.T, self.target_T = 32, 51, 30, 29  # B:���δ�С, N:�ڵ���, T:�������г���, target_T:Ŀ�����г���
        self.n_in = 12  # ����������
        self.n_hid = 64  # ���ز�ά��
        self.do_prob = 0.  # dropout����

        # ͼѧϰ���������
        self.Graph_learner_n_hid = 64  # ͼѧϰ�����ز�ά��
        self.Graph_learner_n_head_dim = 32  # ͼѧϰ��ͷ��ά��
        self.Graph_learner_head = 4  # ͼѧϰ��ͷ��
        self.prior = np.array([0.91, 0.03, 0.03, 0.03])  # �������
        self.temperature = 0.5  # �¶Ȳ���
        self.GRU_n_dim = 64  # GRUά��
        self.max_diffusion_step = 2  # �����ɢ����
        self.num_rnn_layers = 1  # RNN����

        # �����������
        self.save_dir = 'experiments/swat_test'  # ����Ŀ¼
        self.start_epoch = 0  # ��ʼѵ������
        self.param_file = "experiments/swat_test/epoch_147.params"  # ģ�Ͳ����ļ�
        self.save_result = True  # �Ƿ񱣴���
        self.moving_window_ = 2  # �ƶ����ڴ�С
        self.anomaly_file = 'data/SWAT/SWAT_Time.csv'  # �쳣�����ļ�

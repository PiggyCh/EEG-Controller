class arguments():
    def __init__(self):
        self.in_channel = 11
        self.out_dim = 4
        self.path = 'new_data1/XJQ_data1.mat' #'RZH_data.mat' #XJQ_data #DRZ_data DRZ_data.mat
        self.num_label_class = 60
        self.test_ratio = 0.3
        self.lr = 1e-3
        self.device = 'cpu' #cpu
        self.tcp_address = ('192.168.1.103', 5050)
        self.buffer_size = 1000000
        self.path_model = '16h_52m_acc_86.pth'
import torch
from load_data import load_data
from arguments import arguments
import numpy as np
import socket
import json
def trans_tensor(x,dtype):
    return torch.tensor(x,dtype=dtype)
def load_model(path):
    model = torch.load(path,map_location='cpu')
    return model
def get_data_from_source(arg):
    EEG_data, label = load_data(arg.path, arg.num_label_class)
    test_data, test_label = trans_tensor(EEG_data, torch.float32).to(arg.device), trans_tensor(label, torch.long).to(arg.device)
    index = np.array([i for i in range(0, 3600, 4)])
    test_label = test_label.squeeze(1)
    test_data = torch.cat([test_data[:, :, index + i] for i in range(4)], dim=0)
    test_label = torch.cat([test_label for _ in range(4)], dim=0)
    # shuffle Train set
    test_index = np.arange(test_label.shape[0])
    np.random.shuffle(test_index)
    test_data = test_data[test_index, :, :]
    test_label = test_label[test_index]
    return test_data,test_label
def Get_data_from_socket(tcp_adrress,buffer_size):
    sk = socket.socket()
    sk.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,buffer_size)
    sk.bind(tcp_adrress)
    sk.listen()
    conn, addr = sk.accept()
    rece_data = []
    while True:
        ret = conn.recv(buffer_size)
        rece_data.append(ret.decode('utf-8'))
        conn.send(bytes('receving...', encoding = 'utf-8'))
        try:
            print(str(rece_data[-1][-1]))
        except:
            break
        if rece_data[-1][-1] == ']':
            break
    concatenate_s = ""
    for item in rece_data:
        concatenate_s += item
    data = json.loads(concatenate_s)
    print('Data receved')
    conn.close()
    sk.close()
    return data
def test(predictor,arg):
    #test_data = Get_data_from_socket()
    test_data,test_label = get_data_from_source(arg)
    prediction = predictor(test_data)
    test_pre = torch.argmax(prediction[:, ], dim=-1)
    print(str(test_pre.item()))
if __name__ == '__main__':
    path = 'saved_model/9h_36m_acc_84.pth'
    CNN_Model = load_model(path)
    arg =arguments()
    test(CNN_Model,arg)
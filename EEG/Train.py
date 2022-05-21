import time

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from arguments import arguments
from NNmodel import CNN_model,twoDCNN_model,CNN_BN_Nax_model
import numpy as np
from load_data import load_data
def trans_tensor(x,dtype):
    return torch.tensor(x,dtype=dtype)
seed = 8
np.random.seed(seed)
def train(arg):
    #CNN_predictor = CNN_model(arg)
    #CNN_predictor = CNN_BN_Nax_model(arg).to(arg.device)
    CNN_predictor = twoDCNN_model(arg).to(arg.device)
    EEG_data,label= load_data(arg.path,arg.num_label_class)

    index = np.arange(EEG_data.shape[0])
    np.random.shuffle(index)
    EEG_data = EEG_data[index,:,:]
    label = label[index]

    ratio_index = int(arg.test_ratio * EEG_data.shape[0])
    test_data,test_label = trans_tensor(EEG_data[:ratio_index],torch.float32).to(arg.device),trans_tensor(label[:ratio_index],torch.long).to(arg.device)
    train_data, train_label = trans_tensor(EEG_data[ratio_index + 1:], torch.float32).to(arg.device), trans_tensor(label[ratio_index+1:],torch.long).to(arg.device)
    index = np.array([i for i in range(0,3600,4)])
    train_label = train_label.squeeze(1)
    test_label = test_label.squeeze(1)

    train_data = torch.cat([train_data[:, :, index + i] for i in range(4)], dim=0)
    test_data = torch.cat([test_data[:, :, index + i] for i in range(4)], dim=0)
    train_label = torch.cat([train_label for _ in range(4)], dim=0)
    test_label = torch.cat([test_label for _ in range(4)], dim=0)
    #shuffle Train set
    train_index = np.arange(train_data.shape[0])
    np.random.shuffle(train_index)
    train_data = train_data[train_index,:,:]
    train_label = train_label[train_index]
    #shuffle test set
    test_index = np.arange(test_label.shape[0])
    np.random.shuffle(test_index)
    test_data = test_data[test_index,:,:]
    test_label = test_label[test_index]
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(CNN_predictor.parameters(),lr=arg.lr,weight_decay=0.01)
    best_performance = 0
    # down sample#
    for i in range(10000):
        CNN_predictor.train()
        prediction = CNN_predictor(train_data)
        loss = loss_func(prediction, train_label).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train accuracy
        prediction_value = torch.argmax(prediction[:, ], dim=-1)
        train_total_correct = (prediction_value == train_label).sum().float()
        train_accuracy = train_total_correct/train_label.shape[0]
        # test accuracy
        CNN_predictor.eval()
        test_pre = CNN_predictor(test_data)
        test_pre = torch.argmax(test_pre[:,],dim=-1)
        total_correct = (test_pre == test_label).sum().float()
        accuracy = total_correct/test_pre.shape[0]
        if i%10==0:
            if accuracy > best_performance :
                torch.save(CNN_predictor,'saved_model/'+str(time.localtime().tm_hour)+'h_'+str(time.localtime().tm_min)+'m_'+'acc_'+str(int(accuracy.item()*100))+'.pth')
                best_performance = accuracy
            print('epoch: ' + str(i)+
                  ' loss: ' + str(loss.item())[:6]+
                  ' training accuracy: ' + str(train_accuracy.item())[:4]+
                  ' test_accuracy:' + str(accuracy.item())[:6] +
                  ' Best_test: ' + str(best_performance.item())[:5])

    # random.randC
if __name__ == '__main__':
    arg = arguments()
    train(arg)
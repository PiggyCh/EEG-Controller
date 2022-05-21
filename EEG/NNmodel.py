import numpy as np
import torch
import torch.nn as nn
from arguments import arguments as Arg
import torch.functional as F
class twoDCNN_model(nn.Module):
    def __init__(self,arg):
        super(twoDCNN_model, self).__init__()
        self.in_channel = arg.in_channel
        self.out_dim = arg.out_dim
        self.Cov1 = nn.Conv2d(in_channels=1,out_channels= 8, stride=(1,3),kernel_size=(3,12),padding=1) #2,30
        self.BN1 = nn.BatchNorm2d(8)
        self.Cov2 = nn.Conv2d(in_channels=8,out_channels= 2 ,stride=(1,3),kernel_size=(3,12),padding=1) #2,30 kernel
        self.BN2 = nn.BatchNorm2d(2)
        self.Pooling = nn.MaxPool2d(kernel_size=(1,20), stride=(1,10)) #1,30 kernel stride =10
        self.FC1 = nn.Linear(in_features=224,out_features=64) #
        self.FC2 = nn.Linear(in_features=64, out_features=arg.out_dim)
        self.Relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.softmax = nn.Softmax()
    def forward(self,data):
        data = data.unsqueeze(1)
        data = self.Relu(self.BN1(self.Cov1(data)))
        data = self.Relu(self.BN2(self.Cov2(data)))
        data = self.Pooling(data)
        data = data.view(data.shape[0], -1)
        data = self.Relu(self.FC1(data))
        data = self.Relu(self.FC2(data))
        data = self.drop(data)
        data = self.softmax(data)
        return data
        #convert ()
class CNN_BN_Nax_model(nn.Module):
    def __init__(self,arg):
        super(CNN_BN_Nax_model, self).__init__()
        self.in_channel = arg.in_channel
        self.out_dim = arg.out_dim
        self.Cov1 = nn.Conv2d(in_channels=1,out_channels= 8, stride=(1,3),kernel_size=(3,12),padding=2) #2,30
        self.BN1 = nn.BatchNorm2d(8)
        self.Pooling2 = nn.MaxPool2d(kernel_size=(3,10), stride=(1, 3))  # 1,30 kernel stride =10
        self.Cov2 = nn.Conv2d(in_channels=8,out_channels= 2 ,stride=(1,3),kernel_size=(3,12),padding=2) #2,30 kernel
        self.BN2 = nn.BatchNorm2d(2)
        self.Pooling1 = nn.MaxPool2d(kernel_size=(3,10), stride=(1,3)) #1,30 kernel stride =10
        self.FC1 = nn.Linear(in_features=132,out_features=64) #
        self.FC2 = nn.Linear(in_features=64, out_features=arg.out_dim)
        self.Relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.softmax = nn.Softmax()
    def forward(self,data):
        data = data.unsqueeze(1)
        data = self.Relu(self.Cov1(data))
        data = self.BN1(data)
        data = self.Pooling1(data)
        data = self.Relu(self.Cov2(data))
        data = self.BN2(data)
        data = self.Pooling2(data)
        data = data.view(data.shape[0], -1)
        data = self.Relu(self.FC1(data))
        data = self.Relu(self.FC2(data))
        data = self.drop(data)
        data = self.softmax(data)
        return data
class CNN_model(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.in_channel = arg.in_channel
        self.out_dim = arg.out_dim
        self.Cov1 = nn.Conv1d(in_channels=self.in_channel,out_channels= 16, stride=2,kernel_size=(5,))
        self.Cov2 = nn.Conv1d(in_channels=16,out_channels=11 ,stride=2,kernel_size=(5,))
        self.Pooling = nn.AvgPool1d(kernel_size=15, stride=10)
        self.FC1 = nn.Linear(in_features=198,out_features=64)
        self.FC2 = nn.Linear(in_features=64, out_features=arg.out_dim)
        self.Relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
    def forward(self,data):
        data = self.drop(data)
        data = self.Relu(self.Cov1(data))
        data = self.Relu(self.Cov2(data))
        data = self.Pooling(data)
        data = data.view(data.shape[0],-1)
        data = self.Relu(self.FC1(data))
        data = self.Relu(self.FC2(data))
        data = self.softmax(data)
        return data

if __name__ == '__main__':
    arg = Arg()
    model = CNN_model(arg)
    data = torch.tensor(np.zeros([10,5,1024]),dtype=torch.float32)
    model(data)






#
# import gc
# import nndata
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.layers.wrappers import TimeDistributed
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import AveragePooling2D
# from keras.layers.recurrent import LSTM
# from keras import regularizers
# from keras.callbacks import ModelCheckpoint
#
# def create_raw_model(nchan, nclasses, trial_length=960, l1=0):
#     """
#     CNN model definition
#     """
#     input_shape = (trial_length, nchan, 1)
#     model = Sequential()
#     model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
#     model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
#     model.add(AveragePooling2D((30, 1), strides=(15, 1)))
#     model.add(Flatten())
#     model.add(Dense(80, activation="relu"))
#     model.add(Dense(nclasses, activation="softmax"))
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
#     return model
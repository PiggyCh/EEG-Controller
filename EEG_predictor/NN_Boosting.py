from NNmodel12 import twoDCNN_model
import torch
class CNN_Boosting():
    def __init__(self,arg):
        self.CNN_set = [twoDCNN_model(arg) for _ in range(10)]
    def get_result(self,input):
        res = torch.mode(torch.cat([self.CNN_set[i](input) for i in range(len(self.CNN_set))]))
        return res
    def evaluate(self):

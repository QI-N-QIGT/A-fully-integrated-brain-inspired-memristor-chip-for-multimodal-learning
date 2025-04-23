import sys
import torch
from torch import nn
from e100_irtool.cmd.data import load_pickle, save_pickle

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=10, bidirectional=False)

    def forward(self, x, h_0, c_0):
        y = self.lstm(x, (h_0, c_0))
        return y

def main():
    layer_count = 1
    net = Net()
    x = torch.randn(1, 3, 10)
    h_0 = torch.zeros(size=(1, 3, 10))
    c_0 = torch.zeros(size=(1, 3, 10))
    torch.onnx.export(net, (x, h_0, c_0), 'LSTM.onnx', input_names=['input'], output_names=['output'])
if __name__ == '__main__':
    main()
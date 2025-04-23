import sys
import torch
from torch import nn
from e100_irtool.cmd.data import load_pickle, save_pickle


class Loop(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.pool4 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        y = x1 + x2
        y = self.pool4(y)
        return y


class Net(nn.Module):

    def __init__(self, repeat):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 1, bias=False)
        self.loop1 = Loop()
        self.fc3 = nn.Linear(64, 10)
        self.repeat = repeat

    def forward(self, x):
        x = self.conv0(x)
        for i in range(self.repeat):
            x = self.loop1(x)
        x = x.view(-1, 64)
        x = self.fc3(x)
        return x


def main():
    try:
        finp, fwts, foup = sys.argv[1:]
    except:
        sys.exit(f'Usage: {sys.argv[0]} <input> <weights> <output>')

    inp = load_pickle(finp)
    wts = load_pickle(fwts)

    for k in ('conv1', 'conv2'):
        wts[f'loop1.{k}.weight'] = wts.pop(f'loop1-{k}-0.weight')
        wts[f'loop1.{k}.bias'] = wts.pop(f'loop1-{k}-0.bias')

    net = Net(5)
    net.load_state_dict(wts)

    oup = net(inp)
    save_pickle(oup.detach(), foup)


if __name__ == '__main__':
    main()

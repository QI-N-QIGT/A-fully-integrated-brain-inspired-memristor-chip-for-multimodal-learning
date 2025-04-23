#!/usr/bin/env python
import sys
import torch
from torch import nn
from e100_irtool.cmd.data import load_pickle, save_pickle


class Loop(nn.Module):

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(8, 32, 3, padding=1)

    def forward(self, x, y):
        return self.c(x) + y


class Net(nn.Module):

    def __init__(self, repeat):
        super().__init__()
        self.cv = nn.Conv2d(3, 32, 1, bias=False)
        self.lp = Loop()
        self.repeat = repeat

    def forward(self, x):
        x = self.cv(x)
        x = torch.split(x, x.shape[1] // self.repeat, dim=1)
        y = 0
        for i in range(self.repeat):
            y = self.lp(x[i], y)
        return y


def main():
    try:
        finp, fwts, foup = sys.argv[1:]
    except:
        sys.exit(f'Usage: {sys.argv[0]} <input> <weights> <output>')

    inp = load_pickle(finp)
    wts = load_pickle(fwts)
    wts = {
        'cv.weight': wts['cv.weight'],
        'lp.c.weight': wts['lp-c-0.weight'],
        'lp.c.bias': wts['lp-c-0.bias']
    }

    net = Net(4)
    net.load_state_dict(wts)

    oup = net(inp)
    save_pickle(oup.detach(), foup)


if __name__ == '__main__':
    main()

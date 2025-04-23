import sys
import torch
from torch import nn
from e100_irtool.cmd.data import load_pickle, save_pickle

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64, 10, bias=False)

    def forward(self, inp, dump_all=False):
        conv = self.conv(inp)
        relu = self.relu(conv)
        pool = self.pool(relu)
        flat = self.flat(pool)
        fc = self.fc(flat)
        if dump_all:
            return dict(conv=conv, relu=relu, pool=pool, flat=flat, fc=fc)
        else:
            return fc

def main():
    try:
        (in_f, wt_f, out_f) = sys.argv[1:4]
        dump_all = sys.argv[4:5] == ['-a']
    except:
        sys.exit(f'Usage: {sys.argv[0]} <input> <weight> <output> [-a]')
    inp = load_pickle(in_f)
    wts = load_pickle(wt_f)
    net = Net()
    net.load_state_dict(wts)
    out = net(inp, dump_all=dump_all)
    save_pickle(out, out_f)
if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .quant_util import *
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import time
from .quant_util import *
from .base_operator import twn_n, twn_n_nolimit

class conv2d_quant_noise_backwonoise(nn.Conv2d):

    def __init__(self, m: nn.Conv2d, w_quantizer=None, a_quantizer=None, a_out_quantizer=None, int_flag=False):
        assert type(m) == nn.Conv2d
        super(conv2d_quant_noise_backwonoise, self).__init__(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=True if m.bias is not None else False, padding_mode=m.padding_mode)
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        self.a_out_quantizer.int_flag = int_flag
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())
        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)

    def get_int_weight(self):
        (weight_int, scale) = self.w_quantizer.get_int(self.weight)
        return (weight_int, scale)

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]
        weight_bak = self.weight.data.clone()
        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)
        self.weight.data = weight_q.detach()
        x = self._conv_forward(input_q, self.weight, self.bias)
        self.weight.data = weight_bak
        return self.a_out_quantizer(x)

class linear_quant_noise_backwonoise(nn.Linear):

    def __init__(self, m: nn.Linear, w_quantizer=None, a_quantizer=None, a_out_quantizer=None, int_flag=False):
        assert type(m) == nn.Linear
        super(linear_quant_noise_backwonoise, self).__init__(m.in_features, m.out_features, bias=True if m.bias is not None else False)
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(m.weight.detach())
        self.a_out_quantizer.int_flag = int_flag
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())
        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight)

    def get_int_weight(self):
        (weight_int, scale) = self.w_quantizer.get_int(self.weight)
        return (weight_int, scale)

    def forward(self, input):
        if isinstance(input, tuple):
            if input[1] != 0.0:
                input = input[0] * input[1]
            else:
                input = input[0]
        weight_bak = self.weight.data.clone()
        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)
        self.weight.data = weight_q.data.detach()
        x = F.linear(input_q, self.weight, self.bias)
        self.weight.data = weight_bak
        return self.a_out_quantizer(x)
QuanModuleMappingBackWoNoise = {nn.Conv2d: conv2d_quant_noise_backwonoise, nn.Linear: linear_quant_noise_backwonoise}
QuanModule = [conv2d_quant_noise_backwonoise, linear_quant_noise_backwonoise]
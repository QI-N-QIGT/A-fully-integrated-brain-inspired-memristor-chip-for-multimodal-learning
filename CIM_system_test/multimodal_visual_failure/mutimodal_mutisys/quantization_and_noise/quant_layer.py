import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import time
from .quant_util import *
from .base_operator import twn_n, twn_n_nolimit

class conv2d_quant_noise(nn.Conv2d):

    def __init__(self, m: nn.Conv2d, w_quantizer, a_quantizer, a_out_quantizer, int_flag=False):
        assert type(m) == nn.Conv2d
        super(conv2d_quant_noise, self).__init__(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=True if m.bias is not None else False, padding_mode=m.padding_mode)
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
                input = input[0] * input[1].cpu()
            else:
                input = input[0]
        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)
        x = self._conv_forward(input_q, weight_q, self.bias)
        return self.a_out_quantizer(x)

class linear_quant_noise(nn.Linear):

    def __init__(self, m: nn.Linear, w_quantizer, a_quantizer, a_out_quantizer, int_flag=False):
        assert type(m) == nn.Linear
        super(linear_quant_noise, self).__init__(m.in_features, m.out_features, bias=True if m.bias is not None else False)
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
        weight_q = self.w_quantizer(self.weight)
        input_q = self.a_quantizer(input)
        x = F.linear(input_q, weight_q, self.bias)
        return self.a_out_quantizer(x)

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class add_quant(nn.Module):

    def __init__(self, bit=9, all_positive=False, symmetric=True, quant_method=0, shift=0):
        super(add_quant, self).__init__()
        self.bit = bit
        self.a_out_quantizer = LSQ_act_quantizer(bit=bit, all_positive=all_positive, symmetric=symmetric, init_mode='percent', init_percent=0.999)
        self.quant_method = quant_method
        self.shift = shift
        self.int_flag = True
        self.a_out_quantizer.int_flag = True

    def forward(self, input1, input2):
        if not (isinstance(input1, tuple) and isinstance(input2, tuple)):
            raise ValueError('add_quant module need quantized input. ')
        if input1[1] == 0.0 or input2[1] == 0.0:
            raise ValueError('add_quant module need quantized input. ')
        (a1_int, s1) = input1
        (a2_int, s2) = input2
        if self.quant_method == 0:
            if self.shift == 0:
                if s1 > s2:
                    s = s1
                else:
                    s = s2
                x = torch.clamp(a1_int + a2_int, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return (x, s)
                else:
                    return x * s
            elif self.shift == 1:
                if s2 > s1:
                    (a1_int, a2_int) = (a2_int, a1_int)
                    (s1, s2) = (s2, s1)
                k = twn_n(s1 / s2)
                a2_int = floor_pass(a2_int / 2 ** k)
                int_x = a1_int + a2_int
                x = torch.clamp(int_x, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return (x, s1)
                else:
                    return x * s1
        if self.quant_method == 1:
            if self.shift == 0:
                return self.a_out_quantizer(a1_int * s1 + a2_int * s2)
            elif self.shift == 1:
                if s2 > s1:
                    (a1_int, a2_int) = (a2_int, a1_int)
                    (s1, s2) = (s2, s1)
                k0 = twn_n(s1 / s2)
                a2_int = floor_pass(a2_int / 2 ** k0)
                int_x = a1_int + a2_int
                k1 = 0
                if self.a_out_quantizer.get_scale() / s1 > 1.0:
                    k1 = twn_n(self.a_out_quantizer.get_scale() / s1)
                int_x = floor_pass(int_x / 2 ** k1)
                x = torch.clamp(int_x, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return (x, self.a_out_quantizer.get_scale())
                else:
                    return x * self.a_out_quantizer.get_scale()
            elif self.shift == 2:
                k1 = 0
                k2 = 0
                if self.a_out_quantizer.get_scale() / s1 > 1.0:
                    k1 = twn_n(self.a_out_quantizer.get_scale() / s1)
                if self.a_out_quantizer.get_scale() / s2 > 1.0:
                    k2 = twn_n(self.a_out_quantizer.get_scale() / s2)
                a1_int = floor_pass(a1_int / 2 ** k1)
                a2_int = floor_pass(a2_int / 2 ** k2)
                int_x = a1_int + a2_int
                x = torch.clamp(int_x, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
                if self.int_flag:
                    return (x, self.a_out_quantizer.get_scale())
                else:
                    return x * self.a_out_quantizer.get_scale()

class AdaptiveAvgPool2d_quant(nn.AdaptiveAvgPool2d):

    def __init__(self, m: nn.AdaptiveAvgPool2d, quant_flag=False, bit=9, all_positive=False, symmetric=True):
        assert type(m) == nn.AdaptiveAvgPool2d
        assert m.output_size == (1, 1)
        super(AdaptiveAvgPool2d_quant, self).__init__(m.output_size)
        self.quant_flag = quant_flag
        self.bit = bit
        if all_positive:
            assert not symmetric, 'Positive quantization cannot be symmetric'
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        elif symmetric:
            self.thd_neg = -2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
        else:
            self.thd_neg = -2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1

    def forward(self, input):
        if self.quant_flag:
            if not isinstance(input, tuple):
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            if input[1] == 0.0:
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            (input_int, input_scale) = input
            input_size = input_int.size()
            self.size_2D = input_size[2] * input_size[3]
            output = F.adaptive_avg_pool2d(input_int, self.output_size) * self.size_2D
            k = twn_n_nolimit(self.size_2D)
            output = floor_pass(output / 2 ** k)
            output = torch.clamp(output, min=self.thd_neg, max=self.thd_pos)
            return (output, input_scale)
        else:
            if isinstance(input, tuple):
                if input[1] != 0.0:
                    input = input[0] * input[1].cpu()
                else:
                    input = input[0]
            return F.adaptive_avg_pool2d(input, self.output_size)

class AvgPool2d_quant(nn.AvgPool2d):

    def __init__(self, m: nn.AvgPool2d, quant_flag=False, bit=9, all_positive=False, symmetric=True):
        assert type(m) == nn.AvgPool2d
        super(AvgPool2d_quant, self).__init__(m.kernel_size, m.stride, m.padding, m.ceil_mode, m.count_include_pad, m.divisor_override)
        self.quant_flag = quant_flag
        if isinstance(self.kernel_size, tuple):
            self.size_2D = self.kernel_size[0] * self.kernel_size[1]
        else:
            self.size_2D = self.kernel_size ** 2
        self.bit = bit
        if all_positive:
            assert not symmetric, 'Positive quantization cannot be symmetric'
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        elif symmetric:
            self.thd_neg = -2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
        else:
            self.thd_neg = -2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1

    def forward(self, input):
        if self.quant_flag:
            if not isinstance(input, tuple):
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            if input[1] == 0.0:
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            (input_int, input_scale) = input
            output_int = F.avg_pool2d(input_int, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override) * self.size_2D
            k = twn_n_nolimit(self.size_2D)
            output_int = floor_pass(output_int / 2 ** k)
            output_int = torch.clamp(output_int, min=self.thd_neg, max=self.thd_pos)
            return (output_int, input_scale)
        else:
            if isinstance(input, tuple):
                if input[1] != 0.0:
                    input = input[0] * input[1]
                else:
                    input = input[0]
            return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

class BatchNorm2d_quant(nn.BatchNorm2d):

    def __init__(self, m: nn.BatchNorm2d, quant_flag=False, quant_method=1, out_bit=0, w_quantizer=None, bias_quantizer=None, a_out_quantizer=None, *args, **kwargs):
        assert type(m) == nn.BatchNorm2d
        super(BatchNorm2d_quant, self).__init__(m.num_features)
        self.weight = nn.Parameter(m.weight.detach())
        self.bias = nn.Parameter(m.bias.detach())
        self.running_var = m.running_var
        self.running_mean = m.running_mean
        self.track_running_stats = m.track_running_stats
        self.num_batches_tracked = m.num_batches_tracked
        self.w_quantizer = w_quantizer
        self.bias_quantizer = bias_quantizer
        self.a_out_quantizer = a_out_quantizer
        if isinstance(self.w_quantizer, LSQ_weight_quantizer):
            self.w_quantizer.init_scale(m.weight / (m.running_var ** 0.5 + self.eps))
        self.quant_flag = quant_flag
        self.quant_method = quant_method
        if self.quant_method == 2:
            assert out_bit > 1
            self.thd_neg = -2 ** (out_bit - 1) + 1
            self.thd_pos = 2 ** (out_bit - 1) - 1

    def forward(self, input):
        if self.quant_flag:
            if not isinstance(input, tuple):
                raise ValueError('BatchNorm2d_quant with quant_flag=True need quantized input.')
            if input[1] == 0.0:
                raise ValueError('BatchNorm2d_quant with quant_flag=True need quantized input.')
        if isinstance(input, tuple):
            self._check_input_dim(input[0])
        else:
            self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        if self.quant_flag:
            (input_int, input_scale) = input
            input_q = input_int * input_scale
            if self.training:
                tmp = F.batch_norm(input_q, self.running_mean if not self.training or self.track_running_stats else None, self.running_var if not self.training or self.track_running_stats else None, self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
                batch_mean = torch.mean(input_q.permute(1, 0, 2, 3).reshape(self.num_features, -1), dim=1).detach()
                batch_var = torch.var(input_q.permute(1, 0, 2, 3).reshape(self.num_features, -1), dim=1, unbiased=False).detach()
            else:
                batch_mean = self.running_mean
                batch_var = self.running_var
            weight_tmp = self.weight / (batch_var ** 0.5 + self.eps)
            bias_tmp = self.bias - batch_mean * weight_tmp
            (weight_tmp_int, weight_tmp_scale) = self.w_quantizer(weight_tmp)
            if self.quant_method == 1:
                tmp_scale = (input_scale * weight_tmp_scale).detach()
                (bias_tmp_int, bias_tmp_scale) = self.bias_quantizer(bias_tmp, tmp_scale)
                output = input_q * (weight_tmp_int * weight_tmp_scale).reshape(1, self.num_features, 1, 1) + (bias_tmp_int * bias_tmp_scale).reshape(1, self.num_features, 1, 1)
                (out_int, out_scale) = self.a_out_quantizer(output)
                return (out_int, out_scale)
            elif self.quant_method == 2:
                output_tmp = input_q * (weight_tmp_int * weight_tmp_scale).reshape(1, self.num_features, 1, 1)
                (output_int, output_scale) = self.a_out_quantizer(output_tmp)
                (bias_tmp_int, bias_tmp_scale) = self.bias_quantizer(bias_tmp, output_scale.detach())
                output = torch.clamp(round_pass(output_int) + round_pass(bias_tmp_int), self.thd_neg, self.thd_pos)
                return (output, output_scale)
            elif self.quant_method == 3:
                output = input_q * (weight_tmp_int * weight_tmp_scale).reshape(1, self.num_features, 1, 1) + bias_tmp.reshape(1, self.num_features, 1, 1)
                return self.a_out_quantizer(output)
            elif self.quant_method == 4:
                output = input_q * weight_tmp.reshape(1, self.num_features, 1, 1) + bias_tmp.reshape(1, self.num_features, 1, 1)
                return self.a_out_quantizer(output)
        else:
            if isinstance(input, tuple):
                if input[1] != 0.0:
                    input = input[0] * input[1]
                else:
                    input = input[0]
            return F.batch_norm(input, self.running_mean if not self.training or self.track_running_stats else None, self.running_var if not self.training or self.track_running_stats else None, self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

class ReLu_quant(nn.ReLU):

    def __init__(self, m: nn.ReLU):
        assert type(m) == nn.ReLU
        super(ReLu_quant, self).__init__(m.inplace)

    def forward(self, input):
        if isinstance(input, tuple):
            return (F.relu(input[0], inplace=self.inplace), input[1])
        else:
            return F.relu(input, inplace=self.inplace)

class MaxPool2d_quant(nn.MaxPool2d):

    def __init__(self, m: nn.MaxPool2d):
        assert type(m) == nn.MaxPool2d
        super(MaxPool2d_quant, self).__init__(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation)

    def forward(self, input):
        if isinstance(input, tuple):
            return (F.max_pool2d(input[0], self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices), input[1])
        else:
            return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)

class Dropout_quant(nn.Dropout):

    def __init__(self, m: nn.Dropout):
        assert type(m) == nn.Dropout
        super(Dropout_quant, self).__init__(p=m.p, inplace=m.inplace)

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, tuple):
            return (F.dropout(input[0], self.p, self.training, self.inplace), input[1])
        else:
            return F.dropout(input, self.p, self.training, self.inplace)
QuanModuleMapping = {nn.Conv2d: conv2d_quant_noise, nn.Linear: linear_quant_noise}
QuanModule = [conv2d_quant_noise, linear_quant_noise]
ConvMapping = {nn.Conv2d: conv2d_quant_noise}
FcMapping = {nn.Linear: linear_quant_noise}
BnMapping = {nn.BatchNorm2d: BatchNorm2d_quant}
AvgMapping = {nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d_quant, nn.AvgPool2d: AvgPool2d_quant}
OtherMapping = {nn.MaxPool2d: MaxPool2d_quant, nn.ReLU: ReLu_quant, nn.Dropout: Dropout_quant}
totalMappingModule = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d, nn.ReLU, nn.Dropout]
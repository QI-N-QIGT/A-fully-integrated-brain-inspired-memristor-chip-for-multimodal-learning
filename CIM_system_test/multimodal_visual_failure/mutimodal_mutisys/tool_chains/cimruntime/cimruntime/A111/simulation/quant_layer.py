import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import time
from .quant_util import *
from .base_operator import *
import math

class conv2d_quant_noise(nn.Conv2d):

    def __init__(self, *, weight, bias, stride, padding, dilation, group, in_channels, out_channels, kernel_size, offset_vector, Gain_Error, hard_scale_method, number_of_XB, output_clip, XB_lines, w_quantizer=None, a_quantizer=None, a_out_quantizer=None, int_flag=False, A111_process=0, input_method=1, input_shift=0, input_clip_bit=4, in_quant_low_bit=0, using_hard_scale=0, fixed_scale=3, mini_voltage=0.01, G_max=18, ADC_bit=9, ADC_scale=64, input_v=1, shift_num=0.0, GE_wise='Q', GE_mean=1.0, GE_std=0.0, OF_wise='Q', OF_mean=0.0, OF_std=0.0, hard_out_noise_method=0, hard_out_noise=0):
        super(conv2d_quant_noise, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=group, bias=True if bias is not None else False)
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(weight.detach())
        self.a_out_quantizer.int_flag = int_flag
        self.int_flag = int_flag
        self.in_quant_low_bit = in_quant_low_bit
        self.using_hard_scale = using_hard_scale
        self.hard_scale_method = hard_scale_method
        self.fixed_scale = fixed_scale
        self.mini_v = mini_voltage
        self.G_max = G_max
        self.ADC_bit = ADC_bit
        self.ADC_scale = ADC_scale
        self.input_val = input_v
        self.number_of_XB = number_of_XB
        self.XB_lines = XB_lines
        self.offset_vector = offset_vector
        self.Gain_Error = Gain_Error
        self.shift_num = nn.Parameter(torch.tensor(shift_num))
        self.hard_out_noise = hard_out_noise
        self.hard_out_noise_method = hard_out_noise_method
        self.A111_process = A111_process
        self.input_method = input_method
        self.input_shift = input_shift
        self.input_clip_bit = input_clip_bit
        self.GE_wise = GE_wise
        self.GE_mean = GE_mean
        self.GE_std = GE_std
        self.OF_wise = OF_wise
        self.OF_mean = OF_mean
        self.OF_std = OF_std
        self.output_clip = output_clip
        if bias is not None:
            self.bias = nn.Parameter(bias.detach())
        if self.GE_wise != 'Q' or self.OF_wise != 'Q':
            self.a_out_quantizer.int_flag = True if not isinstance(self.a_out_quantizer, NoQuan) else False
        if not isinstance(self.a_out_quantizer, NoQuan):
            self.a_out_quantizer.int_flag = True if not isinstance(self.a_out_quantizer, NoQuan) else False
        if self.in_quant_low_bit:
            self.w_quantizer.int_flag = True if not isinstance(self.w_quantizer, NoQuan) else False
            self.a_quantizer.int_flag = True if not isinstance(self.a_quantizer, NoQuan) else False
            self.a_out_quantizer.int_flag = True if not isinstance(self.a_out_quantizer, NoQuan) else False

    def get_int_weight(self):
        (weight_int, scale) = self.w_quantizer.get_int(self.weight)
        return (weight_int, scale)

    def set_hard_scale(self, mini_voltage=0.01, G_max=18, ADC_bit=9, ADC_scale=16, shift_num=None, input_v=None):
        self.mini_v = mini_voltage
        self.G_max = G_max
        self.ADC_bit = ADC_bit
        self.ADC_scale = ADC_scale
        if shift_num is not None:
            self.shift_num = nn.Parameter(torch.tensor(shift_num))
        if input_v is not None:
            self.input_val = input_v

    def set_offset(self, mean=0, std=0, repeat_times=4):
        temp = mean + std * torch.randn(math.ceil(len(self.offset_vector) / repeat_times))
        for i in range(len(self.offset_vector)):
            self.offset_vector[i] = temp[math.floor(i / repeat_times)]

    def forward(self, input):
        if not isinstance(self.a_quantizer, NoQuan):
            if isinstance(input, tuple):
                input = input[0] * input[1]
            input_q = self.a_quantizer(input)
        else:
            input_q = input
        weight_q = self.w_quantizer(self.weight)
        if not self.A111_process:
            if isinstance(input_q, tuple):
                input_q = input_q[0] * input_q[1]
            if isinstance(weight_q, tuple):
                weight_q = weight_q[0] * weight_q[1]
            x = self._conv_forward(input_q, weight_q, self.bias)
            if not isinstance(self.a_out_quantizer, NoQuan):
                (output, output_S) = self.a_out_quantizer(x)
            else:
                return self.a_out_quantizer(x)
        else:
            assert isinstance(input_q, tuple)
            assert isinstance(weight_q, tuple)
            (input_int, input_scale) = input_q
            (weight_int, weight_scale) = weight_q
            assert self.in_quant_low_bit == 4
            assert self.a_out_quantizer.bit == 9
            if isinstance(self.a_quantizer, NoQuan) and (not isinstance(input, tuple)):
                raise ValueError('input activation is not quantized yet. ')
            if self.A111_process == 2:
                input_int = torch.clamp(round_pass(input_int), 0, 255)
                (input_int, input_scale) = shift_clip_for_4bit_input(input_int, input_scale, self.input_method, self.input_shift, self.input_clip_bit)
                assert self.input_val == 1
            input_int = input_int * self.input_val
            input_scale = input_scale / self.input_val
            if self.using_hard_scale == 1:
                input_scale_hard = self.mini_v
                weight_scale_hard = self.G_max / max(abs(self.w_quantizer.thd_pos), abs(self.w_quantizer.thd_neg))
                output_scale_hard = 2 ** self.ADC_bit / (self.ADC_scale * 2)
                output_scale = self.a_out_quantizer.get_scale()
                output_S = get_hard_output_scale(input_scale, weight_scale, output_scale, self.hard_scale_method, input_scale_hard, weight_scale_hard, output_scale_hard, self.shift_num, True)
                input_int_t = F.unfold(input_int, self.weight.size()[-2:], dilation=self.dilation, padding=self.padding, stride=self.stride)
                weight_int_t = F.unfold(weight_int, self.weight.size()[-2:], dilation=self.dilation, padding=0, stride=self.stride)
                input_int_t.transpose_(1, 2)
                weight_int_t.transpose_(0, 2)
                output_tr = 0
                for i in range(0, self.number_of_XB):
                    xbar_input = input_int_t[:, :, i * self.XB_lines:(i + 1) * self.XB_lines]
                    xbar_weight = weight_int_t[:, i * self.XB_lines:(i + 1) * self.XB_lines, :]
                    xbar_output = xbar_basic_block_hard_method(self.hard_scale_method, xbar_input, input_scale, xbar_weight, weight_scale, output_scale, self.in_quant_low_bit, input_scale_hard, weight_scale_hard, output_scale_hard)
                    output_tr += xbar_output
                input_size_stride = (int(math.sqrt(input_int_t.size()[1])), int(math.sqrt(input_int_t.size()[1])))
                output = F.fold(output_tr.transpose(1, 2), input_size_stride, (1, 1), dilation=self.dilation, padding=0, stride=(1, 1))
                self.shift_num.data = torch.clamp(self.shift_num, 0, 7)
                shift_num = round_pass(self.shift_num)
                output = floor_pass(output / 2 ** shift_num)
            elif self.using_hard_scale == 2:
                input_scale_hard = self.mini_v
                weight_scale_hard = self.G_max / max(abs(self.w_quantizer.thd_pos), abs(self.w_quantizer.thd_neg))
                output_scale_hard = 2 ** self.ADC_bit / (self.ADC_scale * 2)
                input_int_t = F.unfold(input_int, self.weight.size()[-2:], dilation=self.dilation, padding=self.padding, stride=self.stride)
                weight_int_t = F.unfold(weight_int, self.weight.size()[-2:], dilation=self.dilation, padding=0, stride=self.stride)
                input_int_t.transpose_(1, 2)
                weight_int_t.transpose_(0, 2)
                output_tr = 0
                for i in range(0, self.number_of_XB):
                    xbar_input = input_int_t[:, :, i * self.XB_lines:(i + 1) * self.XB_lines]
                    xbar_weight = weight_int_t[:, i * self.XB_lines:(i + 1) * self.XB_lines, :]
                    xbar_output = xbar_basic_block_hard_method(1, xbar_input, 1.0, xbar_weight, 1.0, 1.0, self.in_quant_low_bit, input_scale_hard, weight_scale_hard, output_scale_hard)
                    output_tr += xbar_output
                input_size_stride = (int(math.sqrt(input_int_t.size()[1])), int(math.sqrt(input_int_t.size()[1])))
                output = F.fold(output_tr.transpose(1, 2), input_size_stride, (1, 1), dilation=self.dilation, padding=0, stride=(1, 1))
                self.shift_num.data = torch.clamp(self.shift_num, 0, 7)
                shift_num = round_pass(self.shift_num)
                output = floor_pass(output / 2 ** shift_num)
                output_S = 1.0
            else:
                input_int_t = F.unfold(input_int, self.weight.size()[-2:], dilation=self.dilation, padding=self.padding, stride=self.stride)
                weight_int_t = F.unfold(weight_int, self.weight.size()[-2:], dilation=self.dilation, padding=0, stride=self.stride)
                input_int_t.transpose_(1, 2)
                weight_int_t.transpose_(0, 2)
                output_tr = 0
                for i in range(0, self.number_of_XB):
                    xbar_input = input_int_t[:, :, i * self.XB_lines:(i + 1) * self.XB_lines]
                    xbar_weight = weight_int_t[:, i * self.XB_lines:(i + 1) * self.XB_lines, :]
                    (xbar_output, xbar_output_scale) = xbar_basic_block_v2(xbar_input, input_scale, xbar_weight, weight_scale, self.in_quant_low_bit, self.a_out_quantizer, True)
                    output_tr += xbar_output
                input_size_stride = (int(math.sqrt(input_int_t.size()[1])), int(math.sqrt(input_int_t.size()[1])))
                output = F.fold(output_tr.transpose(1, 2), input_size_stride, (1, 1), dilation=self.dilation, padding=0, stride=(1, 1))
                self.shift_num.data = torch.clamp(self.shift_num, 0, 7)
                shift_num = round_pass(self.shift_num)
                output = floor_pass(output / 2 ** shift_num)
                output_S = xbar_output_scale * 2 ** shift_num
        if self.GE_wise == 'layer':
            ge = self.GE_std * torch.randn(1).clamp_(-3.0, 3.0) + self.GE_mean
            for i in range(len(self.Gain_Error)):
                self.Gain_Error[i] = ge
        elif self.GE_wise == 'channel':
            ge = self.GE_std * torch.randn_like(self.Gain_Error) + self.GE_mean
            for i in range(len(self.Gain_Error)):
                self.Gain_Error[i] = ge[i]
        if self.OF_wise == 'layer':
            of = self.OF_std * torch.randn(1).clamp_(-3.0, 3.0) + self.OF_mean
            for i in range(len(self.offset_vector)):
                self.offset_vector[i] = of
        elif self.OF_wise == 'channel':
            of = self.OF_std * torch.randn_like(self.offset_vector) + self.OF_mean
            for i in range(len(self.offset_vector)):
                self.offset_vector[i] = of[i]
        output = output * self.Gain_Error.unsqueeze(dim=1).unsqueeze(dim=1) + self.offset_vector.unsqueeze(dim=1).unsqueeze(dim=1)
        output = round_pass(output)
        if self.hard_out_noise_method == 1:
            output = add_noise(output, 'add', self.hard_out_noise, 'max_min')
        elif self.hard_out_noise_method == 2:
            output = add_noise_channel_wise(output, 'add', self.hard_out_noise, 'max_min')
        elif self.hard_out_noise_method == 3:
            output = add_noise_element_wise(output, self.hard_out_noise)
        output = round_pass(output)
        if self.output_clip:
            output = torch.clamp(output, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
        if self.int_flag:
            return (output, output_S)
        else:
            return output * output_S

class linear_quant_noise(nn.Linear):

    def __init__(self, *, weight, bias, in_features, out_features, offset_vector, Gain_Error, output_clip, number_of_XB, XB_lines, hard_scale_method, w_quantizer=None, a_quantizer=None, a_out_quantizer=None, int_flag=False, A111_process=0, input_method=1, input_shift=0, input_clip_bit=4, in_quant_low_bit=0, using_hard_scale=0, fixed_scale=3, mini_voltage=0.01, G_max=18, ADC_bit=9, ADC_scale=64, input_v=1, shift_num=0.0, GE_wise='Q', GE_mean=1.0, GE_std=0.0, OF_wise='Q', OF_mean=0.0, OF_std=0.0, hard_out_noise_method=0, hard_out_noise=0):
        super(linear_quant_noise, self).__init__(in_features, out_features, bias=True if bias is not None else False)
        self.w_quantizer = w_quantizer
        self.a_quantizer = a_quantizer
        self.a_out_quantizer = a_out_quantizer
        self.weight = nn.Parameter(weight.detach())
        self.a_out_quantizer.int_flag = int_flag
        self.int_flag = int_flag
        self.in_quant_low_bit = in_quant_low_bit
        self.using_hard_scale = using_hard_scale
        self.hard_scale_method = hard_scale_method
        self.fixed_scale = fixed_scale
        self.mini_v = mini_voltage
        self.G_max = G_max
        self.ADC_bit = ADC_bit
        self.ADC_scale = ADC_scale
        self.input_val = input_v
        self.number_of_XB = number_of_XB
        self.XB_lines = XB_lines
        self.offset_vector = offset_vector
        self.Gain_Error = Gain_Error
        self.shift_num = nn.Parameter(torch.tensor(shift_num))
        self.A111_process = A111_process
        self.input_method = input_method
        self.input_shift = input_shift
        self.input_clip_bit = input_clip_bit
        self.GE_wise = GE_wise
        self.GE_mean = GE_mean
        self.GE_std = GE_std
        self.OF_wise = OF_wise
        self.OF_mean = OF_mean
        self.OF_std = OF_std
        self.hard_out_noise_method = hard_out_noise_method
        self.hard_out_noise = hard_out_noise
        self.output_clip = output_clip
        if bias is not None:
            self.bias = nn.Parameter(bias.detach())
        if self.GE_wise != 'Q' or self.OF_wise != 'Q':
            self.a_out_quantizer.int_flag = True if not isinstance(self.a_out_quantizer, NoQuan) else False
        if not isinstance(self.a_out_quantizer, NoQuan):
            self.a_out_quantizer.int_flag = True if not isinstance(self.a_out_quantizer, NoQuan) else False
        if self.in_quant_low_bit:
            self.w_quantizer.int_flag = True if not isinstance(self.w_quantizer, NoQuan) else False
            self.a_quantizer.int_flag = True if not isinstance(self.a_quantizer, NoQuan) else False
            self.a_out_quantizer.int_flag = True if not isinstance(self.a_out_quantizer, NoQuan) else False

    def get_int_weight(self):
        (weight_int, scale) = self.w_quantizer.get_int(self.weight)
        return (weight_int, scale)

    def set_hard_scale(self, mini_voltage=0.01, G_max=18, ADC_bit=9, ADC_scale=16, shift_num=None, input_v=None):
        self.mini_v = mini_voltage
        self.G_max = G_max
        self.ADC_bit = ADC_bit
        self.ADC_scale = ADC_scale
        if shift_num is not None:
            self.shift_num = nn.Parameter(torch.tensor(shift_num))
        if input_v is not None:
            self.input_val = input_v

    def set_offset(self, mean=0, std=0, repeat_times=4):
        temp = mean + std * torch.randn(math.ceil(len(self.offset_vector) / repeat_times))
        for i in range(len(self.offset_vector)):
            self.offset_vector[i] = temp[math.floor(i / repeat_times)]

    def forward(self, input):
        if not isinstance(self.a_quantizer, NoQuan):
            if isinstance(input, tuple):
                input = input[0] * input[1]
            input_q = self.a_quantizer(input)
        else:
            input_q = input
        weight_q = self.w_quantizer(self.weight)
        if not self.A111_process:
            if isinstance(input_q, tuple):
                input_q = input_q[0] * input_q[1]
            if isinstance(weight_q, tuple):
                weight_q = weight_q[0] * weight_q[1]
            x = F.linear(input_q, weight_q, self.bias)
            if not isinstance(self.a_out_quantizer, NoQuan):
                (output, output_S) = self.a_out_quantizer(x)
            else:
                return self.a_out_quantizer(x)
        else:
            assert isinstance(input_q, tuple)
            assert isinstance(weight_q, tuple)
            (input_int, input_scale) = input_q
            (weight_int, weight_scale) = weight_q
            assert self.in_quant_low_bit == 4
            assert self.a_out_quantizer.bit == 9
            if isinstance(self.a_quantizer, NoQuan) and (not isinstance(input, tuple)):
                raise ValueError('input activation is not quantized yet. ')
            if self.A111_process == 2:
                input_int = torch.clamp(round_pass(input_int), 0, 255)
                (input_int, input_scale) = shift_clip_for_4bit_input(input_int, input_scale, self.input_method, self.input_shift, self.input_clip_bit)
                assert self.input_val == 1
            input_int = input_int * self.input_val
            input_scale = input_scale / self.input_val
            if self.using_hard_scale == 1:
                input_scale_hard = self.mini_v
                weight_scale_hard = self.G_max / max(abs(self.w_quantizer.thd_pos), abs(self.w_quantizer.thd_neg))
                output_scale_hard = 2 ** self.ADC_bit / (self.ADC_scale * 2)
                output_scale = self.a_out_quantizer.get_scale()
                output_S = get_hard_output_scale(input_scale, weight_scale, output_scale, self.hard_scale_method, input_scale_hard, weight_scale_hard, output_scale_hard, self.shift_num, True)
                weight_int_t = weight_int.t()
                output = 0
                for i in range(0, self.number_of_XB):
                    xbar_input = input_int[:, i * self.XB_lines:(i + 1) * self.XB_lines]
                    xbar_weight = weight_int_t[i * self.XB_lines:(i + 1) * self.XB_lines, :]
                    xbar_output = xbar_basic_block_hard_method(self.hard_scale_method, xbar_input, input_scale, xbar_weight, weight_scale, output_scale, self.in_quant_low_bit, input_scale_hard, weight_scale_hard, output_scale_hard)
                    output += xbar_output
                self.shift_num.data = torch.clamp(self.shift_num, 0, 7)
                shift_num = round_pass(self.shift_num)
                output = floor_pass(output / 2 ** shift_num)
            elif self.using_hard_scale == 2:
                input_scale_hard = self.mini_v
                weight_scale_hard = self.G_max / max(abs(self.w_quantizer.thd_pos), abs(self.w_quantizer.thd_neg))
                output_scale_hard = 2 ** self.ADC_bit / (self.ADC_scale * 2)
                weight_int_t = weight_int.t()
                output = 0
                for i in range(0, self.number_of_XB):
                    xbar_input = input_int[:, i * self.XB_lines:(i + 1) * self.XB_lines]
                    xbar_weight = weight_int_t[i * self.XB_lines:(i + 1) * self.XB_lines, :]
                    xbar_output = xbar_basic_block_hard_method(1, xbar_input, 1.0, xbar_weight, 1.0, 1.0, self.in_quant_low_bit, input_scale_hard, weight_scale_hard, output_scale_hard)
                    output += xbar_output
                self.shift_num.data = torch.clamp(self.shift_num, 0, 7)
                shift_num = round_pass(self.shift_num)
                output = floor_pass(output / 2 ** shift_num)
                output_S = 1.0
            else:
                weight_int_t = weight_int.t()
                output = 0
                for i in range(0, self.number_of_XB):
                    xbar_input = input_int[:, i * self.XB_lines:(i + 1) * self.XB_lines]
                    xbar_weight = weight_int_t[i * self.XB_lines:(i + 1) * self.XB_lines, :]
                    (xbar_output, xbar_output_scale) = xbar_basic_block_v2(xbar_input, input_scale, xbar_weight, weight_scale, self.in_quant_low_bit, self.a_out_quantizer, True)
                    output += xbar_output
                self.shift_num.data = torch.clamp(self.shift_num, 0, 7)
                shift_num = round_pass(self.shift_num)
                output = floor_pass(output / 2 ** shift_num)
                output_S = xbar_output_scale * 2 ** shift_num
        if self.GE_wise == 'layer':
            ge = self.GE_std * torch.randn(1).clamp_(-3.0, 3.0) + self.GE_mean
            for i in range(len(self.Gain_Error)):
                self.Gain_Error[i] = ge
        elif self.GE_wise == 'channel':
            ge = self.GE_std * torch.randn_like(self.Gain_Error) + self.GE_mean
            for i in range(len(self.Gain_Error)):
                self.Gain_Error[i] = ge[i]
        if self.OF_wise == 'layer':
            of = self.OF_std * torch.randn(1).clamp_(-3.0, 3.0) + self.OF_mean
            for i in range(len(self.offset_vector)):
                self.offset_vector[i] = of
        elif self.OF_wise == 'channel':
            of = self.OF_std * torch.randn_like(self.offset_vector) + self.OF_mean
            for i in range(len(self.offset_vector)):
                self.offset_vector[i] = of[i]
        output = output * self.Gain_Error + self.offset_vector
        output = round_pass(output)
        if self.hard_out_noise_method == 1:
            output = add_noise(output, 'add', self.hard_out_noise, 'max_min')
        elif self.hard_out_noise_method == 2:
            output = add_noise_channel_wise(output, 'add', self.hard_out_noise, 'max_min')
        elif self.hard_out_noise_method == 3:
            output = add_noise_element_wise(output, self.hard_out_noise)
        output = round_pass(output)
        if self.output_clip:
            if self.number_of_XB == 1 and (not self.input_method in [4, 6]):
                output = torch.clamp(output, -128, 127)
            else:
                output = torch.clamp(output, self.a_out_quantizer.thd_neg, self.a_out_quantizer.thd_pos)
        if self.int_flag:
            return (output, output_S)
        else:
            return output * output_S

class add_quant(nn.Module):

    def __init__(self, *, thd_neg, thd_pos, bit=9, all_positive=False, symmetric=True, quant_method=0, shift=0, clip_before=False):
        super(add_quant, self).__init__()
        self.bit = bit
        self.thd_neg = thd_neg
        self.thd_pos = thd_pos
        self.quant_method = quant_method
        self.shift = shift
        self.int_flag = True
        self.clip_before = clip_before
        self.register_buffer('s1', torch.tensor(1.0))
        self.register_buffer('s2', torch.tensor(1.0))

    def forward(self, input1, input2):
        if not (isinstance(input1, tuple) and isinstance(input2, tuple)):
            return input1 + input2
        if input1[1] == 0.0 or input2[1] == 0.0:
            raise ValueError('add_quant module need quantized input. ')
        (a1_int, s1) = input1
        (a2_int, s2) = input2
        self.s1.data = s1
        self.s2.data = s2
        if self.clip_before:
            a1_int = torch.clamp(a1_int, self.thd_neg, self.thd_pos)
            a2_int = torch.clamp(a2_int, self.thd_neg, self.thd_pos)
        if self.quant_method == 0:
            if self.shift == 0:
                if s1 > s2:
                    s = s1
                else:
                    s = s2
                x = torch.clamp(a1_int + a2_int, self.thd_neg, self.thd_pos)
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
                x = torch.clamp(int_x, self.thd_neg, self.thd_pos)
                if self.int_flag:
                    return (x, s1)
                else:
                    return x * s1

class AdaptiveAvgPool2d_quant(nn.AdaptiveAvgPool2d):

    def __init__(self, *, thd_neg, thd_pos, quant_flag=False, bit=9, all_positive=False, symmetric=True, set_shift_num=False, shift_num=0):
        super(AdaptiveAvgPool2d_quant, self).__init__((1, 1))
        self.quant_flag = quant_flag
        self.bit = bit
        self.thd_neg = thd_neg
        self.thd_pos = thd_pos
        self.set_shift_num = set_shift_num
        self.shift_num = int(shift_num)

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
            if self.set_shift_num:
                k = self.shift_num
            output = floor_pass(output / 2 ** k)
            output = torch.clamp(output, min=self.thd_neg, max=self.thd_pos)
            return (output, input_scale)
        else:
            if isinstance(input, tuple):
                if input[1] != 0.0:
                    input = input[0] * input[1]
                else:
                    input = input[0]
            return F.adaptive_avg_pool2d(input, self.output_size)

class AvgPool2d_quant(nn.AvgPool2d):

    def __init__(self, *, kernel, stride, padding, ceil_mode, quant_flag=False, bit=9, thd_neg=0, thd_pos=0, set_shift_num=False, shift_num=0):
        super(AvgPool2d_quant, self).__init__(kernel, stride, padding, ceil_mode)
        self.quant_flag = quant_flag
        if isinstance(self.kernel_size, tuple):
            self.size_2D = self.kernel_size[0] * self.kernel_size[1]
        else:
            self.size_2D = self.kernel_size ** 2
        self.bit = bit
        self.thd_neg = thd_neg
        self.thd_pos = thd_pos
        self.set_shift_num = set_shift_num
        self.shift_num = shift_num

    def forward(self, input):
        if self.quant_flag:
            if not isinstance(input, tuple):
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            if input[1] == 0.0:
                raise ValueError('AdaptiveAvgPool2d_quant module need quantized input. ')
            (input_int, input_scale) = input
            output_int = F.avg_pool2d(input_int, self.kernel_size, self.stride, self.padding, self.ceil_mode) * self.size_2D
            k = twn_n_nolimit(self.size_2D)
            if self.set_shift_num:
                k = self.shift_num
            output_int = floor_pass(output_int / 2 ** k)
            output_int = torch.clamp(output_int, min=self.thd_neg, max=self.thd_pos)
            return (output_int, input_scale)
        else:
            if isinstance(input, tuple):
                if input[1] != 0.0:
                    input = input[0] * input[1]
                else:
                    input = input[0]
            return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode)

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

    def __init__(self):
        super(ReLu_quant, self).__init__()

    def forward(self, input):
        if isinstance(input, tuple):
            return (F.relu(input[0], inplace=self.inplace), input[1])
        else:
            return F.relu(input, inplace=self.inplace)

class MaxPool2d_quant(nn.MaxPool2d):

    def __init__(self, *, kernel, stride, padding, dilation, ceil_mode):
        super(MaxPool2d_quant, self).__init__(kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)

    def forward(self, input):
        if isinstance(input, tuple):
            return (F.max_pool2d(input[0], self.kernel_size, self.stride, self.padding, self.dilation), input[1])
        else:
            return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, ceil_mode=self.ceil_mode)

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
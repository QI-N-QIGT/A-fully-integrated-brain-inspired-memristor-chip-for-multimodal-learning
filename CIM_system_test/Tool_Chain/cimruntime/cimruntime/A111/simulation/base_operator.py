import torch
from torch.autograd.function import Function
from .quant_util import round_pass, floor_pass


def twn_n(s, max_k=8):
    assert s >= 1, "twn_n func's input s need tobe >= 1. "
    for k in range(0, max_k):
        if (2 ** k) <= s < (2 ** (k + 1)):
            if (s / (2 ** k)) <= ((2 ** (k + 1)) / s):
                return k
            else:
                return k + 1
    return max_k


def twn_n_nolimit(s):
    assert s >= 1, "twn_n func's input s need tobe >= 1. "
    k = -1
    while True:
        k += 1
        if (2 ** k) <= s < (2 ** (k + 1)):
            if (s / (2 ** k)) <= ((2 ** (k + 1)) / s):
                return k
            else:
                return k + 1


def shift_clip_for_4bit_input(input, scale, method=1, shift=0, clip_bit=4):
    if method == 1:  # 取高四位作为低四位
        return torch.clamp(floor_pass(input / 16), 0, 15), scale * 16
    elif method == 2:  # 取低四位
        return (input / 16.).frac() * 16., scale
    elif method == 3:  # 自定义取数位置
        tmp = floor_pass(input / pow(2, shift))
        return (tmp / pow(2, clip_bit)).frac() * pow(2, clip_bit), scale * pow(2, shift)
    elif method == 4: # 取高四位作为高四位
        return torch.clamp(floor_pass(input / 16), 0, 15) * 16, scale
    elif method == 5:  # 取高四位作为低四位
        return torch.clamp(floor_pass(input / 16), 0, 15), 1.0
    elif method == 6:  # 取高四位作为高四位
        return torch.clamp(floor_pass(input / 16), 0, 15) * 16, 1.0
    elif method == 7:  # 自定义取数位置
        tmp = floor_pass(input / pow(2, shift))
        return (tmp / pow(2, clip_bit)).frac() * pow(2, clip_bit), 1.0
    else:
        raise ValueError("Not support method {} for shift_clip_for_4bit_input func yet. ".format(method))


def train_high_bit(input, low_bit, scale):
    class train_high_bit_operater(Function):
        @staticmethod
        def forward(ctx, input, low_bit, scale):
            # input = input.int()
            # input_high = (input & 0xf0) >> 4
            input_int = (input / scale).round()
            input_high = (input_int / (2 ** low_bit)).trunc()
            ctx.save_for_backward(input_high)
            input_high = input_high * scale
            return input_high.float()
        
        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            if torch.sum(input) == 0.0:
                grad_input = None
            else:
                grad_input = grad_output
            return grad_input, None, None
    
    return train_high_bit_operater.apply(input, low_bit, scale)


def train_low_bit(input, low_bit, scale):
    class train_low_bit_operater(Function):
        @staticmethod
        def forward(ctx, input, low_bit, scale):
            input_int = (input / scale).round()
            input_low = (input_int / (2 ** low_bit)).frac() * (2 ** low_bit)
            ctx.save_for_backward(input_low)
            input_low = input_low * scale
            return input_low
        
        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            if torch.sum(input) == 0.0:
                grad_input = None
            else:
                grad_input = grad_output
            return grad_input, None, None
    
    return train_low_bit_operater.apply(input, low_bit, scale)


def xbar_basic_block_v2(input, input_scale, weight, weight_scale, in_quant_low_bit, a_out_quantizer, out_scale_test):
    # 输入拆分为高低4bit
    input_high = train_high_bit(input, in_quant_low_bit, 1)
    input_low = train_low_bit(input, in_quant_low_bit, 1)
    
    # 模拟rram 计算 高低4bit
    output_high = torch.matmul(input_high * input_scale, weight * weight_scale)
    output_low = torch.matmul(input_low * input_scale, weight * weight_scale)
    
    # ADC  A (V) -> D (V)  step1 shift   step2  map
    # ACD操作分为两部分
    # 第一：乘以scale
    output_high_adc1, output_high_adc1_scale = a_out_quantizer(output_high)
    output_low_adc1, output_low_adc1_scale = a_out_quantizer(output_low)
    
    output_high_adc1 = floor_pass(output_high_adc1 / 2)
    output_low_adc1 = floor_pass(output_low_adc1 / 2)
    if out_scale_test:
        output_low_adc1_scale = output_low_adc1_scale * 2
    
    # 高低4bit结果在整合到8bit输出
    output = output_high_adc1 * (2 ** in_quant_low_bit) + output_low_adc1
    
    return output, output_low_adc1_scale


def xbar_basic_block_hard_method(method, input, input_scale, weight, weight_scale, output_scale, in_quant_low_bit,
                                 input_scale_hard, weight_scale_hard, output_scale_hard):
    scale_temp = input_scale_hard * weight_scale_hard * output_scale_hard
    if method == 2:
        tmp = scale_temp * output_scale / (input_scale * weight_scale)
        if tmp > 1:
            k = twn_n(tmp, max_k=7)
        else:
            k = 0
        input = floor_pass(input / (2 ** k))
    
    # 输入拆分为高低4bit
    input_high = train_high_bit(input, in_quant_low_bit, 1)
    input_low = train_low_bit(input, in_quant_low_bit, 1)
    
    mat_value_high = round_pass(torch.matmul(input_high, weight) * scale_temp)
    output_high_adc1 = torch.clamp(mat_value_high, min=-256, max=255)
    mat_value_low = round_pass(torch.matmul(input_low, weight) * scale_temp)
    output_low_adc1 = torch.clamp(mat_value_low, min=-256, max=255)
    
    output_high_adc1 = floor_pass(output_high_adc1 / 2)
    output_low_adc1 = floor_pass(output_low_adc1 / 2)
    
    # 高低4bit结果在整合到8bit输出
    output = output_high_adc1 * (2 ** in_quant_low_bit) + output_low_adc1
    
    return output


def get_hard_output_scale(soft_a_s, soft_w_s, soft_a_out_s, hard_scale_method, input_scale_hard, weight_scale_hard,
                          output_scale_hard, shift_num, out_scale_test):
    input_scale = soft_a_s
    weight_scale = soft_w_s
    output_scale = soft_a_out_s
    # input_scale = input_scale / input_val
    
    scale_temp = input_scale_hard * weight_scale_hard * output_scale_hard
    if hard_scale_method == 0:  # 不依赖量化器的scale
        output_low_adc1_scale = 1.0 / output_scale_hard
    # 根据训练好的输入/权重/输出量化器的scale，计算合适的output scale给下一层
    elif hard_scale_method == 1:
        output_low_adc1_scale = input_scale * weight_scale / scale_temp
    elif hard_scale_method == 2:
        output_low_adc1_scale = output_scale
    elif hard_scale_method == 3:
        output_low_adc1_scale = output_scale
    if out_scale_test:
        xbar_output_scale = output_low_adc1_scale * 2
    
    if not isinstance(shift_num, torch.Tensor):
        shift_num = torch.tensor(shift_num)
    shift_num.data = torch.clamp(shift_num, 0, 7)
    shift_num = shift_num.round()
    if out_scale_test:
        xbar_output_scale = xbar_output_scale * (2 ** shift_num)
    return xbar_output_scale
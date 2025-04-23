import json
from .utils import *
from c200_sdk.sdk_array_newsystem import SDKArray

with open(r'd:\repository\cimruntime\cimruntime\c200\simulation\current_range.json', 'r') as f:
    current_range = json.load(f)

def get_bit_value(tensor, bit_index):
    # 确保输入是int8类型
    tensor = tensor.to(torch.int8)
    
    # 计算符号位 (最左边的一位)
    sign_bit = tensor < 0
    
    # 计算绝对值，并从中提取特定位
    abs_tensor = tensor.abs()
    bit_value = (abs_tensor >> bit_index) & 1

    # 如果符号位是1，并且特定位是1，则返回-1
    # 如果符号位是0，则返回特定位的值
    result = torch.where(sign_bit & (bit_value == 1), torch.tensor(-1, dtype=torch.int8), bit_value)

    return result.float()

def C200_MVM_1bIN_4bOUT_SIM(input_data, weight_data, *, DAC_noise = 0, conductance_noise = 0.0, ADC_noise = 0, ADC_offset = 0,
                            integration_time = 0, max_conductance = 19.16, max_voltage = 0.1, device='cpu',):
    
    # 根据输入和权重，转化为对应的电导值以及电压值
    input_voltage = input_data * max_voltage
    weight_conductance = weight_data / 7 * max_conductance
    
    # 输入加噪
    if DAC_noise != 0:
        input_voltage = input_voltage + DAC_noise * torch.randn_like(input_voltage).to(device)
    
    # 权重加噪
    if conductance_noise != 0:
        # current weight max * noise
        # conductance_noise = torch.abs(weight_conductance).max() * conductance_noise
        conductance_noise = max_conductance * conductance_noise
        # weight conductance + noise
        weight_conductance = weight_conductance + conductance_noise * torch.randn_like(weight_conductance).clamp_(-3.0, 3.0).to(device)
    
    # 计算输出电流
    output_current = F.linear(input_voltage, weight_conductance)
    
    # 输出加噪
    if ADC_noise != 0:
        # current output_current max * noise
        ADC_noise = torch.abs(output_current).max() * ADC_noise
        # output current + noise
        output_current = output_current + ADC_noise * torch.randn_like(output_current).clamp_(-3.0, 3.0).to(device)
    
    # 电流量化
    output_quant = output_current / current_range[f'{integration_time*100}'] * 7
    output_quant = torch.round(output_quant).to(torch.int32)
    # # print(f'C200 analog mac output data type: {output_quant.dtype}')
    
    # 量化偏移误差
    if ADC_offset != 0:
        ADC_offset = torch.round( torch.randn_like(output_current).clamp_(-3.0, 3.0) * ADC_offset).to(device)
        # output_quant = output_quant + torch.randint(-ADC_offset, ADC_offset, size=output_quant.shape).to(device)
        output_quant = output_quant + ADC_offset
    
    # 电流截断
    output_quant = torch.clamp(output_quant, min=-8, max=7)
    
    return output_quant

def C200_Conv_1bIN_4bOUT_SIM(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, max_conductance = 16.6, max_voltage = 0.1, 
                    device='cpu', relu = False,):
    
    # 量化电流的量程与积分时间的对应关系
    max_current = current_range
    
    # 根据输入和权重，转化为对应的电导值以及电压值
    input_voltage = input_data * max_voltage
    weight_conductance = weight_data / 6 * max_conductance
    
    # 前置 relu
    if relu:
        input_voltage = torch.clamp(input_voltage, min=0)
    
    # 输入加噪
    if DAC_noise != 0:
        input_voltage = input_voltage + DAC_noise * torch.randn_like(input_voltage).to(device)
    
    # 权重加噪
    if conductance_noise != 0:
        # current weight max * noise
        conductance_noise = torch.abs(weight_conductance).max() * conductance_noise
        # weight conductance + noise
        weight_conductance = weight_conductance + conductance_noise * torch.randn_like(weight_conductance).clamp_(-3.0, 3.0).to(device)
    
    # 计算输出电流
    output_current = torch.conv2d(input_voltage, weight_conductance, stride=stride, padding=padding)
    
    # 输出加噪
    if ADC_noise != 0:
        # current output_current max * noise
        ADC_noise = torch.abs(output_current).max() * ADC_noise
        # output current + noise
        output_current = output_current + ADC_noise * torch.randn_like(output_current).clamp_(-3.0, 3.0).to(device)
    
    # 电流量化
    output_quant = output_current / max_current[ADC_quant_level] * 7
    output_quant = torch.round(output_quant).to(torch.int32)
    # # print(f'C200 analog mac output data type: {output_quant.dtype}')
    
    # 量化偏移误差
    if ADC_offset != 0:
        ADC_offset = torch.round( torch.randn_like(output_current).clamp_(-3.0, 3.0) * ADC_offset).to(device)
        # output_quant = output_quant + torch.randint(-ADC_offset, ADC_offset, size=output_quant.shape).to(device)
        output_quant = output_quant + ADC_offset
    
    # 电流截断
    output_quant = torch.clamp(output_quant, min=-8, max=7)
    
    return output_quant   

def C200_Conv_4bIN_to_4bOUT_SIM(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, scale=0, 
                    max_conductance = 19.16, max_voltage = 0.1,  device='cpu', relu = False):
    
    if torch.is_tensor(input_data) and len(input_data.shape) == 2:
        input_data = input_data.view(input_data.shape[0], -1, 1, 1)
    
    # 输入拆分
    input_x4 = (input_data / 4).trunc()
    input_x2 = ((input_data / 4).frac() * 2).trunc()
    input_x1 = (input_data / 2).frac() * 2
    
    # 计算三次
    output_x4 =  C200_Conv_1bIN_4bOUT_SIM(input_x4, weight_data, stride=stride, padding=padding, 
                                DAC_noise = DAC_noise, conductance_noise = conductance_noise, 
                                ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                                ADC_quant_level = ADC_quant_level, max_conductance = max_conductance, 
                                max_voltage = max_voltage, device=device, 
                                relu = relu,)

    output_x2 =  C200_Conv_1bIN_4bOUT_SIM(input_x2, weight_data, stride=stride, padding=padding, 
                                DAC_noise = DAC_noise, conductance_noise = conductance_noise, 
                                ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                                ADC_quant_level = ADC_quant_level, max_conductance = max_conductance, 
                                max_voltage = max_voltage, device=device, 
                                relu = relu,)

    output_x1 =  C200_Conv_1bIN_4bOUT_SIM(input_x1, weight_data, stride=stride, padding=padding, 
                                DAC_noise = DAC_noise, conductance_noise = conductance_noise, 
                                ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                                ADC_quant_level = ADC_quant_level, max_conductance = max_conductance, 
                                max_voltage = max_voltage, device=device, 
                                relu = relu,)
    # 移位累加
    output_quant = 4 * output_x4 + 2 * output_x2 + 1 * output_x1
    
    # scale + round ===>量化为 4bit
    output_vector = (output_vector * scale).round().to(torch.int32)
    
    # 输出截断到4bit
    output_quant = torch.clamp(output_quant, min=-7, max=7)
    
    return output_quant

def C200_Conv2d_Chip(chip_id, inputs, addr, *, kernel = 1, stride = 1, padding = 0, input_quant_scale = 1.0, output_dequant_scale = 1.0,
                    activation_bits = 4, input_expansion_method = 0, integration_time = 0, weight_row_copy = 1):
    
    # 1. 输入量化, input quant scale 是 训练过程中学习而来的
    inputs_int = (inputs / input_quant_scale).round()
    thd_value = 2 ** (activation_bits - 1) - 1
    inputs_int = torch.clamp(inputs_int, -thd_value, thd_value)
    
    # 记录维度
    batch_size = inputs.shape[0]
    in_h = inputs.shape[2]
    in_w = inputs.shape[3]
    oc = addr[3]
    
    # 2. 输入与权重复制
    if weight_row_copy > 1:
        assert isinstance(weight_row_copy, int)
        inputs_int = inputs_int.repeat(1, weight_row_copy, 1, 1)
        
    # 3. 输入image2col
    inputs_int = img2col(inputs_int, (kernel,kernel), stride=stride, padding=padding)
    
    # 4. 根据DAC的bit数展开 (默认DAC为1bit) input expansion method = 0 等权展开; input expansion method = 1 按bit数展开
    sdk = SDKArray(chip_id)
    inputs_int = inputs_int.numpy()
    outputs = sdk._calculate(inputs_int, addr, it_time=integration_time, expand_mode=input_expansion_method)
    outputs = torch.from_numpy(outputs)
    
    # 5. 还原输出维度
    out_h = (in_h + 2 * padding - kernel) // stride + 1
    out_w = (in_w + 2 * padding - kernel) // stride + 1
    outputs = outputs.reshape(batch_size, out_h, out_w, oc).permute(0, 3, 1, 2).contiguous()
    
    # 平均复制份数
    # outputs = outputs / weight_row_copy
    
    # 6. 还原输出大小
    outputs = outputs * output_dequant_scale
       
    return outputs

def C200_FC_Chip(chip_id, inputs, addr, *, input_quant_scale = 1.0, output_dequant_scale = 1.0,
                activation_bits = 4, input_expansion_method = 0, integration_time = 0, weight_row_copy = 1):
    
    # 1. 输入量化, input quant scale 是 训练过程中学习而来的
    inputs_int = (inputs / input_quant_scale).round()
    thd_value = 2 ** (activation_bits - 1) - 1
    inputs_int = torch.clamp(inputs_int, -thd_value, thd_value)
    
    # 2. 输入与权重复制
    if weight_row_copy > 1:
        assert isinstance(weight_row_copy, int)
        inputs_int = inputs_int.repeat(1, weight_row_copy)
    
    # 4. 根据DAC的bit数展开 (默认DAC为1bit) input expansion method = 0 等权展开; input expansion method = 1 按bit数展开
    sdk = SDKArray(chip_id)
    inputs_int = inputs_int.numpy()
    outputs = sdk._calculate(inputs_int, addr, it_time=integration_time, expand_mode=input_expansion_method)
    outputs = torch.from_numpy(outputs)
    
    # 平均复制份数
    # outputs = outputs / weight_row_copy
    
    # 还原输出大小
    outputs = outputs * output_dequant_scale
    return outputs


def C200_Conv2d_SIM(inputs, weights, *, kernel = 1, stride = 1, padding = 0, input_quant_scale = 1.0, output_dequant_scale = 1.0,
                    activation_bits = 4, input_expansion_method = 0, integration_time = 0, weight_row_copy = 1, PE_weight_noise=0.0, 
                    device='cpu'):
    
    # 1. 输入量化, input quant scale 是 训练过程中学习而来的
    inputs_int = (inputs / input_quant_scale).round()
    thd_value = 2 ** (activation_bits - 1) - 1
    inputs_int = torch.clamp(inputs_int, -thd_value, thd_value)
    
    # 记录维度
    batch_size = inputs.shape[0]
    in_h = inputs.shape[2]
    in_w = inputs.shape[3]
    oc = weights.shape[0]
    
    # 2. 输入与权重复制
    if weight_row_copy > 1:
        assert isinstance(weight_row_copy, int)
        inputs_int = inputs_int.repeat(1, weight_row_copy, 1, 1)
        weights = weights.repeat(1, weight_row_copy, 1, 1)
    
    # 3. 输入image2col
    inputs_int = img2col(inputs_int, (kernel,kernel), stride=stride, padding=padding)
    
    # 4. 根据DAC的bit数展开 (默认DAC为1bit) input expansion method = 0 等权展开; input expansion method = 1 按bit数展开
    if input_expansion_method == 0:
        inputs_int = input_multi_bits_pulse_expansion(inputs_int, pulse_half_level=1)
        len_ = inputs_int.shape[-1]
        outputs = 0
        for i in range(len_):
            partial_sum = C200_MVM_1bIN_4bOUT_SIM(inputs_int[:,:,i], weights, conductance_noise=PE_weight_noise, integration_time=integration_time, device=device)
            outputs = outputs + partial_sum
    
    elif input_expansion_method == 1:
        # assert thd_value == 7, f'暂不支持 {thd_value} !!!'
        # # 输入拆分
        # input_x4 = (inputs_int / 4).trunc()
        # input_x2 = ((inputs_int / 4).frac() * 2).trunc()
        # input_x1 = (inputs_int / 2).frac() * 2
        
        # # 按bit计算三次
        # output_x4 =  C200_MVM_1bIN_4bOUT_SIM(input_x4, weights, conductance_noise=PE_weight_noise, integration_time=integration_time, device=device)
        # output_x2 =  C200_MVM_1bIN_4bOUT_SIM(input_x2, weights, conductance_noise=PE_weight_noise, integration_time=integration_time, device=device)
        # output_x1 =  C200_MVM_1bIN_4bOUT_SIM(input_x1, weights, conductance_noise=PE_weight_noise, integration_time=integration_time, device=device)
        # # 
        # outputs = output_x4 * 4 + output_x2 * 2 + output_x1
        
        # 支持任意8bit的计算
        bit_length = activation_bits - 1
        outputs = 0
        for bit_index in range(bit_length):
            input_bit_value = get_bit_value(inputs_int, bit_index)
            output_bit_value = C200_MVM_1bIN_4bOUT_SIM(input_bit_value, weights, conductance_noise=PE_weight_noise, ADC_offset = ADC_offset, 
                                                       integration_time=integration_time, device=device)
            outputs = outputs + output_bit_value * 2**(bit_index)
        
    else:
        raise ValueError(f'暂不支持 输入展开方式 {input_expansion_method} !!!')
     
    # 5. 还原输出维度
    out_h = (in_h + 2 * padding - kernel) // stride + 1
    out_w = (in_w + 2 * padding - kernel) // stride + 1
    outputs = outputs.reshape(batch_size, out_h, out_w, oc).permute(0, 3, 1, 2).contiguous()
    
    # 平均复制份数
    # outputs = outputs / weight_row_copy
    
    # 6. 还原输出大小
    outputs = outputs * output_dequant_scale
       
    return outputs

def C200_FC_SIM(inputs, weights, *, input_quant_scale = 1, output_dequant_scale = 1.0, activation_bits = 4, 
                input_expansion_method = 0, integration_time = 0, weight_row_copy = 1, PE_weight_noise=0.0, 
                device='cpu'):
    
    # 1. 输入量化, input quant scale 是 训练过程中学习而来的
    inputs_int = (inputs / input_quant_scale).round()
    thd_value = 2 ** (activation_bits - 1) - 1
    inputs_int = torch.clamp(inputs_int, -thd_value, thd_value)
    
    # 2. 输入与权重复制
    if weight_row_copy > 1:
        assert isinstance(weight_row_copy, int)
        inputs_int = inputs_int.repeat(1, weight_row_copy)
        weights = weights.repeat(1, weight_row_copy)
    
    # 4. 根据DAC的bit数展开 (默认DAC为1bit) input expansion method = 0 等权展开; input expansion method = 1 按bit数展开
    if input_expansion_method == 0:
        inputs_int = input_multi_bits_pulse_expansion(inputs_int)
        len_ = inputs_int.shape[-1]
        outputs = 0
        for i in range(len_):
            partial_sum = C200_MVM_1bIN_4bOUT_SIM(inputs_int[:,:,i], weights, integration_time=integration_time, device=device)
            outputs = outputs + partial_sum
    elif input_expansion_method == 1:
        
        # 支持任意8bit的计算
        bit_length = activation_bits - 1
        outputs = 0
        for bit_index in range(bit_length):
            input_bit_value = get_bit_value(inputs_int, bit_index)
            output_bit_value = C200_MVM_1bIN_4bOUT_SIM(input_bit_value, weights, conductance_noise=PE_weight_noise, ADC_offset = ADC_offset, 
                                                       integration_time=integration_time, device=device)
            outputs = outputs + output_bit_value * 2**(bit_index)
        
        # assert thd_value == 7, f'暂不支持 {thd_value} !!!'
        # # 输入拆分
        # input_x4 = (inputs_int / 4).trunc()
        # input_x2 = ((inputs_int / 4).frac() * 2).trunc()
        # input_x1 = (inputs_int / 2).frac() * 2
        
        # # 按bit计算三次
        # output_x4 =  C200_MVM_1bIN_4bOUT_SIM(input_x4, weights, conductance_noise=PE_weight_noise, integration_time=integration_time, device=device)
        # output_x2 =  C200_MVM_1bIN_4bOUT_SIM(input_x2, weights, conductance_noise=PE_weight_noise, integration_time=integration_time, device=device)
        # output_x1 =  C200_MVM_1bIN_4bOUT_SIM(input_x1, weights, conductance_noise=PE_weight_noise, integration_time=integration_time, device=device)
        
        # # 
        # outputs = output_x4 * 4 + output_x2 * 2 + output_x1
    else:
        raise ValueError(f'暂不支持 输入展开方式 {input_expansion_method} !!!')
    
    # 平均复制份数
    # outputs = outputs / weight_row_copy
    
    # 还原输出大小
    outputs = outputs * output_dequant_scale
    
    return outputs

def C200_Add(inputs):
    
    output = 0
    # 累加
    for i in inputs:
        output += i
    return output
    
def C200_Add_4bit(inputs):
    
    output = 0
    # 累加
    for i in inputs:
        output += i
    output = torch.clamp(output, min=-8, max=7)
    return output

def C200_Add_8bit(inputs):
    
    output = 0
    # 累加
    for i in inputs:
        output += i
    output = torch.clamp(output, min=-128, max=127)
    return output

def C200_ReLU(input): 
    return torch.clamp(input, min=0)

def C200_SiLU_4bit(input, lut): 
    output_query = input + 8
    output_query = output_query.to(torch.int32).long()
    output = lut[output_query]
    return output

def C200_SiLU_8bit(input, lut): 
    output_query = input + 128
    output_query = output_query.to(torch.int32).long()
    output = lut[output_query]
    return output

def C200_Add_ReLU_4bit(inputs):
    
    output = C200_Add_4bit(inputs)
    output = C200_ReLU(output)    
    return output

def C200_Add_ReLU_Split_4bit(inputs, split, dim=0):
    
    output = C200_Add_4bit(inputs)
    output = C200_ReLU(output)
    output = C200_Split(output, split, dim=dim)    
    return output

def C200_Add_ReLU_8bit(inputs):
    
    output = C200_Add_8bit(inputs)
    output = C200_ReLU(output)    
    return output

def C200_Add_ReLU_Split_8bit(inputs, split, dim=0):
    
    output = C200_Add_8bit(inputs)
    output = C200_ReLU(output)
    output = C200_Split(output, split, dim=dim)    
    return output

def C200_Add_SiLU_4bit(inputs, lut):
    
    output = C200_Add_4bit(inputs)
    output = C200_SiLU_4bit(output, lut)    
    return output

def C200_Add_SiLU_8bit(inputs, lut):
    
    output = C200_Add_8bit(inputs)
    output = C200_SiLU_8bit(output, lut)    
    return output

def C200_Add_SiLU_Split_4bit(inputs, lut, split, dim=0):
    
    output = C200_Add_4bit(inputs)
    output = C200_SiLU_4bit(output, lut)
    output = C200_Split(output, split, dim=dim)    
    return output

def C200_Add_SiLU_Split_8bit(inputs, lut, split, dim=0):
    
    output = C200_Add_8bit(inputs)
    output = C200_SiLU_8bit(output, lut) 
    output = C200_Split(output, split, dim=dim)    
    return output

def C200_Concat(inputs, axis=1):
    output = torch.cat(inputs, axis=axis)
    return output

def C200_Split(input, split, dim=0):
    output = torch.split(input, split, dim=dim)
    return output

def C200_Mul_Add_4bit(input_data, scale=1, scale_shift_num=0, offset=0):
    # mul
    output_data = (input_data * scale).to(torch.int32)
    output_data = output_data >> scale_shift_num
    # offset
    output_data = output_data + offset
    output_data = output_data.to(torch.int32)
    output_data = torch.clamp(output_data, min=-8, max=7)
    return output_data

def C200_Mul_Add_8bit(input_data, scale=1, scale_shift_num=0, offset=0):
    # mul
    output_data = (input_data * scale).to(torch.int32)
    output_data = output_data >> scale_shift_num
    # offset
    output_data = output_data + offset
    output_data = output_data.to(torch.int32)
    output_data = torch.clamp(output_data, min=-128, max=127)
    
    return output_data

def C200_Maxpool2d(input_data, kernel_size=1, stride=0, padding=0):
    input_data = input_data.to(torch.float32)
    output_data = torch.max_pool2d(input_data, kernel_size, stride, padding)
    return output_data

def C200_Avgpool2d(input_data, kernel_size=1, stride=0, padding=0, device='cpu'):
    # c = input_data.shape[1]
    # # 1. 生成一个全1的权重
    # pool_weight = torch.ones((c, 1, kernel_size, kernel_size)).to(device)
    # # 2. 计算按通道进行求和，采用group conv
    # output_data = torch.conv2d(input_data, pool_weight, stride=stride, padding=padding, groups=c,)
    # output_data = output_data / (kernel_size * kernel_size)
    output_data = F.avg_pool2d(input_data, kernel_size, stride, padding)
    return output_data

def C200_Resize(input_data, size=None, scale_factor=[1, 1]):
    input_data = input_data.to(torch.float32)
    output_data = F.interpolate(input_data, size=size, scale_factor=scale_factor)
    return output_data

def C200_Concat_Split(inputs, axis, split, dim=0):
    output = C200_Concat(inputs, axis=axis)
    output = C200_Split(output, split, dim=dim)
    return output

def C200_Concat_SiLU_Split_4bit(inputs, lut, axis, split, dim=0):
    output = C200_Concat(inputs, axis=axis)
    output = C200_SiLU_4bit(output, lut)
    output = C200_Split(output, split, dim=dim)
    return output

def C200_Concat_SiLU_Split_8bit(inputs, lut, axis, split, dim=0):
    output = C200_Concat(inputs, axis=axis)
    output = C200_SiLU_8bit(output, lut)
    output = C200_Split(output, split, dim=dim)
    return output

def C200_Concat_SiLU_4bit(inputs, lut, axis):
    output = C200_Concat(inputs, axis=axis)
    output = C200_SiLU_4bit(output, lut)
    return output

def C200_Concat_SiLU_8bit(inputs, lut, axis):
    output = C200_Concat(inputs, axis=axis)
    output = C200_SiLU_8bit(output, lut)
    return output

def C200_Pad(inputs, pad, value):
    output = F.pad(inputs, pad=pad, value=value)
    return output 

def C200_Relu(inputs):
    output = torch.clamp(inputs, min=0)
    return output

def C200_8bit_to_4bit(inputs):
    output = torch.clamp(inputs, min=-8, max=7)
    return output

def C200_4bit_to_8bit(inputs):
    output = inputs
    return output

def C200_BatchNorm2d(inputs, *, epsilon=None, weights=None, bias=None, mean=None, var=None):
    weights = weights.view(1, -1, 1, 1)
    bias = bias.view(1, -1, 1, 1)
    mean = mean.view(1, -1, 1, 1)
    var = var.view(1, -1, 1, 1) 
    return ((inputs - mean) / torch.sqrt(var + epsilon)) * weights + bias
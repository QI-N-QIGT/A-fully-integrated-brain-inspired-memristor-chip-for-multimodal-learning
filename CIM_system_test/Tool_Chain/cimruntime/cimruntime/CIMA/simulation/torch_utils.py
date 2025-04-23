import torch
import torch.nn.functional as F

high_4bit_lut = torch.load(fr'D:\Repository\cimruntime\cimruntime\CIMA\simulation\CIMA_8to4_lut_high.pth')
low_4bit_lut = torch.load(fr'D:\Repository\cimruntime\cimruntime\CIMA\simulation\CIMA_8to4_lut_low.pth')

if torch.cuda.is_available():
    high_4bit_lut_cuda = high_4bit_lut.to('cuda')
    low_4bit_lut_cuda = low_4bit_lut.to('cuda')

def CIMA_PEConv_4bit(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, scale = 1, offset = 0, scale_shift_num = 0, 
                    accumulate_shift_num = 0, max_conductance = 36, max_voltage = 0.0957, device='cpu', 
                    jump_bn = False, relu = False, is_scale_first = False):
    
    # jump_bn only for debug
    
    scale = scale[0]
    offset = offset[0]
    scale_shift_num = scale_shift_num[0]
    
    # 量化电流与档位的对应关系
    max_current = {0:32, 1:40, 2:64, 3:80, 4:120, 5:160, 6:200}
    
    # 根据输入和权重，转化为对应的电导值以及电压值
    input_voltage = input_data / 7 * max_voltage
    weight_conductance = weight_data / 127 * max_conductance
    
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
    output_quant = output_current / max_current[ADC_quant_level] * 127
    output_quant = torch.round(output_quant).to(torch.int32)
    # # print(f'CIMA analog mac output data type: {output_quant.dtype}')

    # 量化偏移误差
    if ADC_offset != 0:
        ADC_offset = torch.round( torch.randn_like(output_current).clamp_(-3.0, 3.0) * ADC_offset).to(device)
        # output_quant = output_quant + torch.randint(-ADC_offset, ADC_offset, size=output_quant.shape).to(device)
        output_quant = output_quant + ADC_offset
    
    # 电流截断
    output_quant = torch.clamp(output_quant, min=-128, max=127)
    
    if not jump_bn:
            
        # scale + 移位 校准 (bn)
        if torch.is_tensor(scale):
            scale = scale.view(1, -1, 1, 1)
        if torch.is_tensor(offset):
            offset = offset.view(1, -1, 1, 1)
        
        if is_scale_first:
            # 先乘 scale, 移位，再加offset
            output_quant = (output_quant * scale).to(torch.int32)
            output_quant = output_quant >> scale_shift_num
            
            # offset 校准 (bn)
            output_quant = (output_quant + offset).to(torch.int32)
        else:
            # 先加offset，再乘 scale, 移位
            # offset 校准 (bn)
            output_quant = (output_quant + offset).to(torch.int32)
            # scale
            output_quant = (output_quant * scale).to(torch.int32)
            output_quant = output_quant >> scale_shift_num
        
        # 输出移位
        if accumulate_shift_num > 0:
            output_quant = output_quant >> accumulate_shift_num
        elif accumulate_shift_num < 0:
            output_quant = output_quant << abs(accumulate_shift_num)
        
        # 输出截断到4bit
        output_quant = torch.clamp(output_quant, min=-8, max=7)
    
    return output_quant   

def CIMA_PEConv_4bIN_to_4bOUT(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, scale=[1], offset=[0], scale_shift_num=[0],
                    accumulate_shift_num = 0, max_conductance = 36, max_voltage = 0.0957, device='cpu', 
                    jump_bn = False, relu = False, is_scale_first = False):
    
    return CIMA_PEConv_4bit(input_data, weight_data, stride=stride, padding=padding, 
                    DAC_noise = DAC_noise, conductance_noise = conductance_noise, ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                    ADC_quant_level = ADC_quant_level, scale = scale, offset = offset, scale_shift_num = scale_shift_num, 
                    accumulate_shift_num = accumulate_shift_num, max_conductance = max_conductance, max_voltage = max_voltage, device=device, 
                    jump_bn = jump_bn, relu = relu, is_scale_first = is_scale_first)

def CIMA_PEConv_4bIN_to_8bOUT(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, scale=[1], offset=[0], scale_shift_num=[0], 
                    accumulate_shift_num = 0, max_conductance = 36, max_voltage = 0.0957, device='cpu', 
                    jump_bn = False, relu = False, is_scale_first = False):
    # jump_bn only for debug
    
    scale = scale[0]
    offset = offset[0]
    scale_shift_num = scale_shift_num[0]
    
    # 量化电流与档位的对应关系
    max_current = {0:32, 1:40, 2:64, 3:80, 4:120, 5:160, 6:200}
    
    # 根据输入和权重，转化为对应的电导值以及电压值
    input_voltage = input_data / 7 * max_voltage
    weight_conductance = weight_data / 127 * max_conductance
    
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
    output_quant = output_current / max_current[ADC_quant_level] * 127
    output_quant = torch.round(output_quant).to(torch.int32)
    # # print(f'CIMA analog mac output data type: {output_quant.dtype}')

    # 量化偏移误差
    if ADC_offset != 0:
        ADC_offset = torch.round( torch.randn_like(output_current).clamp_(-3.0, 3.0) * ADC_offset).to(device)
        # output_quant = output_quant + torch.randint(-ADC_offset, ADC_offset, size=output_quant.shape).to(device)
        output_quant = output_quant + ADC_offset
    
    # 电流截断
    output_quant = torch.clamp(output_quant, min=-128, max=127)
    
    if not jump_bn:
            
        # scale + 移位 校准 (bn)
        if torch.is_tensor(scale):
            scale = scale.view(1, -1, 1, 1)
        if torch.is_tensor(offset):
            offset = offset.view(1, -1, 1, 1)
        
        if is_scale_first:
            # 先乘 scale, 移位，再加offset
            output_quant = (output_quant * scale).to(torch.int32)
            output_quant = output_quant >> scale_shift_num
            
            # offset 校准 (bn)
            output_quant = (output_quant + offset).to(torch.int32)
        else:
            # 先加offset，再乘 scale, 移位
            # offset 校准 (bn)
            output_quant = (output_quant + offset).to(torch.int32)
            # scale
            output_quant = (output_quant * scale).to(torch.int32)
            output_quant = output_quant >> scale_shift_num
        
        # 输出移位
        if accumulate_shift_num > 0:
            output_quant = output_quant >> accumulate_shift_num
        elif accumulate_shift_num < 0:
            output_quant = output_quant << abs(accumulate_shift_num)
        
        # 输出截断到8bit
        output_quant = torch.clamp(output_quant, min=-128, max=127)
    
    return output_quant 

def CIMA_PEConv_8bit(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, scale=[1], offset=[0], scale_shift_num=[0], 
                    accumulate_shift_num = 0, max_conductance = 36, max_voltage = 0.0957, device='cpu', 
                    jump_bn = False, relu=False, is_scale_first=False):
    # jump_bn only for debug
    
    # 区分高低4bit，各自的scale, offset 以及shift_num不同 [H4BIT, L4BIT]
    assert len(scale) == 2
    assert len(offset) == 2
    assert len(scale_shift_num) == 2
    
    if padding != 0:
        input_data = F.pad(input_data, pad=(padding,padding,padding,padding,0,0,0,0))
        padding = 0
    
    # 输入则需要先转换为高低4bit, 然后分别做MAC, 然后再做累加
    if device == 'cuda':
        input_data = input_data.to('cpu')
        # lut index
        input_data_lut_index = torch.LongTensor((input_data + 128).to(torch.int64))
        # high 4bit
        input_high_4bit = high_4bit_lut_cuda[input_data_lut_index]
        # low 4bit
        input_low_4bit = low_4bit_lut_cuda[input_data_lut_index]
    else:
        # lut index
        input_data_lut_index = torch.LongTensor((input_data + 128).to(torch.int64))
        # high 4bit
        input_high_4bit = high_4bit_lut[input_data_lut_index]
        # low 4bit
        input_low_4bit = low_4bit_lut[input_data_lut_index]
    
    # 高4bit计算
    output_high_4bit = CIMA_PEConv_4bIN_to_8bOUT(input_high_4bit, weight_data, stride=stride, padding=padding, 
                    DAC_noise = DAC_noise, conductance_noise = conductance_noise, ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                    ADC_quant_level = ADC_quant_level, scale = scale[0], offset = offset[0], scale_shift_num = scale_shift_num[0], 
                    accumulate_shift_num = 0, max_conductance = max_conductance, max_voltage = max_voltage, device=device, 
                    jump_bn = jump_bn, relu = relu, is_scale_first=is_scale_first)
    # 低4bit计算
    output_low_4bit = CIMA_PEConv_4bIN_to_8bOUT(input_low_4bit, weight_data, stride=stride, padding=padding, 
                    DAC_noise = DAC_noise, conductance_noise = conductance_noise, ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                    ADC_quant_level = ADC_quant_level, scale = scale[1], offset = offset[1], scale_shift_num = scale_shift_num[1], 
                    accumulate_shift_num = 0, max_conductance = max_conductance, max_voltage = max_voltage, device=device, 
                    jump_bn = jump_bn, relu = relu, is_scale_first=is_scale_first)
    
    # 高低4bit移位累加
    output_quant = output_high_4bit * 16 + output_low_4bit
    output_quant = output_quant.to(torch.int32)

    # output_quant = (output_quant * scale_).to(torch.int32)
    # output_quant = output_quant >> scale_shift_num_
    
    # # offset 校准 (bn)
    # output_quant = (output_quant + offset_).to(torch.int32)
    # accumulate_shift_num = 0
    
    # 输出移位
    if accumulate_shift_num > 0:
        output_quant = output_quant >> accumulate_shift_num
    else:
        output_quant = output_quant << abs(accumulate_shift_num)
    
    # 输出截断到8bit
    output_quant = torch.clamp(output_quant, min=-128, max=127)

    return output_quant

def CIMA_PEConv_8bIN_to_8bOUT(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, scale=[1], offset=[0], scale_shift_num=[0],
                    accumulate_shift_num = 0, max_conductance = 36, max_voltage = 0.0957, device='cpu', 
                    jump_bn = False, relu=False, is_scale_first = False):
    
    return CIMA_PEConv_8bit(input_data, weight_data, stride=stride, padding=padding, 
                    DAC_noise = DAC_noise, conductance_noise = conductance_noise, ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                    ADC_quant_level = ADC_quant_level, scale = scale, offset = offset, scale_shift_num = scale_shift_num, 
                    accumulate_shift_num = accumulate_shift_num, max_conductance = max_conductance, max_voltage = max_voltage, device=device, 
                    jump_bn = jump_bn, relu = relu, is_scale_first = is_scale_first)
    
def CIMA_PEConv_8bIN_to_4bOUT(input_data, weight_data, *, stride=1, padding=0, 
                    DAC_noise = 0, conductance_noise = 0, ADC_noise = 0, ADC_offset = 0,
                    ADC_quant_level = 0, scale=[1], offset=[0], scale_shift_num=[0], 
                    accumulate_shift_num = 0, max_conductance = 36, max_voltage = 0.0957, device='cpu', 
                    jump_bn = False, relu=False, is_scale_first = False):
    # jump_bn only for debug
    
    if padding != 0:
        input_data = F.pad(input_data, pad=(1,1,1,1,0,0,0,0))
        padding = 0
    
    # 输入则需要先转换为高低4bit, 然后分别做MAC, 然后再做累加
    if device == 'cuda':
        input_data = input_data.to('cpu')
        # lut index
        input_data_lut_index = torch.LongTensor((input_data + 128).to(torch.int64))
        # high 4bit
        input_high_4bit = high_4bit_lut_cuda[input_data_lut_index]
        # low 4bit
        input_low_4bit = low_4bit_lut_cuda[input_data_lut_index]
    else:
        # lut index
        input_data_lut_index = torch.LongTensor((input_data + 128).to(torch.int64))
        # high 4bit
        input_high_4bit = high_4bit_lut[input_data_lut_index]
        # low 4bit
        input_low_4bit = low_4bit_lut[input_data_lut_index]
    
    # 高4bit计算
    output_high_4bit = CIMA_PEConv_4bIN_to_8bOUT(input_high_4bit, weight_data, stride=stride, padding=padding, 
                    DAC_noise = DAC_noise, conductance_noise = conductance_noise, ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                    ADC_quant_level = ADC_quant_level, scale = scale[0], offset = offset[0], scale_shift_num = scale_shift_num[0], 
                    accumulate_shift_num = 0, max_conductance = max_conductance, max_voltage = max_voltage, device=device, 
                    jump_bn = jump_bn, relu = relu, is_scale_first = is_scale_first)
    # 低4bit计算
    output_low_4bit = CIMA_PEConv_4bIN_to_8bOUT(input_low_4bit, weight_data, stride=stride, padding=padding, 
                    DAC_noise = DAC_noise, conductance_noise = conductance_noise, ADC_noise = ADC_noise, ADC_offset = ADC_offset,
                    ADC_quant_level = ADC_quant_level, scale = scale[1], offset = offset[1], scale_shift_num = scale_shift_num[1], 
                    accumulate_shift_num = 0, max_conductance = max_conductance, max_voltage = max_voltage, device=device, 
                    jump_bn = jump_bn, relu = relu, is_scale_first = is_scale_first)
    
    # 高低4bit移位累加
    output_quant = output_high_4bit * 16 + output_low_4bit
    output_quant = output_quant.to(torch.int32)

    # # scale + 移位 校准 (bn)
    # if torch.is_tensor(scale):
    #     scale = scale.view(1, -1, 1, 1)
    # if torch.is_tensor(offset):
    #     offset = offset.view(1, -1, 1, 1)
    
    # output_quant = (output_quant * scale).to(torch.int32)
    # output_quant = output_quant >> scale_shift_num
    
    # # offset 校准 (bn)
    # output_quant = (output_quant + offset).to(torch.int32)
    
    # 输出移位
    if accumulate_shift_num > 0:
        output_quant = output_quant >> accumulate_shift_num
    else:
        output_quant = output_quant << abs(accumulate_shift_num)
    
    # 输出截断到4bit
    output_quant = torch.clamp(output_quant, min=-8, max=7)

    return output_quant

def CIMA_DMACConv_8bit(input_data, weight_data, *, stride=1, padding=0, scale=[1], offset=[0], scale_shift_num=[0],
                       accumulate_shift_num = 0, jump_bn = True, relu = False, is_scale_first = False):
    # jump_bn only for debug
    
    scale = scale[0]
    offset = offset[0]
    scale_shift_num = scale_shift_num[0]
    
    # 前置 relu
    if relu:
        input_data = torch.clamp(input_data, min=0)   
    
    # 数据类型转换
    input_data = input_data.to(torch.float32)
    weight_data = weight_data.to(torch.float32)
    
    # 计算
    output_data = torch.conv2d(input_data, weight_data, bias=None, stride=stride, padding=padding)
    output_data = output_data.to(torch.int32)
    
    # 输出移位
    if accumulate_shift_num > 0:
        output_data = output_data >> accumulate_shift_num
    else:
        output_data = output_data << abs(accumulate_shift_num)
    
    # 输出截断到8bit
    output_data = torch.clamp(output_data, min=-128, max=127)
    
    if not jump_bn:
        
        # scale + 移位 校准 (bn)
        if torch.is_tensor(scale):
            scale = scale.view(1, -1, 1, 1)
        if torch.is_tensor(offset):
            offset = offset.view(1, -1, 1, 1)
            
        if is_scale_first:
            # 先乘 scale, 移位，再加offset
            output_data = (output_data * scale).to(torch.int32)
            output_data = output_data >> scale_shift_num
            
            # offset 校准 (bn)
            output_data = (output_data + offset).to(torch.int32)
        else:
            # 先加offset，再乘 scale, 移位
            # offset 校准 (bn)
            output_data = (output_data + offset).to(torch.int32)
            # scale
            output_data = (output_data * scale).to(torch.int32)
            output_data = output_data >> scale_shift_num
        
        # 输出截断到8bit
        output_data = torch.clamp(output_data, min=-128, max=127)
     
    return output_data

CIMA_DMACConv_8bIN_to_8bOUT = CIMA_DMACConv_8bit 

def CIMA_Add_4bit(inputs):
    
    output = 0
    # 累加
    for i in inputs:
        output += i
    output = torch.clamp(output, min=-8, max=7)
    return output

def CIMA_Add_8bit(inputs):
    
    output = 0
    # 累加
    for i in inputs:
        output += i
    output = torch.clamp(output, min=-128, max=127)
    return output

def CIMA_ReLU(input): 
    return torch.clamp(input, min=0)

def CIMA_SiLU_4bit(input, lut): 
    output_query = input + 8
    output_query = output_query.to(torch.int32).long()
    output = lut[output_query]
    return output

def CIMA_SiLU_8bit(input, lut): 
    output_query = input + 128
    output_query = output_query.to(torch.int32).long()
    output = lut[output_query]
    return output

def CIMA_Add_ReLU_4bit(inputs):
    
    output = CIMA_Add_4bit(inputs)
    output = CIMA_ReLU(output)    
    return output

def CIMA_Add_ReLU_Split_4bit(inputs, split, dim=0):
    
    output = CIMA_Add_4bit(inputs)
    output = CIMA_ReLU(output)
    output = CIMA_Split(output, split, dim=dim)    
    return output

def CIMA_Add_ReLU_8bit(inputs):
    
    output = CIMA_Add_8bit(inputs)
    output = CIMA_ReLU(output)    
    return output

def CIMA_Add_ReLU_Split_8bit(inputs, split, dim=0):
    
    output = CIMA_Add_8bit(inputs)
    output = CIMA_ReLU(output)
    output = CIMA_Split(output, split, dim=dim)    
    return output

def CIMA_Add_SiLU_4bit(inputs, lut):
    
    output = CIMA_Add_4bit(inputs)
    output = CIMA_SiLU_4bit(output, lut)    
    return output

def CIMA_Add_SiLU_8bit(inputs, lut):
    
    output = CIMA_Add_8bit(inputs)
    output = CIMA_SiLU_8bit(output, lut)    
    return output

def CIMA_Add_SiLU_Split_4bit(inputs, lut, split, dim=0):
    
    output = CIMA_Add_4bit(inputs)
    output = CIMA_SiLU_4bit(output, lut)
    output = CIMA_Split(output, split, dim=dim)    
    return output

def CIMA_Add_SiLU_Split_8bit(inputs, lut, split, dim=0):
    
    output = CIMA_Add_8bit(inputs)
    output = CIMA_SiLU_8bit(output, lut) 
    output = CIMA_Split(output, split, dim=dim)    
    return output

def CIMA_Concat(inputs, axis=1):
    output = torch.cat(inputs, axis=axis)
    return output

def CIMA_Split(input, split, dim=0):
    output = torch.split(input, split, dim=dim)
    return output

def CIMA_Mul_Add_4bit(input_data, scale=1, scale_shift_num=0, offset=0):
    # mul
    output_data = (input_data * scale).to(torch.int32)
    output_data = output_data >> scale_shift_num
    # offset
    output_data = output_data + offset
    output_data = output_data.to(torch.int32)
    output_data = torch.clamp(output_data, min=-8, max=7)
    return output_data

def CIMA_Mul_Add_8bit(input_data, scale=1, scale_shift_num=0, offset=0):
    # mul
    output_data = (input_data * scale).to(torch.int32)
    output_data = output_data >> scale_shift_num
    # offset
    output_data = output_data + offset
    output_data = output_data.to(torch.int32)
    output_data = torch.clamp(output_data, min=-128, max=127)
    
    return output_data

def CIMA_Maxpool2d(input_data, kernel_size=1, stride=0, padding=0):
    input_data = input_data.to(torch.float32)
    output_data = torch.max_pool2d(input_data, kernel_size, stride, padding)
    return output_data

def CIMA_Avgpool2d(input_data, kernel_size=1, stride=0, padding=0, shift_num=0, device='cpu'):
    input_data = input_data.to(torch.float32)
    # 平均池化通过求和+移位来实现
    b, c, h, w = input_data.shape
    # 1. 生成一个全1的权重
    pool_weight = torch.ones((c, 1, kernel_size, kernel_size)).to(device)
    # 2. 计算按通道进行求和，采用group conv
    output_data = torch.conv2d(input_data, pool_weight, stride=stride, padding=padding, groups=c,)
    # 3. 移位
    output_data = output_data.to(torch.int32)
    output_data = output_data >> shift_num
    
    return output_data

def CIMA_Resize(input_data, size=None, scale_factor=[1, 1]):
    input_data = input_data.to(torch.float32)
    output_data = F.interpolate(input_data, size=size, scale_factor=scale_factor)
    return output_data

def CIMA_Concat_Split(inputs, axis, split, dim=0):
    output = CIMA_Concat(inputs, axis=axis)
    output = CIMA_Split(output, split, dim=dim)
    return output

def CIMA_Concat_SiLU_Split_4bit(inputs, lut, axis, split, dim=0):
    output = CIMA_Concat(inputs, axis=axis)
    output = CIMA_SiLU_4bit(output, lut)
    output = CIMA_Split(output, split, dim=dim)
    return output

def CIMA_Concat_SiLU_Split_8bit(inputs, lut, axis, split, dim=0):
    output = CIMA_Concat(inputs, axis=axis)
    output = CIMA_SiLU_8bit(output, lut)
    output = CIMA_Split(output, split, dim=dim)
    return output

def CIMA_Concat_SiLU_4bit(inputs, lut, axis):
    output = CIMA_Concat(inputs, axis=axis)
    output = CIMA_SiLU_4bit(output, lut)
    return output

def CIMA_Concat_SiLU_8bit(inputs, lut, axis):
    output = CIMA_Concat(inputs, axis=axis)
    output = CIMA_SiLU_8bit(output, lut)
    return output

def CIMA_Pad(inputs, pad, value):
    output = F.pad(inputs, pad=pad, value=value)
    return output 

def CIMA_Relu(inputs):
    output = torch.clamp(inputs, min=0)
    return output

def CIMA_8bit_to_4bit(inputs):
    output = torch.clamp(inputs, min=-8, max=7)
    return output

def CIMA_4bit_to_8bit(inputs):
    output = inputs
    return output
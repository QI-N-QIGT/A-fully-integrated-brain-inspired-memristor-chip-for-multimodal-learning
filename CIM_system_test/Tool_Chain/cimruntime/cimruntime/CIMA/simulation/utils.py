import time
import numpy as np

def CIMA_8bit_to_4bit(a):
    
    assert a <= 127 and a>=-128
    complement_code = {'1000':-8, '1001':-7, '1010': -6, '1011': -5, '1100': -4, '1101': -3, '1110': -2, '1111': -1}
    batch, h, w = a.shape
    batch_high_4bit = []
    batch_low_4bit = []
    
    for k in range(batch):
        b_high_4bit = []
        b_low_4bit = []
        for i in range(h):
            b_high_4bit_partial = []
            b_low_4bit_partial = []
            for j in range(w):
                # 1. 将 a 转换为二进制补码表示
                b_ = bin(int(a[i][j]) & 0xff)[2:]
                b_ = list(b_)
                while len(b_) < 8:
                    b_.insert(0, '0')
                b_ = ''.join(b_)
                high_bits = b_[0:4]
                low_bits = b_[4:]
                # 2. 负数的话，将其进行高低4bit的转换, 高4bit作为有符号数，低4bit看作无符号数，然后将低4bit减掉8看作有符号数
                if a[i][j] < 0: 
                    negative_high = complement_code[high_bits]
                    b_high_4bit_partial.append(negative_high)
                else:
                    b_high_4bit_partial.append(int(f'0b{high_bits}', base=2))
                b_low_4bit_partial.append(int(f'0b{low_bits}', base=2) - 8)
            # 高低4bit
            b_high_4bit.append(b_high_4bit_partial)
            b_low_4bit.append(b_low_4bit_partial)
        batch_high_4bit.append(b_high_4bit)
        batch_low_4bit.append(b_low_4bit)
        
    batch_high_4bit = np.array(batch_high_4bit, dtype=np.int32)
    batch_low_4bit = np.array(batch_low_4bit, dtype=np.int32)
    
    return batch_high_4bit, batch_low_4bit

def CIMA_array_MAC(input_data, weight_data, *, 
                    DAC_noise = 0, conductance_noise = 0, 
                    ADC_noise = 0, ADC_offset = 0, ADC_quant_level = 0,
                    scale = 1, offset = 0, scale_shift_num = 0, 
                    max_conductance = 36, max_voltage = 0.0957,
                    max_current = None):
    '''
    Analog array inference flow
    '''
    assert max_current != None, f'缺少ADC电流档位对应关系!!!'
    assert input_data.max() <= 7 and input_data.min() >= -8
    
    # 根据输入和权重，转化为对应的电导值以及电压值
    input_voltage = input_data / 7 * max_voltage
    input_voltage = input_voltage.astype(np.float32)
    
    weight_conductance = weight_data / 127 * max_conductance
    weight_conductance = weight_conductance.astype(np.float32)
    
    # 输入加噪
    if DAC_noise != 0:
        DAC_noise = np.abs(input_voltage).max() * DAC_noise
        input_voltage = input_voltage + DAC_noise * np.random.randn(*input_voltage.shape).clip(-3.0, 3.0)
    
    # 权重加噪
    # conductance_noise = 2 * max_conductance * 0.05 # 权重 5% Noise
    # print(f'conductance_noise: {conductance_noise} uS')
    if conductance_noise != 0:
        conductance_noise = np.abs(weight_conductance).max() * conductance_noise
        weight_conductance = weight_conductance + conductance_noise * np.random.randn(*weight_conductance.shape).clip(-3.0, 3.0)
    
    # 计算输出电流
    output_current = input_voltage @ weight_conductance
    
    # 输出加噪
    # ADC_noise = 2 * max_current[ADC_qunat_level] * 0.05 # 输出 5% Noise
    if ADC_noise != 0:
        ADC_noise = np.abs(output_current).max() * ADC_noise
        output_current = output_current + ADC_noise * np.random.randn(*output_current.shape).clip(-3.0, 3.0)
    
    # 电流量化
    output_quant = output_current / max_current[ADC_quant_level] * 127
    output_quant = np.around(output_quant).astype(np.int32)
    # print(f'CIMA analog mac output data type: {output_quant.dtype}')
    
    # 量化偏移误差
    if ADC_offset != 0:
        ADC_offset = np.round( np.random.randn(*output_current.shape).clip(-3.0, 3.0) * ADC_offset)
        output_quant = output_quant + ADC_offset
    
    # 电流截断
    output_quant = np.clip(output_quant, a_min=-128, a_max=127)
    
    # scale + 移位 校准 (bn)
    output_quant = (output_quant * scale).astype(np.int32)
    output_quant = output_quant >> scale_shift_num
   
    # offset 校准 (bn)
    output_quant = output_quant + offset
    
    return output_quant    

def CIMA_analog_MAC(input_data, weight_data, *, dtype='4bit', 
                    DAC_noise = 0., conductance_noise = 0., 
                    ADC_noise = 0., ADC_offset = 0, ADC_quant_level = 0,
                    scale = 1, offset = 0, scale_shift_num = 0, accumulate_shift_num = 0, 
                    max_conductance = 36, max_voltage = 0.0957):
    '''
    input_data: int8/int4, dtype: '4bit', range 为[-8, 7]; dtype: '8bit', range为[-128, 127];
    weight_data: int8, range[-128, 127], 计算时会被均匀设为电导值，最大电导值由参数 'max_conductance' 确定对应 127;
    dtype: str, 决定输入和输出的数据类型;
    DAC_noise: float, 读电压噪声, 单位为V, 表示读电压的标准差;
    conductance_noise: float, 电导值噪声, 单位为uS, 表示电导值的标准差;
    ADC_noise: float, ADC输入电路的噪声, 单位为uA, 表示读电流噪声; 
    ADC_offset: uint, ADC量化偏移误差, 无量纲, 大小与ADC输出code对应;
    ADC_quant_level: uint, ADC量化电流最大值档位; 共设计7档, 使用数字0~6表示; 0:32uA, 1:40uA; 2:64uA; 3:80uA; 4:120uA; 5:160uA; 6:200uA;
    scale: uint8, ADC 线性失调系数;
    offset: int8, ADC 线性失调截距;
    scale_shift_num: 输出数据移位大小, 0~15;
    accumulate_shift_num: int, 累加之后数据移位大小, 支持左移或者右移, 大于0表示右移(<= 23), 小于0表示左移(>=-7);
    max_conductance: float, 最大映射电导值, 单位: uS;
    max_voltage: float, 最大推理计算输入电压值, 单位: V; 即输入为7对应的输入电压;
    '''
    # assert isinstance(input_data.type, np.int32) and isinstance(weight_data, np.int32)
    
    # assert weight_data.max() <= 127 and weight_data.min() >= -128
    assert accumulate_shift_num <= 23 and accumulate_shift_num >= -7
    assert scale_shift_num >= 0 and scale_shift_num <= 15
    
    if len(input_data.shape) == 3:
        # 输入维度变换 B, OC, IC*H*W ===> B, IC*H*W, OC
        input_data = input_data.transpose(0, 2, 1)
    
    # 量化电流与档位的对应关系
    max_current = {0: 32, 1:40, 2:64, 3:80, 4:120, 5:160, 6:200}
    
    if dtype == '4bit':
        assert input_data.max() <= 7 and input_data.min() >= -8, f'{input_data.max(), input_data.min()}'
        # 根据输入和权重，转化为对应的电导值以及电压值
        # 如果是dtype为4bit, 则可以直接计算
        output_quant = CIMA_array_MAC(input_data, weight_data, 
                                        DAC_noise = DAC_noise, conductance_noise = conductance_noise, 
                                        ADC_noise = ADC_noise, ADC_offset = ADC_offset, ADC_quant_level = ADC_quant_level,
                                        scale = scale, offset = offset, scale_shift_num = scale_shift_num, 
                                        max_conductance = max_conductance, max_voltage = max_voltage, max_current = max_current)
        # 输出移位
        if accumulate_shift_num > 0:
            output_quant = output_quant >> accumulate_shift_num
        elif accumulate_shift_num < 0:
            output_quant = output_quant << abs(accumulate_shift_num)
        
        # 输出截断到4bit
        output_quant = np.clip(output_quant, a_min=-8, a_max=7)
    
    elif dtype == '8bit':
        assert input_data.max() <= 127 and input_data.min() >= -128
        # 根据输入和权重，转化为对应的电导值以及电压值
        # 如果是dtype为8bit, 输入则需要先转换为高低4bit, 然后分别做MAC, 然后再做累加
        input_data_high, input_data_low = CIMA_8bit_to_4bit(input_data)
        
        # offset 中需要考虑 加上 (每一列的权重之和 * 8)
        # 高4bit计算
        output_quant_hight = CIMA_array_MAC(input_data_high, weight_data,  
                                        DAC_noise = DAC_noise, conductance_noise = conductance_noise, 
                                        ADC_noise = ADC_noise, ADC_offset = ADC_offset, ADC_quant_level = ADC_quant_level,
                                        scale = scale, offset = offset, scale_shift_num = scale_shift_num, 
                                        max_conductance = max_conductance, max_voltage = max_voltage, max_current = max_current)
        # 低4bit计算
        output_quant_low = CIMA_array_MAC(input_data_low, weight_data, 
                                        DAC_noise = DAC_noise, conductance_noise = conductance_noise, 
                                        ADC_noise = ADC_noise, ADC_offset = ADC_offset, ADC_quant_level = ADC_quant_level,
                                        scale = scale, offset = offset, scale_shift_num = scale_shift_num, 
                                        max_conductance = max_conductance, max_voltage = max_voltage, max_current = max_current)
        
        # 高低4bit移位累加
        output_quant = output_quant_hight * 16 + output_quant_low
        
        # 输出移位
        if accumulate_shift_num > 0:
            output_quant = output_quant >> accumulate_shift_num
        else:
            output_quant = output_quant << abs(accumulate_shift_num)
        
        # 输出截断到8bit
        output_quant = np.clip(output_quant, a_min=-128, a_max=127)
        
    else:
        raise ValueError(f'CIMA不支持的数据位宽: {dtype} !!!')
    
    return output_quant
    
def CIMA_digital_MAC(input_data, weight_data, *, dtype = '8bit', scale=1, offset=0, scale_shift_num=0, accumulate_shift_num = 0):
    assert dtype == '8bit'
    assert input_data.max() <= 127 and input_data.min() >= -128
    assert weight_data.max() <= 127 and weight_data.min() >= -128
    if len(input_data.shape) == 3:
        # 输入维度变换 B, OC, IC*H*W ===> B, IC*H*W, OC
        input_data = input_data.transpose(0, 2, 1)
    # 计算
    output_data = input_data @ weight_data
    # scale + 移位 校准 (bn)
    
    output_data = (output_data * scale).astype(np.int32)
    output_data = output_data >> scale_shift_num
    
    # offset 校准 (bn)
    output_data = output_data + offset
    output_data = output_data.astype(np.int32)
    
    # 输出移位
    if accumulate_shift_num > 0:
        output_quant = output_data >> accumulate_shift_num
    else:
        output_quant = output_data << abs(accumulate_shift_num)
    
    # 输出截断到8bit
    output_quant = np.clip(output_quant, a_min=-128, a_max=127)
    
    return output_quant

def CIMA_add(*args, dtype='4bit'):
    input_data = 0
    # 累加
    for i in args:
        input_data += i
    # 输出截断
    if dtype == '8bit':
        input_data = np.clip(input_data, a_min=-128, a_max=127)
    elif dtype == '4bit':
        input_data = np.clip(input_data, a_min=-8, a_max=7)
    else:
        raise ValueError(f'暂不支持 {dtype} !!!')
    return input_data

def CIMA_silu(x, lut, data_type='4bit'):
    if data_type == '4bit': 
        output_query = x + 8
    elif data_type == '8bit':
        output_query = x + 128
    else:
        raise ValueError(f'暂不支持 {data_type} !!!')
    output_query = output_query.astype(np.int32)
    output = lut[output_query]
    # 类型对齐为整型
    output = output.astype(np.int32) 
    return output

def CIMA_relu():
    pass

def CIMA_concat(*args, axis=1):
    data = np.concatenate(args, axis=axis)
    return data

def CIMA_mul_add(input_data, *, scale=1, scale_shift_num=0, offset=0, dtype='4bit'):
    
    output_data = input_data * scale 
    output_data = output_data >> scale_shift_num
    output_data = output_data + offset
    
    if dtype == '8bit':
        output_data = np.clip(output_data, a_min=-128, a_max=127)
    elif dtype == '4bit':
        output_data = np.clip(output_data, a_min=-8, a_max=7)
    else:
        raise ValueError(f'暂不支持 {dtype} !!!')
    return output_data

def CIMA_data_conversion():
    pass

# 将 feature_map 转化为下一层忆阻器的输入 array_input
def feature_map_to_input_np_HWC(feature_map, kernel_size, stride, padding, repeat = None, multi_batch=False):
    # feature_map shape = [W_in, H_in, C_in,]
    # array_input shape = [W_out * H_out, C_out]
    if multi_batch:
        if len(feature_map.shape) != 4:
            raise ValueError(f"暂不支持当前维度{feature_map.shape}, 默认为4维[b,c,h,w] !!!")
        # 此时输入默认为HWC的，因此需要先变为CHW，在进行后续变换
        feature_map = feature_map.transpose(0,3,1,2)
        
        batch = feature_map.shape[0]
        array_input = []
        for i in range(batch):
            temp_input = feature_map[i,:,:,:]
            temp_array_input = convert_input_HWC(temp_input, kernel_size,padding,stride)
            temp_array_input = np.expand_dims(temp_array_input,axis=0)
            array_input.append(temp_array_input)
        array_input = np.concatenate(array_input,axis=0)
        assert (len(array_input.shape) == 3) 
    else:
        # 此时输入默认为HWC的，因此需要先变为CHW，在进行后续变换
        feature_map = feature_map.transpose(2,0,1)
        array_input = convert_input_HWC(feature_map,kernel_size,padding,stride)
    if repeat:
        raise ValueError('repeat不在此处完成！！！')
    return array_input

# 转化三维的输入[in_channel,height,width] --> 二维的array_input[array_height,array_width]
def convert_input_HWC(feature_map,kernel_size,padding,stride):
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    in_channels = feature_map.shape[0]
    feature_in_w = feature_map.shape[1]
    feature_in_h = feature_map.shape[2]
    feature_out_w = int((feature_in_w - kernel_size + 2 * padding) / stride + 1)
    feature_out_h = int((feature_in_h - kernel_size + 2 * padding) / stride + 1)
    feature_map = feature_map_padding(feature_map, padding)
    input_rows = kernel_size ** 2 * in_channels
    output_rows = feature_out_w * feature_out_h
    array_input = np.zeros([input_rows, output_rows])
    idx = 0
    for i in range(feature_out_w):
        for j in range(feature_out_h):
            slide_window = feature_map[:, i * stride:i * stride + kernel_size,
                        j * stride:j * stride + kernel_size]
            # 交换axis，channel优先
            slide_window = slide_window.transpose(1,2,0)
            array_input[:, idx] = slide_window.reshape(-1)
            idx += 1
    return array_input

# 给 feature_map 加上 padding
def feature_map_padding(feature_map, padding):
    # feature_map 维度： C, W, H
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    feature_map_pad = np.pad(feature_map, ((0, 0), (padding, padding), (padding, padding)), mode = 'constant')
    return feature_map_pad

# 将忆阻器每层的输出 out_put 转换回 feature_map 的形式
def output_to_feature_map(out_put, out_h, out_w, multi_batch=False):
    # out_put shape = [W_out * H_out, C_out]
    # feature_map shape = [C_out, W_out, H_out]
    # print(out_put.shape)
    if multi_batch:
        batch = out_put.shape[0]
        channels = out_put.shape[2]
        feature_map = out_put.transpose(0, 2, 1).reshape([batch, channels, out_h, out_w])
    else:
        channels = out_put.shape[1]
        feature_map = out_put.transpose(1, 0).reshape([channels, out_h, out_w])
    return feature_map

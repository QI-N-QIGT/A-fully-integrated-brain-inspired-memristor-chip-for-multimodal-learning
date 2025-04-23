import numpy as np
from ..quant import *
# import torch
# import torch.nn as nn
from .bitwise import *
import time
import os
import concurrent.futures

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

# 给 feature_map 加上 padding
def feature_map_padding(feature_map, padding):
    # feature_map 维度： C, W, H
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    feature_map_pad = np.pad(feature_map, ((0, 0), (padding, padding), (padding, padding)), mode = 'constant')
    return feature_map_pad

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

# 将 feature_map 转化为下一层忆阻器的输入 array_input
def feature_map_to_input_np_HWC(feature_map, kernel_size, stride, padding, repeat = None, multi_batch=False):
    # feature_map shape = [W_in, H_in, C_in,]
    # array_input shape = [W_out * H_out, C_out]
    if multi_batch:
        if len(feature_map.shape) != 4:
            raise ValueError(f"暂不支持当前维度{feature_map.shape},默认为4维[b,c,h,w]！！！")
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

# 将 feature_map 转化为下一层忆阻器的输入 array_input
def feature_map_to_input_np_CWH(feature_map, kernel_size, stride, padding, repeat = None):
    # feature_map shape = [C_in, W_in, H_in]
    # array_input shape = [W_out * H_out, C_out]
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
            array_input[:, idx] = slide_window.transpose(1, 2, 0).reshape(-1)
            idx += 1
    if repeat:
        array_input = np.tile(array_input, [repeat[0], 1])
    return array_input


def feature_map_to_input(feature_map, kernel_size, stride, padding, repeat = None, multi_batch = False):
    # unfold = nn.Unfold(kernel_size, padding = padding, stride = stride)
    # feature_map = torch.from_numpy(feature_map)
    # while (len(feature_map.shape) < 4):
    #     feature_map = feature_map.unsqueeze(dim = 0)
    # array_input = unfold(feature_map.float())
    # array_input = array_input.numpy().squeeze()
    # array_input = array_input.numpy()
    # 支持 batch
    if multi_batch:
        if len(feature_map.shape) != 4:
            raise ValueError(f"暂不支持当前维度{feature_map.shape},默认为4维[b,c,h,w]！！！")
        batch = feature_map.shape[0]
        array_input = []
        if batch <= 10000:
            # 
            for i in range(batch):
                temp_input = feature_map[i,:,:,:]
                temp_array_input = image_to_col(temp_input, kernel_size, stride, padding)
                temp_array_input = np.expand_dims(temp_array_input,axis=0)
                array_input.append(temp_array_input)
        else:
            # 使用多进程优化执行
            
            # 定义不同的输入
            tasks = []
            for i in range(batch):
                tasks.append((feature_map[i,:,:,:],  kernel_size, stride, padding))
                
            with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() // 3))) as executor:
                # 使用进程池并行执行任务
                # 使用`map`方法并行执行处理函数，结果按顺序返回
                results = list(executor.map(image_to_col, *zip(*tasks)))

            # 获取结果
            for re in results:
                re = np.expand_dims(re,axis=0)
                array_input.append(re)
                   
        array_input = np.concatenate(array_input,axis=0)
        assert (len(array_input.shape) == 3)
    else:
        if len(feature_map.shape) != 3:
            raise ValueError(f"暂不支持当前维度{feature_map.shape},默认为3维[c,h,w]！！！")
        array_input = image_to_col(feature_map, kernel_size, stride, padding)
    if repeat:
        raise ValueError('repeat不在此处完成！！！')
    return array_input

def image_to_col(feature_map,kernel_size,stride,padding):
    input_channel,feature_height,feature_width = feature_map.shape
    row_length = feature_height + 2 * padding - kernel_size
    col_length = feature_width + 2 * padding - kernel_size
   
    # 默认行列方向stride一样
    stride_row = stride
    stride_col = stride
    
    # 输出阵列的规模
    matmul_length = kernel_size * kernel_size * input_channel
    out_image_height = int ((feature_height + 2 * padding - kernel_size)/ stride_row + 1)
    out_image_width = int ((feature_width + 2 * padding - kernel_size)/ stride_col + 1)
    out_num = out_image_height * out_image_width
    array_input = np.zeros((out_num,matmul_length))
    row_index = 0
    # range左闭右开
    for i in range(0,row_length+1,stride_row):
        for j in range(0,col_length+1,stride_col):
            index = 0
            for c in range(input_channel):
                for k in range(kernel_size):
                    for l in range(kernel_size):
                        if ( (i + k) < padding) or ((i + k) >= (feature_height + padding)) or ((j + l) < padding) or ((j + l) >= (feature_width + padding)):
                            array_input[row_index][index] = 0
                        else:
                            array_input[row_index][index] = feature_map[c][((i- padding) + k)][(j - padding) + l ]
                        index += 1
                        # print(array_input)
                        # input()
            row_index += 1
    array_input = array_input.transpose(1,0)
    return array_input

# 先复制权重，然后切分，全连接层
def calc_mvm(weight_addr_list, input_data, input_scale,
            repeat, out_channel, output_half_level, weight_scale, 
            dac_bits,adc_bits, adc_scale_data, reg_shift_mode, shift_expansion_mode,
            array_data=None, assigned_output_quant_scale = 1,
            output_quant_mode = 1, it_time = 10, 
            output_quant=True, bias_digital=False, bias=None,
            n_scale = None, adc_clamp = None, ADC_LUT = None, 
            adc_quant = None, macro_method=False, runtime='simulation',
            multi_batch=False, fit_k = 1, fit_bias = 0):
    
    '''
    ================================= 
    参数说明
    ================================= 
    input_feature_map:
      输入 feature map, 矩阵大小为 [C, H, W]
    weight_addr:
      144K 上复制后的权重, 矩阵大小为 [rows * repeat[0], cols * repeat[1]]
    repeat:
      144K 上复制权重的次数, repeat[0] 为行复制次数, repeat[1] 为列复制次数
    input_half_level:
      输入数据量化等级
    output_half_level:
      输出数据量化等级
    auto_ADC:
      自动配置 ADC 积分时间
    return_it_time：
      返回计算结果和最佳 ADC 积分时间
    relu:
      是否 relu
    input_quant:
      是否对输入数据进行量化
    '''
    # input_data = input_data.transpose(1,0)
    batch = 1
    
    if multi_batch:
        # 卷积 batch 的维度在 第一个
        batch = input_data.shape[0]
    else:
        input_data = np.expand_dims(input_data,axis=0)
    # print(input_data.max())
    # 输入复制
    array_input = np.tile(input_data, [1, repeat[0], 1])    

    # 计算次数
    cal_times = array_input.shape[2]
    
    array_output_repeat = np.zeros([batch, cal_times, out_channel * repeat[1]])
    array_output = np.zeros([batch, cal_times, out_channel])

    # input()

    ADC_scale = 0
    ADC_scale_count = 0
    for split_idx, split_config in enumerate(weight_addr_list):
        array_idx = split_config['array_idx']
        weight_addr = split_config['weight_addr']
        row_size = weight_addr[2]
        col_size = weight_addr[3]
        array_input_row_start = split_config['array_input_row_start']
        array_output_col_start = split_config['array_output_col_start']
        
        array_input_split = array_input[:,array_input_row_start:array_input_row_start + row_size, :]
        
        ADC_scale_count += 1
        ADC_scale += get_ADC_scale(it_time = it_time, array_idx = array_idx, k=adc_scale_data, LUT = False)
        # 乘加运算(卷积 f1), 使用脉冲展开
        # t_mvm = time.time()
        # 运行时ip
        rpc_api = None
        if runtime == "simulation":
            rpc_api = None
        elif runtime == 'c200':
            rpc_api = split_config[array_idx]
        else:
            raise ValueError(f"暂不支持runtime {runtime}, 仅支持 ('simulation', 'c200',)")
        if shift_expansion_mode == 'bit_shift':
            expand_mode = 1
            [array_output_split, max_percent, min_percent] = mvm_multi_bit_shift_batch(array_idx, array_input_split,
                                                             weight_addr,
                                                             dac_bits = dac_bits,
                                                             repeat = None,
                                                             it_time = it_time,
                                                             array_data=array_data,
                                                             adc_scale_data= adc_scale_data, adc_bits = adc_bits,
                                                             n_scale = n_scale ,adc_clamp = adc_clamp, 
                                                             ADC_LUT = ADC_LUT, adc_quant = adc_quant,
                                                             macro_method=macro_method, runtime = runtime,
                                                             rpc_api = rpc_api, fit_k = fit_k, fit_bias = fit_bias, expand_mode = expand_mode)
        elif shift_expansion_mode == 'bit_pulse':
            expand_mode = 0
            pulse_half_level = 2 ** dac_bits - 1
            [array_output_split,
             max_percent, min_percent] = mvm_multi_bit_pulse_batch(array_idx, array_input_split,
                                                                weight_addr,
                                                                pulse_half_level = pulse_half_level,
                                                                repeat = None,
                                                                it_time = it_time,array_data=array_data,
                                                                adc_scale_data= adc_scale_data, adc_bits = adc_bits,
                                                                n_scale = n_scale ,adc_clamp = adc_clamp, 
                                                                ADC_LUT = ADC_LUT, adc_quant = adc_quant,
                                                                macro_method=macro_method, runtime = runtime,
                                                                rpc_api = rpc_api, fit_k = fit_k, fit_bias = fit_bias, expand_mode = expand_mode)
        elif shift_expansion_mode == 'bit_fast':
            pulse_half_level = 2 ** dac_bits - 1
            [array_output_split,
             max_percent, min_percent] = mvm_multi_bit_fast_batch(array_idx, array_input_split,
                                                             weight_addr,
                                                             pulse_half_level = pulse_half_level,
                                                             repeat = None,
                                                             it_time = it_time,array_data=array_data,
                                                             adc_scale_data= adc_scale_data, adc_bits = adc_bits,
                                                             n_scale = n_scale ,adc_clamp = adc_clamp, 
                                                             ADC_LUT = ADC_LUT, adc_quant = adc_quant,
                                                             macro_method=macro_method, runtime = runtime,
                                                             rpc_api = rpc_api,  fit_k = fit_k, fit_bias = fit_bias)
             
        else:
          raise ValueError(f"暂不支持{shift_expansion_mode}!!!")
        
        # array_output_split = mvm_1_bit_pulse(array_idx, array_input_split, weight_addr, repeat = None,
        #                                      it_time = it_time)
        # print()
        array_output_repeat[:, :, array_output_col_start:array_output_col_start + col_size] += array_output_split
        # t_mvm = time.time() - t_mvm
        # print(f'Time for MVM = {t_mvm}')
    
    ADC_scale /= ADC_scale_count
    # print(f'it_time = {it_time}')
    # print(f'ADC_scale = {ADC_scale}')
    row_repeat = repeat[0]
    col_repeat = repeat[1]
    for i in range(col_repeat):
        array_output += array_output_repeat[:, :, i * out_channel: (i + 1) * out_channel]
    array_output /= col_repeat
    # array_output /= row_repeat
    # array_output = array_output.round()
    
    # input_scale 维度统一成3维
    if multi_batch:
        input_scale = np.squeeze(input_scale, axis=3)
        
    # if type(bias) != type(None):
    if bias_digital:
        if (bias == None).all():
          raise ValueError("Do not have bias data!!!")
        array_output = array_output / input_scale / weight_scale / ADC_scale / row_repeat
        # if bias_quant:
        #     bias, _ = data_quantization_sym(bias,
        #                                     half_level = bias_half_level,
        #                                     isint = 0,
        #                                     boundary_refine = 0)
        array_output += bias
        array_output *= input_scale * weight_scale * ADC_scale * row_repeat
    
    if output_quant:
        if output_quant_mode == 1:
            if multi_batch:
                # array_output_list = []
                # output_quant_scale = []
                # for batch_index in range(batch):
                #     array_output_, output_quant_scale_ = data_quantization_sym(array_output[batch_index,:,:],
                #                                                         half_level = output_half_level,
                #                                                         isint = 1,
                #                                                         reg_shift_mode = reg_shift_mode)
                #     array_output_list.append(np.expand_dims(array_output_,axis=0))
                #     output_quant_scale.append(output_quant_scale_)
                
                # output_quant_scale = np.array(output_quant_scale)
                # # 扩展成 3维, 与输入保持一致
                # output_quant_scale = np.expand_dims(output_quant_scale, axis=1)
                # output_quant_scale = np.expand_dims(output_quant_scale, axis=1)
                
                # # input()
                # array_output = np.concatenate(array_output_list,axis=0)    
                
                array_output, output_quant_scale = data_quantization_sym(array_output,
                                                                        half_level = output_half_level,
                                                                        isint = 1,
                                                                        reg_shift_mode = reg_shift_mode)
                
                output_quant_scale = np.array([output_quant_scale])
                # 扩展成 3维, 与输入保持一致
                output_quant_scale = np.expand_dims(output_quant_scale, axis=1)
                output_quant_scale = np.expand_dims(output_quant_scale, axis=1)
                
            else:
                array_output, output_quant_scale = data_quantization_sym(array_output,
                                                                        half_level = output_half_level,
                                                                        isint = 1,
                                                                        reg_shift_mode = reg_shift_mode)
            # print(f'Using adaptive output_quant_scale {output_quant_scale}')
        elif output_quant_mode == 2:
            # print(f'Using assigned_output_quant_scale {assigned_output_quant_scale}')
            if reg_shift_mode:
                array_output = (array_output * assigned_output_quant_scale).astype(np.int32)
            else:
                array_output = (array_output * assigned_output_quant_scale).round()
            array_output[array_output > output_half_level] = output_half_level
            array_output[array_output < -output_half_level] = -output_half_level
            output_quant_scale = assigned_output_quant_scale
        else:
            print('unknow quant mode')
            exit()
    
    if not multi_batch:
        array_output = np.squeeze(array_output)
    
    output_tot_scale = input_scale * output_quant_scale *  weight_scale * ADC_scale * row_repeat
    # print(array_output.max())
    # print(array_output.min())
    # print(input_scale)
    # print(output_quant_scale)
    # print(weight_scale)
    # print(ADC_scale)
    # print(row_repeat)
    # print(output_tot_scale)
    # input()
    return array_output, output_tot_scale, max_percent, min_percent, output_quant_scale

# return ADC quantization scale
def get_ADC_scale(it_time, array_idx = None, addr = None, k = None, LUT = False):
    '''
    参数统一从k传入，LUT决定使用什么方式读取k
    '''
    if LUT:
        # ADC_scale = np.load(f'ADC_scale_8_array_{array_idx}.npy')
        # ADC_scale = ADC_scale[it_time - 1, addr[1]:addr[1] + addr[3]]
        ADC_scale = k[it_time - 1, addr[1]:addr[1] + addr[3]]
    else:
        ADC_scale = k[array_idx] * it_time
    # print(f'Scale = {ADC_scale}')
    return ADC_scale

def get_weight(array_idx, addr, array_data=None):
    # dir = f'Arrays'
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # array_name = fr'Array_{array_idx}.npy'
    # array_dir = f'{dir}/{array_name}'
    # array = np.load(array_dir)
    if array_data == None:
      raise ValueError("Do not have array data!!!")
    array_data_ = array_data[array_idx]
    x = addr[0]
    y = addr[1]
    h = addr[2]
    w = addr[3]
    # weight = array_data_[x:x + h, y:y + w] - 8
    weight = array_data_[x:x + h, y:y + w]
    return weight

# def set_weight(array_idx, addr, weight_mapping):
#     dir = f'Arrays'
#     array_name = fr'Array_{array_idx}.npy'
#     array_dir = f'{dir}/{array_name}'
#     if os.path.exists(array_dir):
#         array = np.load(array_dir)
#     else:
#         array = np.zeros([cfg.array_size[0], cfg.array_size[1]])
#     x = addr[0]
#     y = addr[1]
#     h = addr[2]
#     w = addr[3]
#     array[x:x + h, y:y + w] = weight_mapping
#     np.save(array_dir, array)

def _macro_matmul(input_d,weight_d,it_time,
                  HalfMaxConductance = 20.43650794,
                  RelativeWeightError = 0.0, # default: add noise before calculation for saving time
                  DACStep = 0.1,
                  DACNoise = 0,
                  ADCNoise = 0,
                  QuanVoltageLSB = 0.3125,
                  ADCOffset = 0,
                  ):
    # 获取权重 
    weight_data = weight_d
    h,w= weight_data.shape

    conductance_data = weight_data
    # 权重加噪
    if RelativeWeightError != 0:
        conductance_data = conductance_data + np.random.randn(h,w) * HalfMaxConductance * RelativeWeightError 
    # 计算电流
    if DACNoise != 0:
        DACStep = DACStep + np.random.randn(1) * DACNoise
    OutCurrent = input_d @ conductance_data * DACStep
    h_,w_ = OutCurrent.shape
    # 电流加噪
    if ADCNoise != 0:
        OutCurrent = OutCurrent + np.random.randn(h_,w_) * ADCNoise
    # 电流积分转电压
    OutVoltage = OutCurrent * 10 **(-6) * it_time * 100 * 10**(-9)  / (5.15 * 10**(-12))
    # fit
    row_num = np.sum(input_d)
    alpha = 0.0002831 * row_num + 0.00668
    OutVoltage = OutVoltage - it_time * np.log(row_num) * alpha 
    if ADCOffset != 0:
        mask = np.random.randint(-1,2,(h_,w_))
        # 电压转数字码
        OutDigitalNum = np.round((np.floor(OutVoltage / QuanVoltageLSB)) + ADCOffset * mask)
    else:
        # 电压转数字码
        OutDigitalNum = np.round((np.floor(OutVoltage / QuanVoltageLSB))) 
    OutDigitalNum = np.clip(OutDigitalNum, -8, 7)
    return OutDigitalNum

# simulate the mvm process in C200 SDK
def sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data = None,
                      adc_scale_data=None,
                      it_time = 5,
                      n_scale = None ,
                      adc_bits = None ,
                      adc_clamp = None,
                      ADC_LUT = None,
                      adc_quant = None,
                      macro_method=False,
                      HalfMaxConductance = 20.43650794,
                      RelativeWeightError = 0.0625,
                      runtime = 'simulation',
                      rpc_api = None,
                      fit_k = 1, 
                      fit_bias = 0,
                      expand_mode = 1):

    it_time = round(it_time)
    
    cal_times = input_expanded.shape[1]
    output_cols = addr[3]
    ADC_output = np.zeros((cal_times, output_cols))
    
    if runtime == 'simulation':
        weight = get_weight(array_idx, addr, array_data=array_data)
        if macro_method:
            h,w = weight.shape
            weight_n = weight + np.random.randn(h,w) * HalfMaxConductance * RelativeWeightError 
        else:
            weight_n = add_noise(weight, n_scale = n_scale)

        ADC_scale = get_ADC_scale(it_time = it_time,
                                array_idx = array_idx,
                                addr = addr,
                                k=adc_scale_data,
                                LUT = ADC_LUT)

        # ADC_half_level = 2 ** adc_bits // 2 - 1
        ADC_half_level =  2 ** adc_bits - 1
        # former version
        '''
        for i in range(cal_times):
            if macro_method:
                input_ = input_expanded[:, i].reshape(1, -1)
                ADC_output[i,:] = _macro_matmul(input_,weight_n,it_time)
            else:
                temp = (input_expanded[:, i].reshape(-1, 1) * weight_n).sum(axis = 0) * ADC_scale
                if adc_quant:
                    temp = temp.round()
                if adc_clamp:
                    temp[temp > ADC_half_level] = ADC_half_level
                    temp[temp < -ADC_half_level - 1] = -ADC_half_level - 1
                ADC_output[i, :] = temp.reshape(1, -1)
        '''
        # new version
        if macro_method:
            input_ = input_expanded.transpose(1, 0)
            ADC_output = _macro_matmul(input_,weight_n,it_time)
        else:
            # time1 = time.time()
            # print(f'输入矩阵维度 : {input_expanded.shape}')
            # print(f'权重维度 : {weight_n.shape}')
            
            input_expanded = input_expanded.transpose(1, 0)
            temp = input_expanded @ weight_n * ADC_scale
            # time2 = time.time()
            # print(f'MVM time: {time2- time1} s')
            # input()
            if adc_quant:
                temp = temp.round()
            if adc_clamp:
                temp[temp > ADC_half_level] = ADC_half_level
                temp[temp < (-ADC_half_level - 1)] = -ADC_half_level - 1
            
            ADC_output = temp
            
    elif runtime == 'c200':
        array_idx_ = int(array_idx.split(":")[-1])
        input_expanded = input_expanded.transpose(1, 0)
        input_expanded_ = input_expanded.tolist()
        ADC_output_ = rpc_api.call('c200_calc',input_expanded_, array_idx_, addr, it_time, expand_mode)
        ADC_output = None
        if str(type(ADC_output_)) == "<class 'list'>":
            ADC_output = np.array(ADC_output_)
        elif str(type(ADC_output_)) == "<class 'numpy.ndarray'>":
            ADC_output = ADC_output_
        else:
            raise ValueError(f"暂不支持 输出返回类型 {type(ADC_output_)}")
        # input()
        
    else:
        raise ValueError(f"不支持的runtime : {runtime}, 仅支持('simulation','c200',)")
    
    # fit value 
    ADC_output = fit_k * ADC_output + fit_bias
    
    return ADC_output
    
# 使用cpu进行unfold后的特征图计算，无需bitwise展开
def mvm_cpu(input_data, array_idx, addr, array_data=None):
    weight = get_weight(array_idx, addr, array_data=array_data)
    cal_times = input_data.shape[1]
    output_cols = addr[3]
    mvm_output = np.zeros([cal_times, output_cols])
    for i in range(cal_times):
        temp = (input_data[:, i].reshape(-1, 1) * weight).sum(axis = 0)
        mvm_output[i, :] = temp.reshape(1, -1)
    return mvm_output

# Add noise to input data
def add_noise(weight, n_scale = 0.05):
    # w -> input data, usually a w
    # n_scale -> noise factor

    w_range = weight.max() - weight.min()
    shape = weight.shape
    w_noise = w_range * n_scale * np.random.randn(*shape)
    # print(f'noise_mean = {w_noise.mean():.5f}')
    weight_noise = weight + w_noise

    return weight_noise

def mvm_multi_bit_shift(array_idx, input_matrix, addr,
                        dac_bits = 1,
                        repeat = None, it_time = 5, verbose = 0,
                        assign_pulses = None, original_weight = None,
                        device = 'cpu', array_data=None, adc_scale_data=None,
                        adc_bits=None,n_scale=None,adc_clamp=None,
                        ADC_LUT=None,adc_quant=None,macro_method=False,
                        runtime = 'simulation', rpc_api = None
                        ):
    # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    cal_times = input_matrix.shape[1]
    # 输出通道数
    output_cols = addr[3]
    # 创建一个全零的输出矩阵
    output = np.zeros([cal_times, output_cols])

    # 对输入的二维矩阵 input 做 bitwise 展开, 默认情况下返回一个稠密矩阵
    #   input_expanded 是一个只有 +1,0，-1的矩阵
    #   bitlen_map 中记录了 input 中每一列展开的最大 bit 位数,
    #   在 144k 计算完毕后会根据 bitlen_map 中记录的 bit 位数做对应行的累加
    [input_expanded,
     bitlen_map] = input_multi_bits_shift_expansion(input_matrix,
                                                    dac_bits = dac_bits)

    output_bitwise = sdk_cal_sim_noise(input_expanded, array_idx, addr, it_time = it_time, array_data= array_data,
                                       adc_scale_data=adc_scale_data,
                                       n_scale = n_scale ,
                                      adc_bits = adc_bits ,
                                      adc_clamp = adc_clamp,
                                      ADC_LUT = ADC_LUT,
                                      adc_quant = adc_quant,
                                      macro_method=macro_method,
                                      runtime = runtime,
                                      rpc_api = rpc_api)

    output_bitwise_row = 0
    output_row = 0

    # 对计算结果按照展开的位数进行求和
    for j in bitlen_map:
        if j == 0:
            output[output_row, :] = 0
        else:
            factor_list = np.array([2 ** (i * dac_bits) for i in range(j)])
            factor_list = factor_list.reshape(j, -1)
            output_temp = output_bitwise[
                          output_bitwise_row:
                          output_bitwise_row + j
                          ] * factor_list
            output[output_row, :] = output_temp.sum(axis = 0)

        output_row += 1
        output_bitwise_row += j

    # ==================================================== #
    max_percent, min_percent = count_overshoot_percent(output_bitwise, adc_bits = adc_bits)
    # ==================================================== #

    # 如果权重复制了, 求出 output 的平均值
    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([cal_times, output_avg_cols])
        for i in range(col_repeat):
            output_avg += output[:, i * output_avg_cols: (i + 1) * output_avg_cols]
        output_avg /= col_repeat
        output_avg /= row_repeat
        return output_avg.round()
    return output, max_percent, min_percent


def mvm_multi_bit_pulse(array_idx, input_matrix, addr,
                        pulse_half_level = 7,
                        repeat = None, it_time = 5, verbose = 0,
                        assign_pulses = None, original_weight = None,
                        array_data= None, adc_scale_data=None,
                        adc_bits=None, n_scale = None ,adc_clamp = None, 
                        ADC_LUT = None, adc_quant = None, macro_method=False,
                        runtime = 'simulation', rpc_api = None):
    # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    cal_times = input_matrix.shape[1]
    
    # 输出通道数
    output_cols = addr[3]
    # 创建一个全零的输出矩阵
    output = np.zeros([cal_times, output_cols])

    # 对输入的二维矩阵 input 做 bitwise 展开, 默认情况下返回一个稠密矩阵
    #   input_expanded 是一个只有 +1,0，-1的矩阵
    #   bitlen_map 中记录了 input 中每一列展开的最大 bit 位数,
    #   在 144k 计算完毕后会根据 bitlen_map 中记录的 bit 位数做对应行的累加
    input_expanded, bitlen_map = input_multi_bits_pulse_expansion(input_matrix,
                                                                  pulse_half_level = pulse_half_level,
                                                                  assign_pulses = assign_pulses)

    output_bitwise = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data = array_data, 
                                       adc_scale_data=adc_scale_data, it_time = it_time,
                                       n_scale = n_scale ,
                                      adc_bits = adc_bits ,
                                      adc_clamp = adc_clamp,
                                      ADC_LUT = ADC_LUT,
                                      adc_quant = adc_quant,
                                      macro_method=macro_method,
                                      runtime = runtime,
                                      rpc_api = rpc_api,
                                      )
    
    output_bitwise_row = 0
    output_row = 0

    # 对计算结果按照展开的位数进行求和
    for j in bitlen_map:
        if j == 0:
            output[output_row, :] = 0
        else:
            output[output_row, :] = output_bitwise[output_bitwise_row:output_bitwise_row + j].sum(axis = 0)
        output_row += 1
        output_bitwise_row += j

    # ==================================================== #
    max_percent, min_percent = count_overshoot_percent(output_bitwise, adc_bits = adc_bits)
    # ==================================================== #

    # 如果权重复制了, 求出 output 的平均值
    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([cal_times, output_avg_cols])
        for i in range(col_repeat):
            output_avg += output[:, i * output_avg_cols: (i + 1) * output_avg_cols]
        output_avg /= col_repeat
        output_avg /= row_repeat
        return output_avg.round()
    return output, max_percent, min_percent

def mvm_multi_bit_shift_batch(array_idx, input_matrix, addr,
                        dac_bits = 1,
                        repeat = None, it_time = 5, verbose = 0,
                        assign_pulses = None, original_weight = None,
                        device = 'cpu', array_data=None, adc_scale_data=None,
                        adc_bits=None,n_scale=None,adc_clamp=None,
                        ADC_LUT=None,adc_quant=None,macro_method=False,
                        runtime = 'simulation', rpc_api = None,  fit_k = 1, 
                        fit_bias = 0, expand_mode =1,
                        ):
    # # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    # cal_times = input_matrix.shape[1]
    # # 输出通道数
    # output_cols = addr[3]
    # # batch
    # batch = input_matrix.shape[0]
    # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    cal_times = input_matrix.shape[2]
    # batch
    batch = input_matrix.shape[0]
    # 输出通道数
    output_cols = addr[3]
    # 创建一个全零的输出矩阵
    output = np.zeros([batch, cal_times, output_cols])

    # 对输入的二维矩阵 input 做 bitwise 展开, 默认情况下返回一个稠密矩阵
    #   input_expanded 是一个只有 +1,0，-1的矩阵
    #   bitlen_map 中记录了 input 中每一列展开的最大 bit 位数,
    #   在 144k 计算完毕后会根据 bitlen_map 中记录的 bit 位数做对应行的累加
    input_expanded = []
    bitlen_map = []
    batch_expansion_len_list = []
    # print(f'expansion begin:')
    # time1 = time.time()
    for i in range(batch):
        input_expanded_, bitlen_map_ = input_multi_bits_shift_expansion(input_matrix[i,:,:],
                                                                    dac_bits = dac_bits)

        input_expanded.append(input_expanded_)
        bitlen_map.append(bitlen_map_)
        # print(bitlen_map)
        # input()
        batch_expansion_len_list.append(input_expanded_.shape[1])
    # time2 = time.time()
    # print(f'expansion time : {time2- time1} s')
    
    # [input_expanded,
    #  bitlen_map] = input_multi_bits_shift_expansion(input_matrix,
    #                                                 dac_bits = dac_bits)

    input_expanded = np.concatenate(input_expanded, axis=1)
    
    output_bitwise_ = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data = array_data, 
                                       adc_scale_data=adc_scale_data, it_time = it_time,
                                       n_scale = n_scale ,
                                      adc_bits = adc_bits ,
                                      adc_clamp = adc_clamp,
                                      ADC_LUT = ADC_LUT,
                                      adc_quant = adc_quant,
                                      macro_method=macro_method,
                                      runtime = runtime,
                                      rpc_api = rpc_api,
                                      fit_k = fit_k, 
                                      fit_bias = fit_bias
                                      )

    start_batch_index = 0
    max_percent = []
    min_percent = []

    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([batch, cal_times, output_avg_cols])
    
    for batch_index in range(batch):

        batch_expansion_len = batch_expansion_len_list[batch_index]
        
        output_bitwise = output_bitwise_[start_batch_index:(start_batch_index+batch_expansion_len ),:]
        
        output_bitwise_row = 0
        output_row = 0

        # 对计算结果按照展开的位数进行求和
        for j in bitlen_map[batch_index]:
            
            if j == 0:
                output[batch_index, output_row, :] = 0
            else:
                factor_list = np.array([2 ** (i * dac_bits) for i in range(j)])
                factor_list = factor_list.reshape(j, -1)
                output_temp = output_bitwise[
                            output_bitwise_row:
                            output_bitwise_row + j
                            ] * factor_list
                # print(output_temp.shape)
                # print(output.shape)
                # input()
                output[batch_index, output_row, :] = output_temp.sum(axis = 0)

            output_row += 1
            output_bitwise_row += j
        
        # # 对计算结果按照展开的位数进行求和
        # for j in bitlen_map[batch_index]:
        #     if j == 0:
        #         output[batch_index, output_row, :] = 0
        #     else:
        #         output[batch_index, output_row, :] = output_bitwise[output_bitwise_row:output_bitwise_row + j].sum(axis = 0)
        #     output_row += 1
        #     output_bitwise_row += j

        # ==================================================== #
        max_percent_, min_percent_ = count_overshoot_percent(output_bitwise, adc_bits = adc_bits)
        # ==================================================== #
        max_percent.append(max_percent_)
        min_percent.append(min_percent_)
        
        # 如果权重复制了, 求出 output 的平均值
        if repeat:
            for i in range(col_repeat):
                output_avg[batch_index,:,:] += output[batch_index, :, i * output_avg_cols: (i + 1) * output_avg_cols]
            output_avg[batch_index,:,:] /= col_repeat
            output_avg[batch_index,:,:] /= row_repeat
            
        start_batch_index += batch_expansion_len

    if repeat:
        output = output_avg
        
    return output, max_percent, min_percent


def mvm_multi_bit_pulse_batch(array_idx, input_matrix, addr,
                        pulse_half_level = 7,
                        repeat = None, it_time = 5,
                        array_data= None, adc_scale_data=None,
                        adc_bits=None, n_scale = None ,adc_clamp = None, 
                        ADC_LUT = None, adc_quant = None, macro_method=False,
                        runtime = 'simulation', rpc_api = None, fit_k = 1,
                        fit_bias = 0, expand_mode=0):
    # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    cal_times = input_matrix.shape[2]
    # batch
    batch = input_matrix.shape[0]
    # 输出通道数
    output_cols = addr[3]
    # 创建一个全零的输出矩阵
    output = np.zeros([batch, cal_times, output_cols])

    # 对输入的二维矩阵 input 做 bitwise 展开, 默认情况下返回一个稠密矩阵
    #   input_expanded 是一个只有 +1,0，-1的矩阵
    #   bitlen_map 中记录了 input 中每一列展开的最大 bit 位数,
    #   在 144k 计算完毕后会根据 bitlen_map 中记录的 bit 位数做对应行的累加
    
    # print(f'expansion begin:')
    # time1 = time.time()
    
    input_expanded = []
    bitlen_map = []
    batch_expansion_len_list = []
    if batch <= 10000:
        for i in range(batch):
            # input_expanded_, bitlen_map_ = input_multi_bits_pulse_expansion_GPT(input_matrix[i,:,:],
            #                                                             pulse_half_level = pulse_half_level,
            #                                                             )
            input_expanded_, bitlen_map_ = input_multi_bits_pulse_expansion(input_matrix[i,:,:],
                                                                        pulse_half_level = pulse_half_level,
                                                                        )
            input_expanded.append(input_expanded_)
            bitlen_map.append(bitlen_map_)
            batch_expansion_len_list.append(input_expanded_.shape[1])
    else:
        # 使用多进程优化执行
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() // 3))) as executor:
            # 使用进程池并行执行任务
            results = list(executor.map(input_multi_bits_pulse_expansion, input_matrix.tolist(), [pulse_half_level]*batch))
        
        input_expanded_, bitlen_map_ = zip(*results)
        
        for i in range(len(input_expanded_)):
            input_expanded.append(input_expanded_[i])
            bitlen_map.append(bitlen_map_[i])
            batch_expansion_len_list.append(input_expanded_[i].shape[1])
        
    input_expanded = np.concatenate(input_expanded, axis=1)
    
    # batch 展开
    # input_expanded, bitlen_map, batch_expansion_len_list  = input_multi_bits_pulse_expansion_batch(input_matrix,
    #                                                                 pulse_half_level = pulse_half_level,
    #                                                                 )
    # time2 = time.time()
    # print(f'Expansion time : {time2- time1} s')
    
    output_bitwise_ = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data = array_data, 
                                       adc_scale_data=adc_scale_data, it_time = it_time,
                                       n_scale = n_scale ,
                                      adc_bits = adc_bits ,
                                      adc_clamp = adc_clamp,
                                      ADC_LUT = ADC_LUT,
                                      adc_quant = adc_quant,
                                      macro_method=macro_method,
                                      runtime = runtime,
                                      rpc_api = rpc_api,
                                      fit_k = fit_k,
                                      fit_bias = fit_bias
                                      )

    start_batch_index = 0
    max_percent = []
    min_percent = []

    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([batch, cal_times, output_avg_cols])
    
    # time1 = time.time()
    
    for batch_index in range(batch):
    
        batch_expansion_len = batch_expansion_len_list[batch_index]
        
        output_bitwise = output_bitwise_[start_batch_index:(start_batch_index+batch_expansion_len ),:]
        
        output_bitwise_row = 0
        output_row = 0

        # 对计算结果按照展开的位数进行求和
        for j in bitlen_map[batch_index]:
            if j == 0:
                output[batch_index, output_row, :] = 0
            else:
                output[batch_index, output_row, :] = output_bitwise[output_bitwise_row:output_bitwise_row + j].sum(axis = 0)
            output_row += 1
            output_bitwise_row += j

        # ==================================================== #
        max_percent_, min_percent_ = count_overshoot_percent(output_bitwise, adc_bits = adc_bits)
        # ==================================================== #
        max_percent.append(max_percent_)
        min_percent.append(min_percent_)
        
        # 如果权重复制了, 求出 output 的平均值
        if repeat:
            
            for i in range(col_repeat):
                output_avg[batch_index,:,:] += output[batch_index, :, i * output_avg_cols: (i + 1) * output_avg_cols]
            output_avg[batch_index,:,:] /= col_repeat
            output_avg[batch_index,:,:] /= row_repeat
            
        start_batch_index += batch_expansion_len

    # time2 = time.time()
    # print(f'Recover Output time : {time2- time1} s')
    
    if repeat:
        output = output_avg
    
    return output, max_percent, min_percent


def mvm_multi_bit_fast_batch(array_idx, input_matrix, addr,
                        pulse_half_level = 7,
                        repeat = None, it_time = 5,
                        array_data= None, adc_scale_data=None,
                        adc_bits=None, n_scale = None ,adc_clamp = None, 
                        ADC_LUT = None, adc_quant = None, macro_method=False,
                        runtime = 'simulation', rpc_api = None, fit_k = 1,
                        fit_bias = 0):
    # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    cal_times = input_matrix.shape[2]
    # batch
    batch = input_matrix.shape[0]
    # 输出通道数
    output_cols = addr[3]
    # # 创建一个全零的输出矩阵
    # output = np.zeros([batch, cal_times, output_cols])
    
    # 对输入的二维矩阵 input / pusle_half_level , 默认情况下返回一个稠密矩阵
    scale = np.max(abs(input_matrix))
    
    input_matrix = input_matrix / (scale + 10**(-6))
    # print(input_matrix.shape)
    # input()
    input_expanded = input_matrix.transpose(0,2,1).reshape(batch * cal_times, -1)
    
    # 保持与之前接口一致，阵列行数在第一维
    input_expanded = input_expanded.transpose(1,0)
    
    output_bitwise_ = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data = array_data, 
                                       adc_scale_data=adc_scale_data, it_time = it_time,
                                       n_scale = n_scale ,
                                      adc_bits = adc_bits ,
                                      adc_clamp = adc_clamp,
                                      ADC_LUT = ADC_LUT,
                                      adc_quant = adc_quant,
                                      macro_method=macro_method,
                                      runtime = runtime,
                                      rpc_api = rpc_api,
                                      fit_k = fit_k,
                                      fit_bias= fit_bias
                                      )
    
    output_bitwise_ = output_bitwise_.reshape(batch, cal_times, output_cols)
    
    output = output_bitwise_
    # start_batch_index = 0
    max_percent = []
    min_percent = []

    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([batch, cal_times, output_avg_cols])
    
    for batch_index in range(batch):
    
        # batch_expansion_len = batch_expansion_len_list[batch_index]
        
        # output_bitwise = output_bitwise_[start_batch_index:(start_batch_index+batch_expansion_len ),:]
        
        # output_bitwise_row = 0
        # output_row = 0

        # # 对计算结果按照展开的位数进行求和
        # for j in bitlen_map[batch_index]:
        #     if j == 0:
        #         output[batch_index, output_row, :] = 0
        #     else:
        #         output[batch_index, output_row, :] = output_bitwise[output_bitwise_row:output_bitwise_row + j].sum(axis = 0)
        #     output_row += 1
        #     output_bitwise_row += j
        
        output_bitwise = output[batch_index, :, :]
        
        # ==================================================== #
        max_percent_, min_percent_ = count_overshoot_percent(output_bitwise, adc_bits = adc_bits)
        # ==================================================== #
        max_percent.append(max_percent_)
        min_percent.append(min_percent_)
        
        # 如果权重复制了, 求出 output 的平均值
        if repeat:
            
            for i in range(col_repeat):
                output_avg[batch_index,:,:] += output[batch_index, :, i * output_avg_cols: (i + 1) * output_avg_cols]
            output_avg[batch_index,:,:] /= col_repeat
            output_avg[batch_index,:,:] /= row_repeat
            
        # start_batch_index += batch_expansion_len

    if repeat:
        output = output_avg
    
    # 还原 output
    output = output * scale
    
    return output, max_percent, min_percent
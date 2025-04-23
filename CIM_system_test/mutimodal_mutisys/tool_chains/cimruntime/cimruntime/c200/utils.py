import numpy as np
from ..quant import *
from .bitwise import *
import time
import os
import concurrent.futures

def output_to_feature_map(out_put, out_h, out_w, multi_batch=False):
    if multi_batch:
        batch = out_put.shape[0]
        channels = out_put.shape[2]
        feature_map = out_put.transpose(0, 2, 1).reshape([batch, channels, out_h, out_w])
    else:
        channels = out_put.shape[1]
        feature_map = out_put.transpose(1, 0).reshape([channels, out_h, out_w])
    return feature_map

def feature_map_padding(feature_map, padding):
    while len(feature_map.shape) < 3:
        feature_map = np.expand_dims(feature_map, axis=0)
    feature_map_pad = np.pad(feature_map, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    return feature_map_pad

def convert_input_HWC(feature_map, kernel_size, padding, stride):
    while len(feature_map.shape) < 3:
        feature_map = np.expand_dims(feature_map, axis=0)
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
            slide_window = feature_map[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
            slide_window = slide_window.transpose(1, 2, 0)
            array_input[:, idx] = slide_window.reshape(-1)
            idx += 1
    return array_input

def feature_map_to_input_np_HWC(feature_map, kernel_size, stride, padding, repeat=None, multi_batch=False):
    if multi_batch:
        if len(feature_map.shape) != 4:
            raise ValueError(f'暂不支持当前维度{feature_map.shape},默认为4维[b,c,h,w]！！！')
        feature_map = feature_map.transpose(0, 3, 1, 2)
        batch = feature_map.shape[0]
        array_input = []
        for i in range(batch):
            temp_input = feature_map[i, :, :, :]
            temp_array_input = convert_input_HWC(temp_input, kernel_size, padding, stride)
            temp_array_input = np.expand_dims(temp_array_input, axis=0)
            array_input.append(temp_array_input)
        array_input = np.concatenate(array_input, axis=0)
        assert len(array_input.shape) == 3
    else:
        feature_map = feature_map.transpose(2, 0, 1)
        array_input = convert_input_HWC(feature_map, kernel_size, padding, stride)
    if repeat:
        raise ValueError('repeat不在此处完成！！！')
    return array_input

def feature_map_to_input_np_CWH(feature_map, kernel_size, stride, padding, repeat=None):
    while len(feature_map.shape) < 3:
        feature_map = np.expand_dims(feature_map, axis=0)
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
            slide_window = feature_map[:, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
            array_input[:, idx] = slide_window.transpose(1, 2, 0).reshape(-1)
            idx += 1
    if repeat:
        array_input = np.tile(array_input, [repeat[0], 1])
    return array_input

def feature_map_to_input(feature_map, kernel_size, stride, padding, repeat=None, multi_batch=False):
    if multi_batch:
        if len(feature_map.shape) != 4:
            raise ValueError(f'暂不支持当前维度{feature_map.shape},默认为4维[b,c,h,w]！！！')
        batch = feature_map.shape[0]
        array_input = []
        if batch <= 10000:
            for i in range(batch):
                temp_input = feature_map[i, :, :, :]
                temp_array_input = image_to_col(temp_input, kernel_size, stride, padding)
                temp_array_input = np.expand_dims(temp_array_input, axis=0)
                array_input.append(temp_array_input)
        else:
            tasks = []
            for i in range(batch):
                tasks.append((feature_map[i, :, :, :], kernel_size, stride, padding))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() // 3))) as executor:
                results = list(executor.map(image_to_col, *zip(*tasks)))
            for re in results:
                re = np.expand_dims(re, axis=0)
                array_input.append(re)
        array_input = np.concatenate(array_input, axis=0)
        assert len(array_input.shape) == 3
    else:
        if len(feature_map.shape) != 3:
            raise ValueError(f'暂不支持当前维度{feature_map.shape},默认为3维[c,h,w]！！！')
        array_input = image_to_col(feature_map, kernel_size, stride, padding)
    if repeat:
        raise ValueError('repeat不在此处完成！！！')
    return array_input

def image_to_col(feature_map, kernel_size, stride, padding):
    (input_channel, feature_height, feature_width) = feature_map.shape
    row_length = feature_height + 2 * padding - kernel_size
    col_length = feature_width + 2 * padding - kernel_size
    stride_row = stride
    stride_col = stride
    matmul_length = kernel_size * kernel_size * input_channel
    out_image_height = int((feature_height + 2 * padding - kernel_size) / stride_row + 1)
    out_image_width = int((feature_width + 2 * padding - kernel_size) / stride_col + 1)
    out_num = out_image_height * out_image_width
    array_input = np.zeros((out_num, matmul_length))
    row_index = 0
    for i in range(0, row_length + 1, stride_row):
        for j in range(0, col_length + 1, stride_col):
            index = 0
            for c in range(input_channel):
                for k in range(kernel_size):
                    for l in range(kernel_size):
                        if i + k < padding or i + k >= feature_height + padding or j + l < padding or (j + l >= feature_width + padding):
                            array_input[row_index][index] = 0
                        else:
                            array_input[row_index][index] = feature_map[c][i - padding + k][j - padding + l]
                        index += 1
            row_index += 1
    array_input = array_input.transpose(1, 0)
    return array_input

def calc_mvm(weight_addr_list, input_data, input_scale, repeat, out_channel, output_half_level, weight_scale, dac_bits, adc_bits, adc_scale_data, reg_shift_mode, shift_expansion_mode, array_data=None, assigned_output_quant_scale=1, output_quant_mode=1, it_time=10, output_quant=True, bias_digital=False, bias=None, n_scale=None, adc_clamp=None, ADC_LUT=None, adc_quant=None, macro_method=False, runtime='simulation', multi_batch=False, fit_k=1, fit_bias=0):
    
    batch = 1
    if multi_batch:
        batch = input_data.shape[0]
    else:
        input_data = np.expand_dims(input_data, axis=0)
    array_input = np.tile(input_data, [1, repeat[0], 1])
    cal_times = array_input.shape[2]
    array_output_repeat = np.zeros([batch, cal_times, out_channel * repeat[1]])
    array_output = np.zeros([batch, cal_times, out_channel])
    ADC_scale = 0
    ADC_scale_count = 0
    for (split_idx, split_config) in enumerate(weight_addr_list):
        array_idx = split_config['array_idx']
        weight_addr = split_config['weight_addr']
        row_size = weight_addr[2]
        col_size = weight_addr[3]
        array_input_row_start = split_config['array_input_row_start']
        array_output_col_start = split_config['array_output_col_start']
        array_input_split = array_input[:, array_input_row_start:array_input_row_start + row_size, :]
        ADC_scale_count += 1
        ADC_scale += get_ADC_scale(it_time=it_time, array_idx=array_idx, k=adc_scale_data, LUT=False)
        rpc_api = None
        if runtime == 'simulation':
            rpc_api = None
        elif runtime == 'c200':
            rpc_api = split_config[array_idx]
        else:
            raise ValueError(f"暂不支持runtime {runtime}, 仅支持 ('simulation', 'c200',)")
        if shift_expansion_mode == 'bit_shift':
            [array_output_split, max_percent, min_percent] = mvm_multi_bit_shift_batch(array_idx, array_input_split, weight_addr, dac_bits=dac_bits, repeat=None, it_time=it_time, array_data=array_data, adc_scale_data=adc_scale_data, adc_bits=adc_bits, n_scale=n_scale, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api, fit_k=fit_k, fit_bias=fit_bias)
        elif shift_expansion_mode == 'bit_pulse':
            pulse_half_level = 2 ** dac_bits - 1
            [array_output_split, max_percent, min_percent] = mvm_multi_bit_pulse_batch(array_idx, array_input_split, weight_addr, pulse_half_level=pulse_half_level, repeat=None, it_time=it_time, array_data=array_data, adc_scale_data=adc_scale_data, adc_bits=adc_bits, n_scale=n_scale, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api, fit_k=fit_k, fit_bias=fit_bias)
        elif shift_expansion_mode == 'bit_fast':
            pulse_half_level = 2 ** dac_bits - 1
            [array_output_split, max_percent, min_percent] = mvm_multi_bit_fast_batch(array_idx, array_input_split, weight_addr, pulse_half_level=pulse_half_level, repeat=None, it_time=it_time, array_data=array_data, adc_scale_data=adc_scale_data, adc_bits=adc_bits, n_scale=n_scale, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api, fit_k=fit_k, fit_bias=fit_bias)
        else:
            raise ValueError(f'暂不支持{shift_expansion_mode}!!!')
        array_output_repeat[:, :, array_output_col_start:array_output_col_start + col_size] += array_output_split
    ADC_scale /= ADC_scale_count
    row_repeat = repeat[0]
    col_repeat = repeat[1]
    for i in range(col_repeat):
        array_output += array_output_repeat[:, :, i * out_channel:(i + 1) * out_channel]
    array_output /= col_repeat
    if multi_batch:
        input_scale = np.squeeze(input_scale, axis=3)
    if bias_digital:
        if (bias == None).all():
            raise ValueError('Do not have bias data!!!')
        array_output = array_output / input_scale / weight_scale / ADC_scale / row_repeat
        array_output += bias
        array_output *= input_scale * weight_scale * ADC_scale * row_repeat
    if output_quant:
        if output_quant_mode == 1:
            if multi_batch:
                array_output_list = []
                output_quant_scale = []
                for batch_index in range(batch):
                    (array_output_, output_quant_scale_) = data_quantization_sym(array_output[batch_index, :, :], half_level=output_half_level, isint=1, reg_shift_mode=reg_shift_mode)
                    array_output_list.append(np.expand_dims(array_output_, axis=0))
                    output_quant_scale.append(output_quant_scale_)
                output_quant_scale = np.array(output_quant_scale)
                output_quant_scale = np.expand_dims(output_quant_scale, axis=1)
                output_quant_scale = np.expand_dims(output_quant_scale, axis=1)
                array_output = np.concatenate(array_output_list, axis=0)
            else:
                (array_output, output_quant_scale) = data_quantization_sym(array_output, half_level=output_half_level, isint=1, reg_shift_mode=reg_shift_mode)
        elif output_quant_mode == 2:
            if reg_shift_mode:
                array_output = (array_output * assigned_output_quant_scale).astype(np.int32)
            else:
                array_output = (array_output * assigned_output_quant_scale).round()
            output_quant_scale = assigned_output_quant_scale
        else:
            print()
            exit()
    if not multi_batch:
        array_output = np.squeeze(array_output)
    output_tot_scale = input_scale * output_quant_scale * weight_scale * ADC_scale * row_repeat
    return (array_output, output_tot_scale, max_percent, min_percent, output_quant_scale)

def get_ADC_scale(it_time, array_idx=None, addr=None, k=None, LUT=False):
    
    if LUT:
        ADC_scale = k[it_time - 1, addr[1]:addr[1] + addr[3]]
    else:
        ADC_scale = k[array_idx] * it_time
    return ADC_scale

def get_weight(array_idx, addr, array_data=None):
    if array_data == None:
        raise ValueError('Do not have array data!!!')
    array_data_ = array_data[array_idx]
    x = addr[0]
    y = addr[1]
    h = addr[2]
    w = addr[3]
    weight = array_data_[x:x + h, y:y + w]
    return weight

def _macro_matmul(input_d, weight_d, it_time, HalfMaxConductance=20.43650794, RelativeWeightError=0.0, DACStep=0.1, DACNoise=0, ADCNoise=0, QuanVoltageLSB=0.3125, ADCOffset=0):
    weight_data = weight_d
    (h, w) = weight_data.shape
    conductance_data = weight_data
    if RelativeWeightError != 0:
        conductance_data = conductance_data + np.random.randn(h, w) * HalfMaxConductance * RelativeWeightError
    if DACNoise != 0:
        DACStep = DACStep + np.random.randn(1) * DACNoise
    OutCurrent = input_d @ conductance_data * DACStep
    (h_, w_) = OutCurrent.shape
    if ADCNoise != 0:
        OutCurrent = OutCurrent + np.random.randn(h_, w_) * ADCNoise
    OutVoltage = OutCurrent * 10 ** (-6) * it_time * 100 * 10 ** (-9) / (5.15 * 10 ** (-12))
    row_num = np.sum(input_d)
    alpha = 0.0002831 * row_num + 0.00668
    OutVoltage = OutVoltage - it_time * np.log(row_num) * alpha
    if ADCOffset != 0:
        mask = np.random.randint(-1, 2, (h_, w_))
        OutDigitalNum = np.round(np.floor(OutVoltage / QuanVoltageLSB) + ADCOffset * mask)
    else:
        OutDigitalNum = np.round(np.floor(OutVoltage / QuanVoltageLSB))
    OutDigitalNum = np.clip(OutDigitalNum, -8, 7)
    return OutDigitalNum

def sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data=None, adc_scale_data=None, it_time=5, n_scale=None, adc_bits=None, adc_clamp=None, ADC_LUT=None, adc_quant=None, macro_method=False, HalfMaxConductance=20.43650794, RelativeWeightError=0.0625, runtime='simulation', rpc_api=None, fit_k=1, fit_bias=0):
    it_time = round(it_time)
    cal_times = input_expanded.shape[1]
    output_cols = addr[3]
    ADC_output = np.zeros((cal_times, output_cols))
    if runtime == 'simulation':
        weight = get_weight(array_idx, addr, array_data=array_data)
        if macro_method:
            (h, w) = weight.shape
            weight_n = weight + np.random.randn(h, w) * HalfMaxConductance * RelativeWeightError
        else:
            weight_n = add_noise(weight, n_scale=n_scale)
        ADC_scale = get_ADC_scale(it_time=it_time, array_idx=array_idx, addr=addr, k=adc_scale_data, LUT=ADC_LUT)
        ADC_half_level = 2 ** adc_bits - 1
        '\n        for i in range(cal_times):\n            if macro_method:\n                input_ = input_expanded[:, i].reshape(1, -1)\n                ADC_output[i,:] = _macro_matmul(input_,weight_n,it_time)\n            else:\n                temp = (input_expanded[:, i].reshape(-1, 1) * weight_n).sum(axis = 0) * ADC_scale\n                if adc_quant:\n                    temp = temp.round()\n                if adc_clamp:\n                    temp[temp > ADC_half_level] = ADC_half_level\n                    temp[temp < -ADC_half_level - 1] = -ADC_half_level - 1\n                ADC_output[i, :] = temp.reshape(1, -1)\n        '
        if macro_method:
            input_ = input_expanded.transpose(1, 0)
            ADC_output = _macro_matmul(input_, weight_n, it_time)
        else:
            input_expanded = input_expanded.transpose(1, 0)
            temp = input_expanded @ weight_n * ADC_scale
            if adc_quant:
                temp = temp.round()
            if adc_clamp:
                temp[temp > ADC_half_level] = ADC_half_level
                temp[temp < -ADC_half_level - 1] = -ADC_half_level - 1
            ADC_output = temp
    elif runtime == 'c200':
        array_idx_ = int(array_idx.split(':')[-1])
        input_expanded = input_expanded.transpose(1, 0)
        input_expanded_ = input_expanded.tolist()
        ADC_output_ = rpc_api.call('c200_calc', input_expanded_, array_idx_, addr, it_time)
        ADC_output = None
        if str(type(ADC_output_)) == "<class 'list'>":
            ADC_output = np.array(ADC_output_)
        elif str(type(ADC_output_)) == "<class 'numpy.ndarray'>":
            ADC_output = ADC_output_
        else:
            raise ValueError(f'暂不支持 输出返回类型 {type(ADC_output_)}')
    else:
        raise ValueError(f"不支持的runtime : {runtime}, 仅支持('simulation','c200',)")
    ADC_output = fit_k * ADC_output + fit_bias
    return ADC_output

def mvm_cpu(input_data, array_idx, addr, array_data=None):
    weight = get_weight(array_idx, addr, array_data=array_data)
    cal_times = input_data.shape[1]
    output_cols = addr[3]
    mvm_output = np.zeros([cal_times, output_cols])
    for i in range(cal_times):
        temp = (input_data[:, i].reshape(-1, 1) * weight).sum(axis=0)
        mvm_output[i, :] = temp.reshape(1, -1)
    return mvm_output

def add_noise(weight, n_scale=0.05):
    w_range = weight.max() - weight.min()
    shape = weight.shape
    w_noise = w_range * n_scale * np.random.randn(*shape)
    weight_noise = weight + w_noise
    return weight_noise

def mvm_multi_bit_shift(array_idx, input_matrix, addr, dac_bits=1, repeat=None, it_time=5, verbose=0, assign_pulses=None, original_weight=None, device='cpu', array_data=None, adc_scale_data=None, adc_bits=None, n_scale=None, adc_clamp=None, ADC_LUT=None, adc_quant=None, macro_method=False, runtime='simulation', rpc_api=None):
    cal_times = input_matrix.shape[1]
    output_cols = addr[3]
    output = np.zeros([cal_times, output_cols])
    [input_expanded, bitlen_map] = input_multi_bits_shift_expansion(input_matrix, dac_bits=dac_bits)
    output_bitwise = sdk_cal_sim_noise(input_expanded, array_idx, addr, it_time=it_time, array_data=array_data, adc_scale_data=adc_scale_data, n_scale=n_scale, adc_bits=adc_bits, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api)
    output_bitwise_row = 0
    output_row = 0
    for j in bitlen_map:
        if j == 0:
            output[output_row, :] = 0
        else:
            factor_list = np.array([2 ** (i * dac_bits) for i in range(j)])
            factor_list = factor_list.reshape(j, -1)
            output_temp = output_bitwise[output_bitwise_row:output_bitwise_row + j] * factor_list
            output[output_row, :] = output_temp.sum(axis=0)
        output_row += 1
        output_bitwise_row += j
    (max_percent, min_percent) = count_overshoot_percent(output_bitwise, adc_bits=adc_bits)
    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([cal_times, output_avg_cols])
        for i in range(col_repeat):
            output_avg += output[:, i * output_avg_cols:(i + 1) * output_avg_cols]
        output_avg /= col_repeat
        output_avg /= row_repeat
        return output_avg.round()
    return (output, max_percent, min_percent)

def mvm_multi_bit_pulse(array_idx, input_matrix, addr, pulse_half_level=7, repeat=None, it_time=5, verbose=0, assign_pulses=None, original_weight=None, array_data=None, adc_scale_data=None, adc_bits=None, n_scale=None, adc_clamp=None, ADC_LUT=None, adc_quant=None, macro_method=False, runtime='simulation', rpc_api=None):
    cal_times = input_matrix.shape[1]
    output_cols = addr[3]
    output = np.zeros([cal_times, output_cols])
    (input_expanded, bitlen_map) = input_multi_bits_pulse_expansion(input_matrix, pulse_half_level=pulse_half_level, assign_pulses=assign_pulses)
    output_bitwise = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data=array_data, adc_scale_data=adc_scale_data, it_time=it_time, n_scale=n_scale, adc_bits=adc_bits, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api)
    output_bitwise_row = 0
    output_row = 0
    for j in bitlen_map:
        if j == 0:
            output[output_row, :] = 0
        else:
            output[output_row, :] = output_bitwise[output_bitwise_row:output_bitwise_row + j].sum(axis=0)
        output_row += 1
        output_bitwise_row += j
    (max_percent, min_percent) = count_overshoot_percent(output_bitwise, adc_bits=adc_bits)
    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([cal_times, output_avg_cols])
        for i in range(col_repeat):
            output_avg += output[:, i * output_avg_cols:(i + 1) * output_avg_cols]
        output_avg /= col_repeat
        output_avg /= row_repeat
        return output_avg.round()
    return (output, max_percent, min_percent)

def mvm_multi_bit_shift_batch(array_idx, input_matrix, addr, dac_bits=1, repeat=None, it_time=5, verbose=0, assign_pulses=None, original_weight=None, device='cpu', array_data=None, adc_scale_data=None, adc_bits=None, n_scale=None, adc_clamp=None, ADC_LUT=None, adc_quant=None, macro_method=False, runtime='simulation', rpc_api=None, fit_k=1, fit_bias=0):
    cal_times = input_matrix.shape[2]
    batch = input_matrix.shape[0]
    output_cols = addr[3]
    output = np.zeros([batch, cal_times, output_cols])
    input_expanded = []
    bitlen_map = []
    batch_expansion_len_list = []
    for i in range(batch):
        (input_expanded_, bitlen_map_) = input_multi_bits_shift_expansion(input_matrix[i, :, :], dac_bits=dac_bits)
        input_expanded.append(input_expanded_)
        bitlen_map.append(bitlen_map_)
        batch_expansion_len_list.append(input_expanded_.shape[1])
    input_expanded = np.concatenate(input_expanded, axis=1)
    output_bitwise_ = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data=array_data, adc_scale_data=adc_scale_data, it_time=it_time, n_scale=n_scale, adc_bits=adc_bits, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api, fit_k=fit_k, fit_bias=fit_bias)
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
        output_bitwise = output_bitwise_[start_batch_index:start_batch_index + batch_expansion_len, :]
        output_bitwise_row = 0
        output_row = 0
        for j in bitlen_map[batch_index]:
            if j == 0:
                output[batch_index, output_row, :] = 0
            else:
                factor_list = np.array([2 ** (i * dac_bits) for i in range(j)])
                factor_list = factor_list.reshape(j, -1)
                output_temp = output_bitwise[output_bitwise_row:output_bitwise_row + j] * factor_list
                output[batch_index, output_row, :] = output_temp.sum(axis=0)
            output_row += 1
            output_bitwise_row += j
        (max_percent_, min_percent_) = count_overshoot_percent(output_bitwise, adc_bits=adc_bits)
        max_percent.append(max_percent_)
        min_percent.append(min_percent_)
        if repeat:
            for i in range(col_repeat):
                output_avg[batch_index, :, :] += output[batch_index, :, i * output_avg_cols:(i + 1) * output_avg_cols]
            output_avg[batch_index, :, :] /= col_repeat
            output_avg[batch_index, :, :] /= row_repeat
        start_batch_index += batch_expansion_len
    if repeat:
        output = output_avg
    return (output, max_percent, min_percent)

def mvm_multi_bit_pulse_batch(array_idx, input_matrix, addr, pulse_half_level=7, repeat=None, it_time=5, array_data=None, adc_scale_data=None, adc_bits=None, n_scale=None, adc_clamp=None, ADC_LUT=None, adc_quant=None, macro_method=False, runtime='simulation', rpc_api=None, fit_k=1, fit_bias=0):
    cal_times = input_matrix.shape[2]
    batch = input_matrix.shape[0]
    output_cols = addr[3]
    output = np.zeros([batch, cal_times, output_cols])
    input_expanded = []
    bitlen_map = []
    batch_expansion_len_list = []
    if batch <= 10000:
        for i in range(batch):
            (input_expanded_, bitlen_map_) = input_multi_bits_pulse_expansion(input_matrix[i, :, :], pulse_half_level=pulse_half_level)
            input_expanded.append(input_expanded_)
            bitlen_map.append(bitlen_map_)
            batch_expansion_len_list.append(input_expanded_.shape[1])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() // 3))) as executor:
            results = list(executor.map(input_multi_bits_pulse_expansion, input_matrix.tolist(), [pulse_half_level] * batch))
        (input_expanded_, bitlen_map_) = zip(*results)
        for i in range(len(input_expanded_)):
            input_expanded.append(input_expanded_[i])
            bitlen_map.append(bitlen_map_[i])
            batch_expansion_len_list.append(input_expanded_[i].shape[1])
    input_expanded = np.concatenate(input_expanded, axis=1)
    output_bitwise_ = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data=array_data, adc_scale_data=adc_scale_data, it_time=it_time, n_scale=n_scale, adc_bits=adc_bits, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api, fit_k=fit_k, fit_bias=fit_bias)
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
        output_bitwise = output_bitwise_[start_batch_index:start_batch_index + batch_expansion_len, :]
        output_bitwise_row = 0
        output_row = 0
        for j in bitlen_map[batch_index]:
            if j == 0:
                output[batch_index, output_row, :] = 0
            else:
                output[batch_index, output_row, :] = output_bitwise[output_bitwise_row:output_bitwise_row + j].sum(axis=0)
            output_row += 1
            output_bitwise_row += j
        (max_percent_, min_percent_) = count_overshoot_percent(output_bitwise, adc_bits=adc_bits)
        max_percent.append(max_percent_)
        min_percent.append(min_percent_)
        if repeat:
            for i in range(col_repeat):
                output_avg[batch_index, :, :] += output[batch_index, :, i * output_avg_cols:(i + 1) * output_avg_cols]
            output_avg[batch_index, :, :] /= col_repeat
            output_avg[batch_index, :, :] /= row_repeat
        start_batch_index += batch_expansion_len
    if repeat:
        output = output_avg
    return (output, max_percent, min_percent)

def mvm_multi_bit_fast_batch(array_idx, input_matrix, addr, pulse_half_level=7, repeat=None, it_time=5, array_data=None, adc_scale_data=None, adc_bits=None, n_scale=None, adc_clamp=None, ADC_LUT=None, adc_quant=None, macro_method=False, runtime='simulation', rpc_api=None, fit_k=1, fit_bias=0):
    cal_times = input_matrix.shape[2]
    batch = input_matrix.shape[0]
    output_cols = addr[3]
    scale = np.max(abs(input_matrix))
    input_matrix = input_matrix / (scale + 10 ** (-6))
    input_expanded = input_matrix.transpose(0, 2, 1).reshape(batch * cal_times, -1)
    input_expanded = input_expanded.transpose(1, 0)
    output_bitwise_ = sdk_cal_sim_noise(input_expanded, array_idx, addr, array_data=array_data, adc_scale_data=adc_scale_data, it_time=it_time, n_scale=n_scale, adc_bits=adc_bits, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=macro_method, runtime=runtime, rpc_api=rpc_api, fit_k=fit_k, fit_bias=fit_bias)
    output_bitwise_ = output_bitwise_.reshape(batch, cal_times, output_cols)
    output = output_bitwise_
    max_percent = []
    min_percent = []
    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = (output_cols / col_repeat).round()
        output_avg = np.zeros([batch, cal_times, output_avg_cols])
    for batch_index in range(batch):
        output_bitwise = output[batch_index, :, :]
        (max_percent_, min_percent_) = count_overshoot_percent(output_bitwise, adc_bits=adc_bits)
        max_percent.append(max_percent_)
        min_percent.append(min_percent_)
        if repeat:
            for i in range(col_repeat):
                output_avg[batch_index, :, :] += output[batch_index, :, i * output_avg_cols:(i + 1) * output_avg_cols]
            output_avg[batch_index, :, :] /= col_repeat
            output_avg[batch_index, :, :] /= row_repeat
    if repeat:
        output = output_avg
    output = output * scale
    return (output, max_percent, min_percent)
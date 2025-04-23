import math
import numpy as np

# @wang ze
# numpy
# data quant 
def data_quantization_sym(data_float, half_level = 15, data_range = None,
                          isint = 1, clamp_std = 0, boundary_refine = False,
                          reg_shift_mode = False, reg_shift_bits = None):
    # isint = 1 -> return quantized values as integer levels
    # isint = 0 -> return quantized values as float numbers with the same range as input
    # reg_shift_mode -> force quant_scale to be exponent of 2, i.e., quant_scale = 2^n (n is integer)
    data_float = data_float + 0

    if half_level <= 0:
        return data_float, 1

    if boundary_refine:
        pass
        # half_level += 0.4999

    if clamp_std:
        std = data_float.std()
        data_float[data_float < (clamp_std * -std)] = (clamp_std * -std)
        data_float[data_float > (clamp_std * std)] = (clamp_std * std)

    if data_range == None or data_range == 0:
        data_range = round(abs(data_float).max(),7)

    if data_range == 0:
        return data_float, 1

    if reg_shift_mode:
        if reg_shift_bits != None:
            quant_scale = 2 ** reg_shift_bits
        else:
            shift_bits = round(math.log(1 / data_range * half_level, 2))
            quant_scale = 2 ** shift_bits
        data_quantized = (data_float * quant_scale).astype(np.int)
        # print(f'quant_scale = {quant_scale}')
        # print(f'reg_shift_bits = {reg_shift_bits}')
    else:
        data_quantized = np.round((data_float / data_range * half_level))
        quant_scale = 1 / data_range * half_level

    if isint == 0:
        data_quantized = data_quantized * data_range / half_level
        quant_scale = 1

    return data_quantized, quant_scale

# @wang namin
def binary_quant(data_float):
    '''
    二值量化
    input:
        x
    return:
         1 ; x>=0
        -1 ; x<0
    '''
    data_pos = np.int32(data_float >=0 )
    data_neg = np.int32(data_float <0 ) * -1
    return (data_pos + data_neg)

# @wang namin
def thres_binary_quant(data_float,thr=0.5):
    '''
    带阈值的二值量化
    input:
        x
    return:
        1 ; x>=thr
        0 ; x<thr
    '''
    
    data = np.int32(data_float >= thr)
    return data

# @wang namin
# Quantize input data, uniform quantize
def data_quantization(data_float, symmetric = True, bit = 8, clamp_std = None,
                        th_point='max', th_scale=None, all_positive=False):
    # data_float -> Input data needs to be quantized
    # symmetric -> whether use symmetric quantized, int range: [-(2**(bit-1)-1), 2**(bit-1)-1]
    # bit -> quant bits
    # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]
    # th_point -> clamp data_float mode
    # th_scale -> scale the clamp thred, used together with th_point
    # all_positive -> whether data_float is all positive, int range: [0, 2**bit-1]

    std = data_float.std()
    max_data = data_float.max()
    min_data = data_float.min()
    # if min_data.item() >= 0:
    #     all_positive = True

    if clamp_std != None and clamp_std != 0 and th_scale != None:
        raise ValueError("clamp_std and th_scale, only one clamp method can be used. ")
    if clamp_std != None and clamp_std != 0:
        data_float = np.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)
    else:
        if min_data.item() * max_data.item() < 0. and th_point == 'min':
            th = min(max_data.abs().item(), min_data.abs().item())
        else:
            th = max(max_data.abs().item(), min_data.abs().item())
        if th_scale != None:
            th *= th_scale
        data_float = np.clamp(data_float, min = -th, max = th)

    if all_positive:
        if data_float.min().item() < 0:
            raise ValueError("all_positive uniform_quantizer's data_float is not all positive. ")
        data_range = data_float.max()
        quant_range = 2**bit-1
        zero_point = 0
    elif symmetric:
        data_range = 2*abs(data_float).max()
        quant_range = 2**bit - 2
        zero_point = 0
    else:
        data_range = data_float.max() - data_float.min()
        quant_range = 2**bit - 1
        zero_point = data_float.min() / data_range * quant_range

    if data_range == 0:
        return data_float, 0

    scale = data_range / quant_range
    data_quantized = ((data_float / scale - zero_point).round() + zero_point) * scale

    return data_quantized
    
import numpy as np
import math

def input_bitwise_expansion_fast(input_matrix, dense=True, assign_pulses=None):
    input_matrix = input_matrix.round()
    if (input_matrix == 0).all():
        return (input_matrix, [])
    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int32)
    input_matrix = input_matrix.astype(np.int32)
    input_ori = input_matrix + 0
    max_range = int(np.max(abs(input_matrix)))
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]
    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)
    input_expanded = np.zeros((rows * cols, max_range), dtype=int)
    for i in range(max_range):
        bit_cur = (input_matrix > 0) * 1 + (input_matrix < 0) * -1
        input_expanded[:, i:i + 1] = bit_cur
        input_matrix -= bit_cur
    input_expanded = bitwise_shift(input_expanded)
    input_expanded = input_expanded.reshape(-1, rows, max_range).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)
    if dense == True:
        mask = abs(input_expanded).sum(axis=0)
        input_expanded = input_expanded[:, mask != 0]
        zero_array_num = (mask == 0).astype(int).reshape(1, -1)
        zero_array_num = zero_array_num.reshape(-1, max_range).sum(axis=1)
        bitlen_map = max_range - zero_array_num
    else:
        bitlen_map = (np.ones([cols]) * abs(input_ori).max()).astype(np.int32)
    return (input_expanded, bitlen_map)

def input_multi_bits_shift_expansion(input_matrix, dac_bits=1):
    input_matrix = input_matrix.round()
    if (input_matrix == 0).all():
        return (input_matrix, [])
    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int32)
    input_matrix = input_matrix.astype(np.int32)
    input_ori = input_matrix + 0
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]
    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)
    shift_value = 2 ** dac_bits - 1
    input_bits = math.floor(math.log2(max(abs(input_matrix)))) + 1
    max_expansion_times = math.ceil(input_bits / dac_bits)
    input_expanded = np.zeros((rows * cols, max_expansion_times), dtype=int)
    input_matrix_sign = np.sign(input_matrix)
    for i in range(max_expansion_times):
        input_matrix = abs(input_matrix)
        pulse_cur = (input_matrix & shift_value) * input_matrix_sign
        input_expanded[:, i:i + 1] = pulse_cur
        input_matrix = input_matrix >> dac_bits
    input_expanded = input_expanded.reshape(-1, rows, max_expansion_times).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)
    pulse_len_map = (np.ones([cols]) * max_expansion_times).astype(np.int32)
    return (input_expanded, pulse_len_map)

def count_overshoot_percent(output_bitwise, adc_bits=None):
    ADC_half_level = 2 ** adc_bits // 2 - 1
    count_max = (output_bitwise >= ADC_half_level - 1).sum()
    count_min = (output_bitwise <= -ADC_half_level + 1).sum()
    tot_element = output_bitwise.size
    max_percent = count_max / tot_element
    min_percent = count_min / tot_element
    return (max_percent, min_percent)

def input_multi_bits_pulse_expansion(input_matrix, pulse_half_level=7):
    if isinstance(input_matrix, list):
        input_matrix = np.array(input_matrix)
    input_matrix = input_matrix.round()
    assert input_matrix.max() <= 127
    assert input_matrix.min() >= -128
    if (input_matrix == 0).all():
        return (input_matrix, [])
    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int8)
    input_matrix = input_matrix.astype(np.int8)
    input_ori = input_matrix + 0
    max_expansion_times = math.ceil(np.max(abs(input_matrix)) / pulse_half_level)
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]
    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)
    input_expanded = np.zeros((rows * cols, max_expansion_times), dtype=np.int8)
    for i in range(max_expansion_times - 1):
        pulse_cur = (input_matrix >= pulse_half_level) * pulse_half_level + (input_matrix <= -pulse_half_level) * -pulse_half_level
        input_expanded[:, i:i + 1] = pulse_cur
        input_matrix -= pulse_cur
    input_expanded[:, -1] = input_matrix.reshape(-1)
    input_expanded = bitwise_shift(input_expanded)
    input_expanded = input_expanded.reshape(-1, rows, max_expansion_times).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)
    mask = abs(input_expanded).sum(axis=0)
    input_expanded = input_expanded[:, mask != 0]
    zero_array_num = (mask == 0).astype(int).reshape(1, -1)
    zero_array_num = zero_array_num.reshape(-1, max_expansion_times).sum(axis=1)
    pulse_len_map = max_expansion_times - zero_array_num
    return (input_expanded, pulse_len_map)

def expand_GPT(mat_in, pulse_half_level):
    max_val = (np.abs(mat_in) + pulse_half_level - 1) // pulse_half_level
    max_val = max_val.max()
    mat_in_expanded = np.zeros((np.prod(mat_in.shape), max_val)).astype(np.int32)
    depth = np.abs(mat_in) // pulse_half_level
    remainder = np.abs(mat_in) % pulse_half_level
    sign = np.sign(mat_in)
    range_mat = np.broadcast_to(np.arange(max_val)[None, :], (np.prod(mat_in.shape), max_val))
    mat_in_expanded = np.where(range_mat < depth, sign * pulse_half_level, 0)
    mat_in_expanded = np.where(range_mat == depth, sign * remainder, mat_in_expanded)
    return mat_in_expanded

def input_multi_bits_pulse_expansion_GPT(input_matrix, pulse_half_level=7):
    input_matrix = input_matrix.round()
    assert input_matrix.max() <= 127
    assert input_matrix.min() >= -128
    if (input_matrix == 0).all():
        return (input_matrix, [])
    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int8)
    input_matrix = input_matrix.astype(np.int8)
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]
    max_expansion_times = math.ceil(np.max(abs(input_matrix)) / pulse_half_level)
    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)
    input_expanded = expand_GPT(input_matrix, pulse_half_level)
    input_expanded = bitwise_shift(input_expanded)
    input_expanded = input_expanded.reshape(-1, rows, max_expansion_times).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)
    mask = abs(input_expanded).sum(axis=0)
    input_expanded = input_expanded[:, mask != 0]
    zero_array_num = (mask == 0).astype(int).reshape(1, -1)
    zero_array_num = zero_array_num.reshape(-1, max_expansion_times).sum(axis=1)
    pulse_len_map = max_expansion_times - zero_array_num
    return (input_expanded, pulse_len_map)

def input_multi_bits_pulse_expansion_batch(input_matrix, pulse_half_level=7):
    batch = input_matrix.shape[0]
    input_expanded = []
    pulse_len_map = []
    batch_expansion_len_list = []
    for i in range(batch):
        (input_expanded_, bitlen_map_) = input_multi_bits_pulse_expansion(input_matrix[i, :, :], pulse_half_level=pulse_half_level)
        input_expanded.append(input_expanded_)
        pulse_len_map.append(bitlen_map_)
        batch_expansion_len_list.append(input_expanded_.shape[1])
    input_expanded = np.concatenate(input_expanded, axis=1)
    return (input_expanded, pulse_len_map, batch_expansion_len_list)

def bitwise_shift(input_expanded):
    bitlen = input_expanded.shape[1]
    if bitlen <= 1:
        return input_expanded
    input_shift_count = input_expanded + 0
    roll = abs(input_shift_count).sum(axis=1)
    roll = (np.cumsum(roll) - roll) % bitlen
    (rows, column_indices) = np.ogrid[:input_expanded.shape[0], :input_expanded.shape[1]]
    roll[roll < 0] += input_expanded.shape[1]
    column_indices = column_indices - roll[:, np.newaxis]
    input_expanded_shifted = input_expanded[rows, column_indices]
    return input_expanded_shifted

def bitwise_shift_batch(input_expanded):
    bitlen = input_expanded.shape[1]
    if bitlen <= 1:
        return input_expanded
    input_shift_count = input_expanded + 0
    roll = abs(input_shift_count).sum(axis=1)
    roll = (np.cumsum(roll) - roll) % bitlen
    (rows, column_indices) = np.ogrid[:input_expanded.shape[0], :input_expanded.shape[1]]
    roll[roll < 0] += input_expanded.shape[1]
    column_indices = column_indices - roll[:, np.newaxis]
    input_expanded_shifted = input_expanded[rows, column_indices]
    return input_expanded_shifted
import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from a111sdk import a111_mapping_weight, a111_read_weight, FC_one_layer, FC_two_layer, Conv_one_layer
from utilize.plot_api import *
plt.rcParams.update({'font.size': 15})

def auto_mapping_weight(bias_rram_weight, tile=3, xb=0, row_begin=0, colum_begin=0, program_times=3):
    
    shapes = bias_rram_weight.shape
    (row_length, colum_length) = (shapes[0], shapes[1])
    index = [row_begin, colum_begin, row_length, colum_length]
    a111_mapping_weight(bias_rram_weight, tile_id=tile, xb_id=xb, addr=index, program_times=program_times)

def calculate_FC_one_layer(input_data, tile=0, xb=[0], num_column=128, shift_num=[2], adc_range=[32], relu=True):
    
    begin = time.time()
    tile_id = tile
    xb_id_list = xb
    shift_list = [2, 2, 2, 2]
    adc_range_list = [1, 1, 1, 1, 1, 1, 1, 1]
    for (ith, xb_id) in enumerate(xb):
        shift_list[xb_id // 2] = shift_num[ith]
        adc_range_list[ith] = adc_range[ith] // 16 - 1
    output = FC_one_layer(tile_id, xb_id_list, input_data, output_column=[0, num_column], adc_range_list=adc_range_list, shift_list=shift_list, relu=relu)
    end = time.time() - begin
    print()
    return output

def calculate_FC_two_layer(input_data, tile=0, xb=[0, 2], output_columns=[64, 10], shift_num=[2, 2], adc_range=[32, 32]):
    
    begin = time.time()
    tile_id = tile
    xb_id_list = xb
    shift_list = [2, 2, 2, 2]
    adc_range_list = [1, 1, 1, 1, 1, 1, 1, 1]
    for (ith, xb_id) in enumerate(xb):
        shift_list[xb_id // 2] = shift_num[ith]
        adc_range_list[ith] = adc_range[ith] // 16 - 1
    output = FC_two_layer(tile_id, xb_id_list, input_data, output_column1=[0, output_columns[0]], output_column2=[0, output_columns[1]], adc_range_list=adc_range_list, shift_list=shift_list, second_relu=False)
    end = time.time() - begin
    print()
    return output

def calculate_Conv_one_layer(input_data, tile=0, xb=[0], num_column=128, kernel_size=3, stride=1, padding=1, relu=False, shift_num=[2], adc_range=[32], bias=False, bias_num=[2], bias_input_value_list=[[0]]):
    
    begin = time.time()
    tile_id = tile
    xb_id_list = xb
    shift_list = [2, 2, 2, 2]
    adc_range_list = [1, 1, 1, 1, 1, 1, 1, 1]
    for (ith, xb_id) in enumerate(xb):
        shift_list[xb_id // 2] = shift_num[ith]
        adc_range_list[ith] = adc_range[ith] // 16 - 1
    output = Conv_one_layer(tile_id, xb_id_list, input_data, output_column=[0, num_column], kernel_size=kernel_size, stride=stride, padding=padding, relu=relu, adc_range_list=adc_range_list, shift_list=shift_list, bias=bias, bias_num=bias_num, bias_input_value_list=bias_input_value_list)
    end = time.time() - begin
    print()
    return output

def make_rram_bias_weight(offsets_m, rram_b_weight, rram_read_weight=False, bias_row=10, threshold=10, pos_sa=5, neg_sa=5):
    bias_colums = 128
    rram_bias_row = bias_row * 2
    mapping_weight = np.zeros([((rram_bias_row - 1) // 32 + 1) * 32, bias_colums])
    if rram_read_weight:
        rram_b_weight = rram_b_weight[256 * 2 + 16 * 2:256 * 2 + 16 * 2 + mapping_weight.shape[0], :]
    pos_weight = rram_b_weight[::2, :].copy()
    neg_weight = rram_b_weight[1::2, :].copy()
    offsets_m = pt_sequence_2_rram_discretization(offsets_m[None])[0]
    offsets_m_128 = np.zeros(bias_colums)
    offsets_m_128[:offsets_m.shape[0]] = offsets_m
    offset_gt_th = offsets_m_128 > threshold
    offset_lt_th = offsets_m_128 < -threshold
    neg_weight[:, offset_gt_th] = neg_sa
    pos_weight[:, offset_gt_th] = 0
    pos_weight[:, offset_lt_th] = pos_sa
    neg_weight[:, offset_lt_th] = 0
    mapping_weight[::2, :] = pos_weight
    mapping_weight[1::2, :] = neg_weight
    mapping_weight = mapping_weight.astype(np.int8)
    return mapping_weight

def make_rram_bias_weight_from_pt_weight(pt_weight, pos_sa=5, neg_sa=5):
    (row, colum) = pt_weight.shape
    rram_weight = np.zeros([row * 2, colum])
    pos_weight = np.zeros_like(pt_weight)
    neg_weight = np.zeros_like(pt_weight)
    flag = pt_weight > 0
    pos_weight[flag] = pos_sa
    flag = pt_weight < 0
    neg_weight[flag] = neg_sa
    rram_weight[::2, :] = pos_weight
    rram_weight[1::2, :] = neg_weight
    sub_mapping_weight = pt_sequence_2_rram_discretization(rram_weight)
    mapping_weight = np.zeros([32 * ((row * 2 - 1) // 32 + 1), colum])
    mapping_weight[:rram_weight.shape[0]] = sub_mapping_weight
    mapping_weight = mapping_weight.astype(np.uint8)
    return mapping_weight

def pt_sequence_2_rram_discretization(pt_sequence):
    (pt_sequence_row, pt_sequence_colum) = pt_sequence.shape
    rram_discretization = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum[:, :pt_sequence_colum] = pt_sequence
    for rram_colum in range(127):
        mapping_index = 4 * rram_colum % 127
        rram_discretization[:, mapping_index] = pt_sequence_128colum[:, rram_colum]
    rram_discretization[:, 127] = pt_sequence_128colum[:, 127]
    return rram_discretization

def trans_pt_weight_2_rram(pt_weight, row_begin=8, pos_sa=3, neg_sa=3):
    (row, colum) = pt_weight.shape
    rram_weight = np.zeros([row * 2, colum])
    pos_weight = np.zeros_like(pt_weight)
    neg_weight = np.zeros_like(pt_weight)
    flag = pt_weight > 0
    pos_weight[flag] = pos_sa
    flag = pt_weight < 0
    neg_weight[flag] = neg_sa
    rram_weight[::2, :] = pos_weight
    rram_weight[1::2, :] = neg_weight
    sub_mapping_weight = pt_sequence_2_rram_discretization(rram_weight)
    mapping_weight = np.zeros([640, 128])
    mapping_weight[row_begin * 2:row_begin * 2 + rram_weight.shape[0]] = sub_mapping_weight
    mapping_weight = mapping_weight.astype(np.uint8)
    return mapping_weight

def trans_pt_data_2_rram(pt_data, voltage=136):
    rram_data = pt_data * voltage
    rram_data = rram_data.astype(np.uint8)
    return rram_data

def trans_rram_weight_pt(read_weight, SA_scale=3):
    read_weight = read_weight.astype(np.float32)
    (dim1, dim2) = read_weight.shape
    data_weight = read_weight[::2, :] - read_weight[1::2, :]
    data_weight = data_weight / SA_scale
    new_data_weight = np.zeros([dim1 // 2, dim2])
    assert dim2 // 4 == 32, 'array shape error'
    for t in range(4):
        new_data_weight[:, 32 * t:(t + 1) * 32] = data_weight[:, t::4]
    return new_data_weight

def pt_weight_4dim_2_rram_weight_2dim(value):
    return value.transpose(0, 2, 3, 1).reshape(value.shape[0], -1).T

def rram_weight_2dim_2_pt_weight_4dim(value, in_channel, kernel_size=3):
    shape = value.shape
    return value.T.reshape(shape[1], kernel_size, kernel_size, in_channel).transpose(0, 3, 1, 2)

def check_conv_read_weight(pt_weight, pt_input, tile=0, xb=[0], index=[0, 0, 640, 128], row_begin=8, save_root=None, show_hist=False, show_acc=True, show_read_weight_hist=False, kernel_size=3, stride=1, padding=1):
    pt_2_rram_weight = pt_weight_4dim_2_rram_weight_2dim(pt_weight)
    row_len = pt_2_rram_weight.shape[0] // len(xb)
    col_len = pt_2_rram_weight.shape[1]
    rram_read_weights = []
    rram_weight_2_pts = []
    for xb_id in xb:
        rram_read_weight = a111_read_weight(tile_id=tile, xb_id=xb_id, addr=index)
        rram_weight_2_pt = trans_rram_weight_pt(rram_read_weight)
        rram_read_weights.append(rram_read_weight)
        rram_weight_2_pts.append(rram_weight_2_pt[row_begin:row_begin + row_len, :col_len])
    rram_read_weight = np.vstack(rram_read_weights)
    rram_weight_2_pt = np.vstack(rram_weight_2_pts)
    rram_weight_2_pt = rram_weight_2dim_2_pt_weight_4dim(rram_weight_2_pt, in_channel=pt_weight.shape[1], kernel_size=kernel_size)
    if show_read_weight_hist:
        (_, bin, _) = plt.hist(rram_read_weight.flatten(), bins=8, range=(0, 8), density=True, align='mid')
        plt.xticks(bin[:-1] + 1 / 2, list(range(8)))
        plt.title('read weight hist')
        plt.ylim(ymax=0.5)
        plt.show()
        plt.close()
    if show_hist:
        pt_out = conv2d(pt_input, pt_weight, stride=stride, padding=padding)
        read_weight_out = conv2d(pt_input, rram_weight_2_pt, stride=stride, padding=padding)
        show_sim_rram_out_hist(pt_out, read_weight_out)
    if save_root is not None:
        for (xb_id, rram_read_weight) in zip(xb, rram_read_weights):
            save_path = os.path.join(save_root, 'rram_read_weight_tile_%d_xb%d.csv' % (tile, xb_id))
            pd.DataFrame(rram_read_weight).to_csv(save_path, header=False, index=False)
        if show_hist:
            save_path = os.path.join(save_root, 'pt_weight_sim_out_%d_xb%s.npy' % (tile, '_'.join(list(map(str, xb)))))
            np.save(save_path, pt_out)
            save_path = os.path.join(save_root, 'rram_read_weight_sim_out_%d_xb%s.npy' % (tile, '_'.join(list(map(str, xb)))))
            np.save(save_path, read_weight_out)
    return rram_weight_2_pt

def check_fc_read_weight(pt_weight, pt_input, tile=0, xb=[0], index=[0, 0, 640, 128], row_begin=8, save_root=None, show_hist=False, show_acc=True, show_read_weight_hist=False):
    pt_2_rram_weight = pt_weight
    row_len = pt_2_rram_weight.shape[0] // len(xb)
    col_len = pt_2_rram_weight.shape[1]
    rram_read_weights = []
    rram_weight_2_pts = []
    for xb_id in xb:
        rram_read_weight = a111_read_weight(tile_id=tile, xb_id=xb_id, addr=index)
        rram_weight_2_pt = trans_rram_weight_pt(rram_read_weight)
        rram_read_weights.append(rram_read_weight)
        rram_weight_2_pts.append(rram_weight_2_pt[row_begin:row_begin + row_len, :col_len])
    rram_read_weight = np.vstack(rram_read_weights)
    rram_weight_2_pt = np.vstack(rram_weight_2_pts)
    if show_read_weight_hist:
        (_, bin, _) = plt.hist(rram_read_weight.flatten(), bins=8, range=(0, 8), density=True, align='mid')
        plt.xticks(bin[:-1] + 1 / 2, list(range(8)))
        plt.title('read weight hist')
        plt.ylim(ymax=0.5)
        plt.show()
        plt.close()
    if show_hist:
        pt_out = linear(pt_input, pt_weight)
        read_weight_out = linear(pt_input, rram_weight_2_pt)
        show_sim_rram_out_hist(pt_out, read_weight_out)
    if save_root is not None:
        for (xb_id, rram_read_weight) in zip(xb, rram_read_weights):
            save_path = os.path.join(save_root, 'rram_read_weight_tile_%d_xb%d.csv' % (tile, xb_id))
            pd.DataFrame(rram_read_weight).to_csv(save_path, header=False, index=False)
        if show_hist:
            save_path = os.path.join(save_root, 'pt_weight_sim_out_%d_xb%s.npy' % (tile, '_'.join(list(map(str, xb)))))
            np.save(save_path, pt_out)
            save_path = os.path.join(save_root, 'rram_read_weight_sim_out_%d_xb%s.npy' % (tile, '_'.join(list(map(str, xb)))))
            np.save(save_path, read_weight_out)
    return rram_weight_2_pt

def auto_adjust_fc_offsets_old(adc_range, tile, xb=[0], save_root=None, columns=128, row_begin=288, sample_num=5, show_offset=True):
    xb_num = len(xb)
    offset_input = np.zeros([sample_num, 320 * xb_num]).astype(np.uint8)
    for i in range(1):
        offsets = calculate_FC_one_layer(offset_input, tile=tile, xb=xb, num_column=columns, shift_num=[4] * xb_num, adc_range=[adc_range] * xb_num, relu=False)
    offset_mean = show_offsets(offsets[1:], columns=columns, show=show_offset)
    median = np.median(offset_mean)
    first_open_weight_row_num = abs(int(median / 10))
    bias_weight = np.zeros([32, 128])
    if median > 10:
        bias_sign = -1
    elif median < -10:
        bias_sign = 1
    else:
        bias_sign = 0
    bias_weight[:first_open_weight_row_num] = bias_sign
    bias_weight = bias_weight.astype(np.int8)
    first_bias_weight = make_rram_bias_weight_from_pt_weight(bias_weight, pos_sa=5, neg_sa=5)
    xb_index = row_begin // 320
    auto_mapping_weight(first_bias_weight, row_begin=(row_begin - 320 * xb_index) * 2, colum_begin=0, tile=tile, xb=xb[xb_index])
    offset_input = np.zeros([sample_num, 320 * xb_num]).astype(np.uint8)
    offset_input[:, row_begin:row_begin + first_open_weight_row_num] = 255
    offsets = calculate_FC_one_layer(offset_input, tile=tile, xb=xb, num_column=columns, shift_num=[4] * xb_num, adc_range=[adc_range] * xb_num, relu=False)
    offset_mean = show_offsets(offsets[1:], columns=columns, show=show_offset)
    median = np.median(offset_mean)
    while median > 10:
        first_open_weight_row_num += 1
        if first_open_weight_row_num > 32:
            break
        bias_weight = np.zeros([32, 128])
        bias_weight[:first_open_weight_row_num] = -1
        bias_weight = bias_weight.astype(np.int8)
        first_bias_weight = make_rram_bias_weight_from_pt_weight(bias_weight, pos_sa=5, neg_sa=5)
        auto_mapping_weight(first_bias_weight, row_begin=(row_begin - 320 * xb_index) * 2, colum_begin=0, tile=tile, xb=xb[xb_index])
        offset_input = np.zeros([sample_num, 320]).astype(np.uint8)
        offset_input[:, row_begin:row_begin + first_open_weight_row_num] = 255
        offsets = calculate_FC_one_layer(offset_input, tile=tile, xb=xb, num_column=columns, shift_num=[4] * xb_num, adc_range=[adc_range] * xb_num, relu=False)
        offset_mean = show_offsets(offsets[1:], columns=columns, show=show_offset)
        median = np.median(offset_mean)
    second_open_weight_row_num = 2
    second_bias_weight = first_bias_weight.copy()
    for t in range(int(32 - first_open_weight_row_num) // second_open_weight_row_num):
        offset_max = np.abs(offset_mean).max()
        if offset_max < 10:
            break
        elif offset_max > 20:
            threshold = 15
            voltage = 255
        else:
            threshold = 10
            voltage = 51
        temp = np.zeros([32, 128]).astype(np.int8)
        temp = make_rram_bias_weight(offset_mean, rram_b_weight=temp, bias_row=second_open_weight_row_num, threshold=threshold)[:second_open_weight_row_num * 2, :]
        begin_weight_index = 2 * first_open_weight_row_num + 2 * second_open_weight_row_num * t
        end_weight_index = begin_weight_index + 2 * second_open_weight_row_num
        second_bias_weight[begin_weight_index:end_weight_index, :] = temp
        auto_mapping_weight(second_bias_weight, row_begin=(row_begin - 320 * xb_index) * 2, colum_begin=0, tile=tile, xb=xb[xb_index])
        begin_input_index = row_begin + first_open_weight_row_num + second_open_weight_row_num * t
        end_input_index = begin_input_index + second_open_weight_row_num
        offset_input[:, begin_input_index:end_input_index] = voltage
        offsets = calculate_FC_one_layer(offset_input, tile=tile, xb=xb, num_column=columns, shift_num=[4] * xb_num, adc_range=[adc_range] * xb_num, relu=False)
        offset_mean = show_offsets(offsets, columns=columns, show=show_offset)
    if save_root is not None:
        pd.DataFrame(second_bias_weight).to_csv(os.path.join(save_root, 'second_bias_weight_xb%d.csv' % xb[0]), header=False, index=False)
        pd.DataFrame(offset_input).to_csv(os.path.join(save_root, 'offset_inputs_xb%d.csv' % xb[0]), header=False, index=False)
        pd.DataFrame(offsets).to_csv(os.path.join(save_root, 'offsets_xb%d.csv' % xb[0]), header=False, index=False)
    return offset_input

def auto_adjust_conv_offsets(rram_input, rram_weight, pt_input, pt_weight, tile, xb, num_column, kernel_size, stride, padding, relu, shift_num, adc_range, bias, bias_num, offset_row_begin, pos_sa=5, neg_sa=5, map_bias_weight=True, sample_num=10, show_offset=True):
    offset_input = np.zeros_like(rram_input[:sample_num]).astype(np.uint8)
    offsets = calculate_Conv_one_layer(offset_input, tile=tile, xb=xb, num_column=num_column, kernel_size=kernel_size, stride=stride, padding=padding, relu=relu, shift_num=shift_num, adc_range=adc_range, bias=bias, bias_num=bias_num, bias_input_value_list=[[0]] * len(xb))
    offset_mean = show_offsets(offsets[1:].transpose((1, 0, 2, 3)).reshape(num_column, -1).T, columns=num_column, show=show_offset)
    median = np.median(offset_mean)
    if median > 10:
        bias_sign = -1
    elif median < -10:
        bias_sign = 1
    else:
        bias_sign = 0
    first_open_weight_row_num = 8
    bias_weight = np.zeros([16, 128])
    bias_weight[:first_open_weight_row_num] = bias_sign
    bias_weight = bias_weight.astype(np.int8)
    first_bias_weight = make_rram_bias_weight_from_pt_weight(bias_weight, pos_sa=pos_sa, neg_sa=neg_sa)
    for (ith, xb_id) in enumerate(xb):
        rram_weight_xb = rram_weight[ith]
        first_bias_weight[offset_row_begin * 2:32] = rram_weight_xb[offset_row_begin * 2:32]
        if map_bias_weight:
            auto_mapping_weight(first_bias_weight, row_begin=0, colum_begin=0, tile=tile, xb=xb_id)
    offsets = calculate_Conv_one_layer(offset_input, tile=tile, xb=xb, num_column=num_column, kernel_size=kernel_size, stride=stride, padding=padding, relu=relu, shift_num=shift_num, adc_range=adc_range, bias=bias, bias_num=bias_num, bias_input_value_list=[[0]] * len(xb))
    offset_mean = show_offsets(offsets[1:].transpose((1, 0, 2, 3)).reshape(num_column, -1).T, columns=num_column, show=show_offset)
    median = np.median(offset_mean)
    bias_input_value = 0
    bias_input_stride = 17895697
    flag1 = median > 0
    while abs(median) > 10:
        bias_input_value += bias_input_stride
        offsets = calculate_Conv_one_layer(offset_input[:sample_num], tile=tile, xb=xb, num_column=num_column, kernel_size=kernel_size, stride=stride, padding=padding, relu=relu, shift_num=shift_num, adc_range=adc_range, bias=bias, bias_num=bias_num, bias_input_value_list=[[bias_input_value]] * len(xb))
        offset_mean = show_offsets(offsets[1:].transpose((1, 0, 2, 3)).reshape(num_column, -1).T, columns=num_column, show=show_offset)
        median = np.median(offset_mean)
        flag2 = median > 0
        if bias_input_value + bias_input_stride > 268435455 or flag1 != flag2:
            break
    print()
    return bias_input_value

def auto_adjust_conv_offsets_v2(rram_input, rram_weight, pt_input, pt_weight, tile, xb, num_column, kernel_size, stride, padding, relu, shift_num, adc_range, bias, bias_num, offset_row_begin, pos_sa=5, neg_sa=5, map_bias_weight=True, sample_num=10):
    rram_input = rram_input * 0
    rram_output = calculate_Conv_one_layer(rram_input[:sample_num], tile=tile, xb=xb, num_column=num_column, kernel_size=kernel_size, stride=stride, padding=padding, relu=relu, shift_num=shift_num, adc_range=adc_range, bias=bias, bias_num=bias_num, bias_input_value_list=[[0]] * len(xb))
    rram_output = np.mean(rram_output, axis=(0, 2, 3))
    mean = rram_output.mean()
    th_mean = 15
    if mean > th_mean:
        bias_sign = -1
    elif mean < -th_mean:
        bias_sign = 1
    else:
        bias_sign = 0
    first_open_weight_row_num = 8
    bias_weight = np.zeros([16, 128])
    bias_weight[:first_open_weight_row_num] = bias_sign
    bias_weight = bias_weight.astype(np.int8)
    first_bias_weight = make_rram_bias_weight_from_pt_weight(bias_weight, pos_sa=pos_sa, neg_sa=neg_sa)
    for (ith, xb_id) in enumerate(xb):
        rram_weight_xb = rram_weight[ith]
        first_bias_weight[offset_row_begin * 2:32] = rram_weight_xb[offset_row_begin * 2:32]
        if map_bias_weight:
            auto_mapping_weight(first_bias_weight, row_begin=0, colum_begin=0, tile=tile, xb=xb_id)
    rram_output = calculate_Conv_one_layer(rram_input[:sample_num], tile=tile, xb=xb, num_column=num_column, kernel_size=kernel_size, stride=stride, padding=padding, relu=relu, shift_num=shift_num, adc_range=adc_range, bias=bias, bias_num=bias_num, bias_input_value_list=[[0]] * len(xb))
    rram_output_col = np.mean(rram_output, axis=(0, 2, 3))
    mean = rram_output_col.mean()
    bias_input_value = 0
    bias_input_stride = 17895697
    mean_iterate = []
    bias_input_iterate = []
    mean_list = []
    mean_col = rram_output_col
    if not (abs(mean) > th_mean or abs(mean_col.max()) > 2 * th_mean):
        mean_list.append(mean_col)
        mean_iterate.append(abs(mean))
        bias_input_iterate.append(bias_input_value)
    while abs(mean) > th_mean or abs(mean_col.max()) > 2 * th_mean:
        bias_input_value += bias_input_stride
        rram_output = calculate_Conv_one_layer(rram_input[:sample_num], tile=tile, xb=xb, num_column=num_column, kernel_size=kernel_size, stride=stride, padding=padding, relu=relu, shift_num=shift_num, adc_range=adc_range, bias=bias, bias_num=bias_num, bias_input_value_list=[[bias_input_value]] * len(xb))
        mean = rram_output.mean()
        mean_col = np.mean(rram_output, axis=(0, 2, 3))
        mean_list.append(mean_col)
        mean_iterate.append(abs(mean))
        bias_input_iterate.append(bias_input_value)
        if bias_input_value + bias_input_stride > 268435455:
            break
    mean_list = np.stack(mean_list)
    mean_min = min(mean_iterate)
    mean_min_index = mean_iterate.index(mean_min)
    bias_input_value = bias_input_iterate[mean_min_index]
    print()
    print()
    x = np.linspace(1, len(mean_iterate), len(mean_iterate))
    y = np.array(mean_iterate)
    dot_line_plot(x=x, y=y, y2=None, title='title', xlabel='bias input', ylabel='offset af bias', line1_label='mean_iterate', line2_label='', line_color='b', line_width=2, save_path=None)
    return_dict = {}
    return_dict['bias_input_value'] = bias_input_value
    return_dict['mean_min_columns'] = mean_list[mean_min_index]
    return_dict['mean_min'] = mean_min
    return bias_input_value

def auto_adjust_fc_offsets(rram_input, rram_weight, pt_input, pt_weight, tile, xb, num_column, relu, shift_num, adc_range, bias, offset_row_begin, pos_sa=5, neg_sa=5, map_bias_weight=True, show_offset=True):
    sample_num = 100
    offset_input = np.zeros_like(rram_input[:sample_num]).astype(np.uint8)
    rram_output = calculate_FC_one_layer(offset_input, tile=tile, xb=xb, num_column=num_column, relu=relu, shift_num=shift_num, adc_range=adc_range)
    print()
    print()
    exit()
    median = np.median(rram_output)
    offset_mean = show_offsets(rram_output[1:], columns=num_column, show=show_offset)
    if median > 10:
        bias_sign = -1
    elif median < -10:
        bias_sign = 1
    else:
        bias_sign = 0
    first_open_weight_row_num = 8
    bias_weight = np.zeros([16, 128])
    bias_weight[:first_open_weight_row_num] = bias_sign
    bias_weight = bias_weight.astype(np.int8)
    first_bias_weight = make_rram_bias_weight_from_pt_weight(bias_weight, pos_sa=pos_sa, neg_sa=neg_sa)
    for (ith, xb_id) in enumerate(xb):
        rram_weight_xb = rram_weight[ith]
        first_bias_weight[offset_row_begin * 2:32] = rram_weight_xb[offset_row_begin * 2:32]
        if map_bias_weight:
            auto_mapping_weight(first_bias_weight, row_begin=0, colum_begin=0, tile=tile, xb=xb_id)
    rram_output = calculate_FC_one_layer(offset_input, tile=tile, xb=xb, num_column=num_column, relu=relu, shift_num=shift_num, adc_range=adc_range)
    median = np.median(rram_output)
    bias_input_value = 0
    bias_input_stride = 17
    bias_input = offset_input.copy()
    flag1 = median > 0
    while abs(median) > 10:
        bias_input_value += bias_input_stride
        bias_input[:, :offset_row_begin] = bias_input_value
        rram_output = calculate_FC_one_layer(bias_input, tile=tile, xb=xb, num_column=num_column, relu=relu, shift_num=shift_num, adc_range=adc_range)
        median = np.median(rram_output)
        offset_mean = show_offsets(rram_output[1:], columns=num_column, show=show_offset)
        flag2 = median > 0
        if bias_input_value + bias_input_stride > 255 or flag1 != flag2:
            break
    print()
    return bias_input_value

def pd_read_data(data_path):
    data = pd.read_csv(data_path, header=None, index_col=None).values
    return data

def auto_batch_size(input_size):
    batch_size1 = 1024 * 1024 / 2 / input_size
    batch_size2 = int(131072) / input_size
    return int(min(batch_size1, batch_size2))

def cal_fc2(fc2_input, fc2_w):
    out2 = np.dot(fc2_input, fc2_w)
    return out2

def cal_acc_from_fc2(fc2_input, fc2_w=None, quant=False, input_alpha=3.12, fc1_alpha=0.091, quant_level=255):
    if fc2_w is None:
        fc2_w = pd.read_csv(os.path.join('../simulated_data', 'fc2_weight.csv'), header=None).values
    inputs_labels = pd.read_csv(os.path.join('../simulated_data', 'input_labels.csv'), header=None).values
    targets = inputs_labels[:len(fc2_input), 0]
    if quant:
        fc2_input = ((fc2_input / (input_alpha / fc1_alpha)).clip(max=1) * quant_level).round()
    out2 = cal_fc2(fc2_input, fc2_w)
    predictions = np.argmax(out2, axis=1)
    accuracy_count = (predictions == targets).sum()
    accuracy = accuracy_count * 1.0 / len(targets)
    return accuracy

def show_offsets(offsets, columns=128, show=False):
    offsets = offsets.reshape(-1, columns)
    mean = offsets.mean(axis=0)
    std = offsets.std(axis=0)
    if show:
        plt.figure(figsize=(10, 8))
        plt.scatter(list(range(len(mean))), mean)
        plt.errorbar(list(range(len(mean))), mean, std)
        plt.title('%d colums offsets' % columns)
        plt.xlabel('colums')
        plt.ylabel('values')
        plt.show()
        plt.close()
    return mean

def show_sim_rram_out_hist(out1, out2, scatter_min=None, scatter_max=None, xlabel='s', ylabel='r', x_min=None, x_max=None):
    sim_data = out1.flatten()
    rram_data = out2.flatten()
    plt.figure(figsize=(10, 10))
    (left, width) = (0.1, 0.65)
    (bottom, height) = (0.1, 0.65)
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    plt.figure(1, figsize=(8, 8))
    nullfmt = NullFormatter()
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    axScatter.set_xlabel(xlabel)
    axScatter.set_ylabel(ylabel)
    axScatter.scatter(sim_data, rram_data, s=5, cmap=plt.cm.Spectral)
    if scatter_min:
        axScatter.set_ylim(bottom=scatter_min)
    if scatter_max:
        axScatter.set_ylim(top=scatter_max)
    if x_min:
        axScatter.set_xlim(left=x_min)
    if x_max:
        axScatter.set_xlim(right=x_max)
    binwidth = 0.25
    axHistx.hist(sim_data, histtype='step')
    axHisty.hist(rram_data, orientation='horizontal', histtype='step')
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    plt.show()
    plt.close()

def conv2d(x, w, stride=1, padding=1, b=None):
    x = np.pad(x, pad_width=[(0, 0), (0, 0), (padding, padding), (padding, padding)], mode='constant', constant_values=(0, 0))
    (n, ic, ih, iw) = x.shape
    (oc, _, kh, kw) = w.shape
    H_new = int(1 + (ih - kh) / stride)
    W_new = int(1 + (iw - kw) / stride)
    out = np.zeros((n, oc, H_new, W_new))
    for i in range(H_new):
        for j in range(W_new):
            x_windows = x[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            for k in range(n):
                for l in range(oc):
                    out[k, l, i, j] = np.sum(x_windows[k] * w[l])
                if b != None:
                    out[k, l, i, j] += b
    return out

def linear(x, w):
    out = np.dot(x, w)
    return out

def pooling(x, w_height=2, w_width=2, stride=2):
    (bs, cs, in_height, in_width) = x.shape
    out_height = int((in_height - w_height) / stride) + 1
    out_width = int((in_width - w_width) / stride) + 1
    out = np.zeros((bs, cs, out_height, out_width))
    for b in range(bs):
        for c in range(cs):
            for i in range(out_height):
                for j in range(out_width):
                    start_i = i * stride
                    start_j = j * stride
                    end_i = start_i + w_height
                    end_j = start_j + w_width
                    out[b, c, i, j] = np.max(x[b, c, start_i:end_i, start_j:end_j])
    return out

def l2_normalize(x):
    x = x.flatten()
    norm = np.linalg.norm(x, ord=2)
    if norm == 0:
        return x
    return x / norm

def rmse_cal(y_obs, y_pred):
    y_obs = y_obs.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(((y_obs - y_pred) ** 2).sum() / len(y_pred))
    return rmse

def auto_map_read_adjust_offsets_and_calculate(pt_weights_path, pt_inputs_path, pt_labels_path, tile=0, adc_range=32, offset_row_begin=8, save_root=None, map_num=1, map_bias_weight=True):
    pt_weights = np.load(pt_weights_path, allow_pickle=True).item()
    pt_inputs = np.load(pt_inputs_path, allow_pickle=True).item()
    pt_labels = np.load(pt_labels_path, allow_pickle=True)
    debug_img_number = 5
    for (key, value) in pt_inputs.items():
        pt_inputs[key] = pt_inputs[key]
    pt_labels = pt_labels
    rram_weights = {}
    for (key, value) in pt_weights.items():
        if 'conv' in key:
            rram_weights[key] = pt_weight_4dim_2_rram_weight_2dim(value)
        else:
            pt_weights[key] = pt_weights[key].T
            rram_weights[key] = pt_weights[key]
    conv2_weight = rram_weights['conv2_weight']
    conv3_weight = rram_weights['conv3_weight']
    fc_weight = rram_weights['fc_weight']
    rram_conv2_weight = trans_pt_weight_2_rram(conv2_weight, row_begin=offset_row_begin)
    row_nums = conv3_weight.shape[0]
    rram_conv3_1_weight = trans_pt_weight_2_rram(conv3_weight[:row_nums // 2], row_begin=offset_row_begin)
    rram_conv3_2_weight = trans_pt_weight_2_rram(conv3_weight[row_nums // 2:], row_begin=offset_row_begin)
    rram_fc_weight = trans_pt_weight_2_rram(fc_weight, row_begin=offset_row_begin)
    rram_inputs = {}
    for (key, value) in pt_inputs.items():
        rram_inputs[key] = trans_pt_data_2_rram(pt_inputs[key], voltage=int(1))
        if 'fc' in key:
            input_shape = rram_inputs[key].shape
            temp = np.zeros([input_shape[0], 320 * ((input_shape[1] - 1) // 320 + 1)])
            temp[:, offset_row_begin:offset_row_begin + input_shape[1]] = rram_inputs[key]
            rram_inputs[key] = temp.astype(np.uint8)
    repeat_map_num = map_num
    for i in range(repeat_map_num):
        auto_mapping_weight(rram_conv2_weight, tile=tile, xb=2)
        auto_mapping_weight(rram_conv3_1_weight, tile=tile, xb=4)
        auto_mapping_weight(rram_conv3_2_weight, tile=tile, xb=5)
        auto_mapping_weight(rram_fc_weight, tile=tile, xb=2)
        rram_read_weight_conv1 = check_conv_read_weight(pt_weights['conv2_weight'], pt_inputs['conv2_input_data'][:debug_img_number], tile=tile, xb=[2], row_begin=offset_row_begin, save_root=save_root, show_hist=True, show_acc=False, show_read_weight_hist=True, kernel_size=3, stride=1, padding=1)
    bias_input_value = auto_adjust_conv_offsets(rram_inputs['conv2_input_data'], [rram_conv2_weight], pt_inputs['conv2_input_data'], pt_weights['conv2_weight'], tile=tile, xb=[2], num_column=64, kernel_size=3, stride=1, padding=1, relu=False, shift_num=[4], adc_range=[adc_range], bias=True, bias_num=[2], offset_row_begin=offset_row_begin, map_bias_weight=map_bias_weight)
    conv2_bias_input_value = bias_input_value
    conv2_bias_input_value = bias_input_value = 268435455
    print()
    rram_output = calculate_Conv_one_layer(rram_inputs['conv2_input_data'][:debug_img_number], tile=tile, xb=[2], num_column=64, kernel_size=3, stride=1, padding=1, relu=False, shift_num=[4], adc_range=[adc_range], bias=True, bias_num=[2], bias_input_value_list=[[bias_input_value]])
    sim_output = conv2d(x=pt_inputs['conv2_input_data'][:debug_img_number], w=pt_weights['conv2_weight'], stride=1)
    show_sim_rram_out_hist(sim_output, rram_output, scatter_min=-136, scatter_max=134)
    mean = rram_output.mean()
    print()

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    
    (N, C, H, W) = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    
    (N, C, H, W) = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]

def calculate_std_noise(sw_out, hw_out):
    zero_indices = np.where(sw_out == 0)
    y_values_at_zero = hw_out[zero_indices]
    std_at_zero_indices = np.std(y_values_at_zero)
    hw_max = np.max(hw_out)
    hw_min = np.min(hw_out)
    hw_range = hw_max - hw_min
    std_noise = std_at_zero_indices / hw_range
    return std_noise

def std_cal(x, y, method=1, XB_num=0, shift_num=0):
    x = x.flatten()
    y = y.flatten()
    x_set = list(set(x))
    max_x = max(x_set)
    min_x = min(x_set)
    if method == 1:
        y_range = np.max(y) - np.min(y)
    elif method == 2:
        y_range = np.mean(y[np.where(x == max_x)]) - np.mean(y[np.where(x == min_x)])
    if method == 3:
        if not XB_num:
            raise ValueError('need XB_num setting for computing std_cal. ')
        temp_range = math.floor(XB_num * 127 * 17 / 2 ** shift_num) - math.floor(XB_num * -128 * 17 / 2 ** shift_num)
        y_range = min(511, temp_range)
        assert y_range > 0
    if y_range <= 0:
        y_range = 1.0
    std_list = []
    for data in x_set:
        ind = np.where(x == data)
        std_y = np.std(y[ind])
        std_list.append(std_y)
    return np.mean(std_list) / y_range

def linear_fit(x, y):
    x = x.flatten()
    y = y.flatten()
    fit_params = np.polyfit(x, y, 1)
    scale = fit_params[0]
    bias = fit_params[1]
    return (scale, bias)

def linear_fit_channel_conv(x, y, plot_flag=False):
    x_shape = x.shape
    x = np.transpose(x, (0, 2, 3, 1))
    x = x.reshape(-1, x_shape[1])
    y = np.transpose(y, (0, 2, 3, 1))
    y = y.reshape(-1, x_shape[1])
    (m, n) = x.shape
    scales = []
    biases = []
    for i in range(n):
        (scale, bias) = np.polyfit(x[:, i], y[:, i], 1)
        scales.append(scale)
        biases.append(bias)
        if plot_flag:
            plt.scatter(x[:, i], y[:, i])
            plt.plot(x[:, i], scale * x[:, i] + bias)
    if plot_flag:
        plt.show()
    return (scales, biases)

def linear_fit_channel_fc(x, y, plot_flag=False):
    x_shape = x.shape
    (m, n) = x.shape
    scales = []
    biases = []
    for i in range(n):
        (scale, bias) = np.polyfit(x[:, i], y[:, i], 1)
        scales.append(scale)
        biases.append(bias)
        if plot_flag:
            plt.scatter(x[:, i], y[:, i])
            plt.plot(x[:, i], scale * x[:, i] + bias)
    if plot_flag:
        plt.show()
    return (scales, biases)

def linear_fit_and_plot(x, y, title='conv19', yrange=[0, 1]):
    x = x.flatten()
    y = y.flatten()
    (slope, intercept) = np.polyfit(x, y, 1)
    print()
    fit_x = np.array([min(x), max(x)])
    fit_y = slope * fit_x + intercept
    plt.scatter(x, y, label='test value', s=10, alpha=0.3)
    plt.ylim(yrange)
    plt.plot(fit_x, fit_y, color='red', label='fit')
    plt.xlabel('SW result')
    plt.ylabel('HW result')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.close()
from e100_irtool.core.ir import load_ir, BaseIR
from e100_irtool.tools import flatten_layers
import pickle
import numpy as np
from .quant import *

def pickle_load(file, **kwargs):
    with open(file, 'rb') as f:
        return pickle.load(f, **kwargs)

def get_addr_record(mappings):
    row_start_addr_record = {}
    col_start_addr_record = {}
    for (k, v) in mappings.items():
        (r_index, h_index, w_index) = v.index
        if h_index in row_start_addr_record.keys():
            if h_index != 0:
                assert v.address[2] + row_start_addr_record[h_index - 1] == row_start_addr_record[h_index]
        elif h_index == 0:
            row_start_addr_record[h_index] = 0
        else:
            row_start_addr_record[h_index] = v.address[2] + row_start_addr_record[h_index - 1]
        if w_index in col_start_addr_record.keys():
            if w_index != 0:
                assert v.address[3] + col_start_addr_record[w_index - 1] == col_start_addr_record[w_index]
        elif w_index == 0:
            col_start_addr_record[w_index] = 0
        else:
            col_start_addr_record[w_index] = v.address[3] + col_start_addr_record[w_index - 1]
    return (row_start_addr_record, col_start_addr_record)

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

def trans_pt_weight_2_rram(pt_weight, bias_flag=False, pos_sa=5, neg_sa=5):
    bias_row_num = 8 * 2
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
    if bias_flag:
        mapping_weight[bias_row_num:bias_row_num + rram_weight.shape[0]] = sub_mapping_weight
    else:
        mapping_weight[:rram_weight.shape[0]] = sub_mapping_weight
    mapping_weight = mapping_weight.astype(np.uint8)
    return mapping_weight

def gen_array_weight(ir, weight_file=None, format='CHW', device_shape=(576, 128), device='C200', **kwargs):
    if isinstance(ir, str):
        ir = load_ir(ir)
    elif isinstance(ir, BaseIR):
        ir = ir
    else:
        raise ValueError(f'暂不支持ir类型:{type(ir)}')
    weight = None
    if isinstance(weight_file, str):
        weight = pickle_load(weight_file)
    elif isinstance(weight_file, dict):
        weight = weight_file
    else:
        raise ValueError(f'暂不支持weight 数据类型:{type(weight_file)}')
    array_data = {}
    systemc_weight_data = {}
    layers = ir.flatten_layers()
    if device.lower() == 'a111':
        pos_sa = 5
        neg_sa = 5
        if 'pos_sa' in kwargs.keys():
            pos_sa = kwargs['pos_sa']
        if 'neg_sa' in kwargs.keys():
            neg_sa = kwargs['neg_sa']
    for (name, layer) in layers.items():
        if layer.type in ['input', 'output', 'reuse']:
            continue
        mapping_info = None
        if device.lower() == 'c200':
            mapping_info = layer.c200_mapping_info
        elif device.lower() == 'a111':
            mapping_info = layer.a111_mapping_info
        else:
            raise ValueError(f'暂不支持设备 {device} ！！！')
        op_id = layer.op.op_id
        if op_id in ['matmul', 'fc', 'linear', 'conv2d', 'conv_transpose2d'] and mapping_info != None:
            weight_name = name + '.weight'
            assert weight_name in weight.keys(), f'{weight_name} 不存在 {weight.keys()} 中 !!!'
            col_repeat_num = mapping_info.col_repeat_num
            row_repeat_num = mapping_info.row_repeat_num
            wd = weight[weight_name]
            if op_id in ['conv2d', 'conv_transpose2d']:
                if op_id == 'conv_transpose2d':
                    wd = np.flip(wd, axis=(-1, -2))
                    wd = wd.transpose(1, 0, 2, 3)
                if format == 'HWC':
                    wd = wd.transpose(0, 2, 3, 1)
                    wd = wd.reshape(wd.shape[0], -1, wd.shape[3])
                elif format == 'CHW':
                    wd = wd.reshape(wd.shape[0], -1)
                    wd = wd.transpose(1, 0)
                    wd = np.tile(wd, [row_repeat_num, col_repeat_num])
                else:
                    raise ValueError(f'暂不支持数据格式{format}')
            elif op_id in ['matmul', 'fc', 'linear']:
                if format == 'HWC':
                    if len(layer.inputs) != 1:
                        raise ValueError('全连接层目前只支持一个动态的输入值，权重为静态！！！')
                    former_layer_name = layer.inputs[0].ref
                    former_layer = layers[former_layer_name]
                    if former_layer.op.op_id in ['reshape', 'flatten']:
                        in_channel = former_layer.inputs[0].channel
                        in_h = former_layer.inputs[0].height
                        in_w = former_layer.inputs[0].width
                        assert wd.shape[1] == in_channel * in_h * in_w
                        out_d = wd.shape[0]
                        wd = wd.reshape(out_d, in_channel, in_h, in_w)
                        wd = wd.transpose(0, 2, 3, 1)
                        wd = wd.reshape(out_d, -1)
                    elif former_layer.op.op_id == 'concat':
                        current_input_row_start = 0
                        transformed_fc_weight = []
                        for in_ in former_layer.inputs:
                            former_former_layer = layers[in_.ref]
                            if former_former_layer.op.op_id in ['reshape', 'flatten']:
                                assert len(former_former_layer.inputs) == 1
                                in_channel = former_former_layer.inputs[0].channel
                                in_h = former_former_layer.inputs[0].height
                                in_w = former_former_layer.inputs[0].width
                                row_num = in_channel * in_h * in_w
                                current_input_row_end = current_input_row_start + row_num
                                current_layer_fc_weight = wd[:, current_input_row_start:current_input_row_end] + 0
                                out_d = current_layer_fc_weight.shape[0]
                                current_layer_fc_weight = current_layer_fc_weight.reshape(out_d, in_channel, in_h, in_w)
                                current_layer_fc_weight = current_layer_fc_weight.transpose(0, 2, 3, 1)
                                current_layer_fc_weight = current_layer_fc_weight.reshape(out_d, -1)
                                transformed_fc_weight.append(current_layer_fc_weight)
                                current_input_row_start = current_input_row_end
                            else:
                                in_channel = in_.channel
                                in_h = in_.height
                                in_w = in_.width
                                row_num = in_channel * in_h * in_w
                                current_input_row_end = current_input_row_start + row_num
                                current_layer_fc_weight = wd[:, current_input_row_start:current_input_row_end] + 0
                                transformed_fc_weight.append(current_layer_fc_weight)
                                current_input_row_start = current_input_row_end
                        transformed_fc_weight = np.concatenate(transformed_fc_weight, axis=1)
                        assert transformed_fc_weight.shape == wd.shape
                        wd = transformed_fc_weight
                wd = wd.transpose(1, 0)
                wd = np.tile(wd, [row_repeat_num, col_repeat_num])
            (row_record, col_record) = get_addr_record(mapping_info.mappings)
            systemc_id = 0
            for (k, v) in mapping_info.mappings.items():
                (r_index, h_index, w_index) = v.index
                if h_index == 0:
                    input_row_start = 0
                else:
                    input_row_start = row_record[h_index]
                if w_index == 0:
                    input_col_start = 0
                else:
                    input_col_start = col_record[w_index]
                array_id = v.device
                current_row_num = v.address[2]
                current_col_num = v.address[3]
                array_row_start = v.address[0]
                array_col_start = v.address[1]
                array_row_end = array_row_start + current_row_num
                array_col_end = array_col_start + current_col_num
                input_row_end = input_row_start + current_row_num
                input_col_end = input_col_start + current_col_num
                if array_id not in array_data.keys():
                    array_data[array_id] = np.zeros(shape=device_shape)
                if op_id == 'conv2d' and format == 'HWC':
                    kernel_size = layer.op.kernel
                    assert current_row_num % (kernel_size ** 2 * row_repeat_num) == 0
                    assert input_row_start % (kernel_size ** 2 * row_repeat_num) == 0
                    assert kernel_size ** 2 == wd.shape[1]
                    current_channel_num = int(current_row_num / (kernel_size ** 2 * row_repeat_num))
                    input_channel_start = int(input_row_start / (kernel_size ** 2 * row_repeat_num))
                    input_channel_end = input_channel_start + current_channel_num
                    current_wd = wd[:, :, input_channel_start:input_channel_end] + 0
                    current_wd = current_wd.reshape(current_wd.shape[0], -1)
                    current_wd = current_wd.transpose(1, 0)
                    current_wd = np.tile(current_wd, [row_repeat_num, col_repeat_num])
                    if device.lower() == 'a111':
                        array_data[array_id] = trans_pt_weight_2_rram(current_wd[:, input_col_start:input_col_end], bias_flag=True, pos_sa=pos_sa, neg_sa=neg_sa)
                    elif device.lower() == 'c200':
                        array_data[array_id][array_row_start:array_row_end, array_col_start:array_col_end] = current_wd[:, input_col_start:input_col_end]
                    systemc_weight_data[name + f':{systemc_id}'] = current_wd[:, input_col_start:input_col_end].transpose(1, 0)
                else:
                    if device.lower() == 'a111':
                        array_data[array_id] = trans_pt_weight_2_rram(wd[input_row_start:input_row_end, input_col_start:input_col_end], pos_sa=pos_sa, neg_sa=neg_sa)
                    elif device.lower() == 'c200':
                        array_data[array_id][array_row_start:array_row_end, array_col_start:array_col_end] = wd[input_row_start:input_row_end, input_col_start:input_col_end]
                    systemc_weight_data[name + f':{systemc_id}'] = wd[input_row_start:input_row_end, input_col_start:input_col_end].transpose(1, 0)
                systemc_id += 1
    return (array_data, systemc_weight_data)
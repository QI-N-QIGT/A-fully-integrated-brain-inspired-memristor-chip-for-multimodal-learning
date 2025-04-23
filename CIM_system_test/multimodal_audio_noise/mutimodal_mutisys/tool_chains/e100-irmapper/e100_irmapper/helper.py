import math
from .device.c200 import *
from .device.a111 import *
from .device.CIMA import *
from model2ir.onnx2ir.converter import ConvertONNX
import numpy as np
from .fused_op.op import *
from e100_irtool.core.layer import make_layer, make_op
import copy
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import warnings

def get_max_time_layer(layer_time):
    
    a1 = sorted(layer_time.items(), key=lambda x: x[1], reverse=True)
    layer_name = a1[0][0]
    max_ = a1[0][1]
    return {layer_name: max_}

def split_node(node_shape, split_num):
    
    node_shape_split = {}
    for node_name in node_shape.keys():
        h = []
        w = []
        [W, H] = node_shape[node_name]
        [pda, psa, w_split, h_split] = split_num[node_name]
        h_i = h_split
        w_i = w_split
        _h = math.floor(H / h_i)
        _w = math.floor(W / w_i)
        for i in range(h_i):
            if H - _h > 0:
                h.append(_h)
                H = H - _h
            else:
                h.append(H)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        for k in range(pda):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name + '.' + str(k) + '.' + str(i) + '.' + str(j)] = [w[j], h[i]]
    return node_shape_split

def split_node_window_duplicate(node_info, xb_size, split_num):
    
    node_shape_split = {}
    for node_name in node_info.keys():
        h = []
        w = []
        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            [para_num, repeat_num, w_split, h_split] = split_num[node_name]
            kz = node_info[node_name]['kernel_size']
            stride = node_info[node_name]['stride']
            in_channel = node_info[node_name]['in_channel']
            out_channel = node_info[node_name]['out_channel']
            W = out_channel * repeat_num
            H = (kz + (repeat_num - 1) * stride) * in_channel * kz
            h_i = math.ceil(H / xb_size[1])
            w_i = math.ceil(W / xb_size[0])
            _h = math.floor(H / h_i)
            _w = math.floor(W / w_i)
            split_num[node_name] = [para_num, repeat_num, w_i, h_i]
        elif node_info[node_name]['op_type'] in ['matmul', 'fc', 'linear', 'fused_fc']:
            [para_num, w_split, h_split] = split_num[node_name]
            _h = h_split
            _w = w_split
        for i in range(h_i):
            if H - _h > 0:
                h.append(_h)
                H = H - _h
            else:
                h.append(H)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        for k in range(para_num):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name + '.' + str(k) + '.' + str(i) + '.' + str(j) + '_wd'] = [w[j], h[i]]
    return (node_shape_split, split_num)

def split_node_HWC(node_weight, node_info, para_num, XB_size, dmac_size=None, dmac_layer=None, device='rram-144k'):
    
    node_shape_split = {}
    node_split_num = {}
    for node_name in node_weight.keys():
        h = []
        w = []
        [W, H] = node_weight[node_name]
        array_size = XB_size
        if dmac_layer != None and node_name in dmac_layer:
            assert dmac_size != None
            array_size = dmac_size
        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            kernel_size = node_info[node_name]['kernel_size']
            if 'a111-tile' in device and kernel_size not in [1, 3, 7]:
                warnings.warn(f'当前层 {node_name} 不适合A111计算 , 采用数字模块(cpu)计算 !!!')
                continue
            in_channel = node_info[node_name]['in_channel']
            if 'a111-tile' in device and in_channel not in [4, 8, 16, 32, 64, 128, 256, 512]:
                warnings.warn(f'当前层 {node_name} 不适合A111计算 , 采用数字模块(cpu)计算 !!!')
                continue
            out_channel = node_info[node_name]['out_channel']
            if 'a111-tile' in device and out_channel not in [8, 16, 32, 64, 128, 256, 512]:
                warnings.warn(f'当前层 {node_name} 不适合A111计算 , 采用数字模块(cpu)计算 !!!')
                continue
            assert H % (kernel_size ** 2 * in_channel) == 0
            row_repeat_avg = H / (kernel_size ** 2 * in_channel)
            if H <= array_size[1]:
                h.append(H)
            else:
                h_temp = H
                t = 1
                split_ic = []
                while h_temp > array_size[1]:
                    if t == 1:
                        t += 1
                    else:
                        t *= 2
                    (max_split_channel_num, split_ic) = get_max_channel_split_num(in_channel, t)
                    if np.array(split_ic).mean() != max_split_channel_num:
                        warnings.warn(f'当前层 {node_name} 的输入通道拆分是非均匀的!!! 拆分后的通道数为: {split_ic} !!!')
                    h_temp = max_split_channel_num * kernel_size * kernel_size * row_repeat_avg
                    if t == in_channel:
                        raise ValueError('暂不支持kernel_size平方超过array_size_H !!!')
                assert split_ic != []
                for ic_ in split_ic:
                    h.append(ic_ * kernel_size * kernel_size * row_repeat_avg)
        else:
            h_split = math.ceil(H / array_size[1])
            h_i = h_split
            _h = math.floor(H / h_i)
            for i in range(h_i):
                if H - _h > 0:
                    h.append(_h)
                    H = H - _h
                else:
                    h.append(H)
        w_split = math.ceil(W / array_size[0])
        w_i = w_split
        _w = math.floor(W / w_i)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        if 'a111-tile' in device:
            if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
                if len(w) > 1 and len(h) > 1:
                    warnings.warn(f'A111-tile 卷积 不支持 行列 方向都切分！！！该层 {node_name} 片外计算!!!')
                    continue
                elif len(w) > 1:
                    warnings.warn(f'A111-tile 全连接 不支持 列 方向切分！！！该层 {node_name} 片外计算!!!')
                    continue
            if len(h) > 4 or len(w) > 4:
                warnings.warn(f'A111-tile 全连接单次计算 不支持 行或者列方向 拆分成大于4份 !!! 该层 {node_name} 片外计算!!!')
                continue
        repeat = 1
        if para_num != None:
            repeat = para_num[node_name][0]
        diff_array_repeat = repeat
        same_array_repeat = 1
        node_split_num[node_name] = [diff_array_repeat, same_array_repeat, len(w), len(h)]
        for k in range(repeat):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name + '.' + str(k) + '.' + str(i) + '.' + str(j)] = [w[j], h[i]]
    return (node_shape_split, node_split_num)

def get_max_channel_split_num(ic, split_num):
    
    t = math.ceil(ic / split_num)
    w = []
    rest = ic
    for i in range(split_num):
        temp = rest - t
        if temp > 0:
            w.append(t)
            rest = temp
        else:
            w.append(rest)
    return (np.array(w).max(), w)

def get_layer_ref(inputs, layer, ref):
    
    for i in inputs:
        ref_name = i.ref
        if ':' in ref_name:
            ref_name = ref_name.split(':')[0]
        if 'graph_input' in ref_name:
            ref.append(ref_name)
        elif layer[ref_name].type == 'reuse':
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d', 'linear', 'matmul', 'fc', 'fused_fc']:
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['constant', 'split', 'add', 'fused_add', 'fused_concat', 'concat']:
            ref.append(ref_name)
        else:
            get_layer_ref(layer[ref_name].inputs, layer, ref)

def get_conv_shape(op_info):
    
    kernel_size = op_info.kernel
    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    bias = False
    if bias:
        unroll_shape_h = kernel_size * kernel_size * in_channel + 1
    else:
        unroll_shape_h = kernel_size * kernel_size * in_channel
    unroll_shape_w = out_channel
    return [unroll_shape_w, unroll_shape_h]

def get_linear_shape(op_info):
    
    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    bias = False
    if bias:
        unroll_shape_h = in_channel + 1
    else:
        unroll_shape_h = in_channel
    unroll_shape_w = out_channel
    return [unroll_shape_w, unroll_shape_h]

def get_conv_info(layer):
    
    if layer.inputs[0].dtype != None:
        intype = layer.inputs[0].dtype
    else:
        intype = 8
    if layer.outputs[0].dtype != None:
        outtype = layer.outputs[0].dtype
    else:
        outtype = 8
    kz = layer.op.kernel
    stride = layer.op.stride
    padding = layer.op.padding
    out_height = layer.outputs[0].height
    out_width = layer.outputs[0].width
    in_channel = layer.op.in_channel
    out_channel = layer.op.out_channel
    copy_const = out_height
    calc_num = out_height * out_width
    op_type = layer.op.op_id
    in_data_len = (layer.inputs[0].height + 2 * padding) * (layer.inputs[0].width + 2 * padding) * layer.inputs[0].channel
    out_data_len = out_height * out_width * out_channel
    input_shape = [layer.inputs[0].height, layer.inputs[0].width]
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, kernel_size=kz, stride=stride, calc_num=calc_num, in_precision=intype, out_precision=outtype, copy_constraint=copy_const, in_data_len=in_data_len, out_data_len=out_data_len, input_shape=input_shape)

def get_linear_info(layer):
    
    if layer.inputs[0].dtype != None:
        intype = layer.inputs[0].dtype
    else:
        intype = 8
    if layer.outputs[0].dtype != None:
        outtype = layer.outputs[0].dtype
    else:
        outtype = 8
    in_channel = layer.op.in_channel
    out_channel = layer.op.out_channel
    calc_num = 1
    kz = 1
    stride = 1
    op_type = layer.op.op_id
    copy_const = 1
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, kernel_size=kz, stride=stride, calc_num=calc_num, in_precision=intype, out_precision=outtype, copy_constraint=copy_const, in_data_len=in_channel, out_data_len=out_channel)

def get_split_concat_info(layer):
    
    in_channel = []
    input_shape = []
    for in_ in layer.inputs:
        in_channel.append(in_.channel)
        input_shape.append([in_.height, in_.width])
    out_channel = []
    if layer.outputs != None:
        for out_ in layer.outputs:
            out_channel.append(out_.channel)
    op_type = layer.op.op_id
    axis = layer.op.axis
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, axis=axis, input_shape=input_shape)

def get_add_info(layer):
    
    in_channel = []
    input_shape = []
    for in_ in layer.inputs:
        in_channel.append(in_.channel)
        input_shape.append([in_.height, in_.width])
    out_channel = []
    for out_ in layer.outputs:
        out_channel.append(out_.channel)
    op_type = layer.op.op_id
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, input_shape=input_shape)

def list_reverse(list_):
    
    len_ = len(list_)
    reverse_list = []
    for i in range(len_ - 1, -1, -1):
        reverse_list.append(list_[i])
    return reverse_list

def make_mapped_ir(ir, split_info, place_info, copy_info=None, cpu_layer=None, calc_info=None, device='rram-144k', runtime='simulation', **kwargs):
    
    for (name, layer) in ir.iter_layers():
        if layer.type == 'op':
            if cpu_layer != None and name in cpu_layer:
                continue
            if layer.op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d'] or layer.op.op_id in ['linear', 'matmul', 'fc', 'fused_fc']:
                if 'rram-144k' in device:
                    split_num = split_info[name]
                    if copy_info != None and name in copy_info.keys():
                        col_repeat_num = copy_info[name][1]
                        row_repeat_num = copy_info[name][0]
                    else:
                        col_repeat_num = 1
                        row_repeat_num = 1
                    assert len(split_num) == 4
                    if isinstance(runtime, dict):
                        if name in runtime.keys():
                            runtime_ = runtime[name]
                        else:
                            runtime_ = 'simulation'
                    elif isinstance(runtime, str):
                        runtime_ = runtime
                    else:
                        raise ValueError(f'暂不支持 runtime 类型 {type(runtime)} !!!')
                    layer.c200_mapping_info = C200MappingInfo(col_split_num=split_num[2], row_split_num=split_num[3], col_repeat_num=col_repeat_num, row_repeat_num=row_repeat_num, para_same_array=split_num[1], para_diff_array=split_num[0], runtime=runtime_, mappings=place_info[name])
                    if calc_info == None:
                        layer.c200_calc_info = C200CalcInfo().clone()
                    elif isinstance(calc_info, dict):
                        if name not in calc_info.keys():
                            Warning(f'layer{name}未设置运行参数，采用默认配置！！！')
                            layer.c200_calc_info = C200CalcInfo().clone()
                        else:
                            layer.c200_calc_info = calc_info[name]
                    else:
                        layer.c200_calc_info = calc_info.clone()
                elif 'a111-tile' in device:
                    if name not in split_info.keys():
                        continue
                    split_num = split_info[name]
                    if copy_info != None:
                        col_repeat_num = copy_info[name][1]
                        row_repeat_num = copy_info[name][0]
                    else:
                        col_repeat_num = 1
                        row_repeat_num = 1
                    input_buffer_addr = kwargs['input_buffer_addr'][name]
                    output_buffer_addr = kwargs['output_buffer_addr'][name]
                    in_buf_type = kwargs['in_buf_type'][name]
                    out_buf_type = kwargs['out_buf_type'][name]
                    layer.a111_mapping_info = A111MappingInfo(col_split_num=split_num[2], row_split_num=split_num[3], col_repeat_num=col_repeat_num, row_repeat_num=row_repeat_num, para_diff_array=split_num[1], input_buffer_addr=input_buffer_addr, output_buffer_addr=output_buffer_addr, in_buf_type=in_buf_type, out_buf_type=out_buf_type, mappings=place_info[name])
                    if calc_info == None:
                        layer.a111_calc_info = A111CalcInfo().clone()
                    elif isinstance(calc_info, dict):
                        if name not in calc_info.keys():
                            Warning(f'layer{name}未设置运行参数，采用默认配置！！！')
                            layer.a111_calc_info = A111CalcInfo().clone()
                        else:
                            layer.a111_calc_info = calc_info[name]
                    else:
                        layer.a111_calc_info = calc_info.clone()
                elif 'cima' in device:
                    split_num = split_info[name]
                    if copy_info != None:
                        col_repeat_num = copy_info[name][1]
                        row_repeat_num = copy_info[name][0]
                    else:
                        col_repeat_num = 1
                        row_repeat_num = 1
                    in_line_buffer_addr = kwargs['in_line_buffer_addr'][name]
                    credit_len = kwargs['credit_len'][name]
                    layer.CIMA_mapping_info = CIMAMappingInfo(col_split_num=split_num[2], row_split_num=split_num[3], col_repeat_num=col_repeat_num, row_repeat_num=row_repeat_num, para_diff_array=split_num[0], in_line_buffer_addr=in_line_buffer_addr, credit_len=credit_len, mappings=place_info[name])
                    if calc_info == None:
                        layer.CIMA_calc_info = CIMACalcInfo().clone()
                    elif isinstance(calc_info, dict):
                        if name not in calc_info.keys():
                            Warning(f'layer{name}未设置运行参数，采用默认配置！！！')
                            layer.CIMA_calc_info = CIMACalcInfo().clone()
                        else:
                            layer.CIMA_calc_info = calc_info[name]
                    else:
                        layer.CIMA_calc_info = calc_info.clone()
                else:
                    raise ValueError(f'暂不支持的device : {device} !!!')
            elif 'cima' in device and layer.op.op_id in ['add', 'maxpool2d', 'avgpool2d', 'concat', 'split', 'fused_add', 'fused_concat', 'identity', 'global_avg_pool2d', 'silu', 'resize']:
                in_line_buffer_addr = kwargs['in_line_buffer_addr'][name]
                credit_len = kwargs['credit_len'][name]
                layer.CIMA_mapping_info = CIMAMappingInfo(col_split_num=None, row_split_num=None, col_repeat_num=None, row_repeat_num=None, para_diff_array=None, in_line_buffer_addr=in_line_buffer_addr, credit_len=credit_len, mappings=place_info[name])
    if 'a111-tile' in device:
        assert kwargs['tile_all'] != []
        tile_occupied_xb = kwargs['tile_occupied_xb']
        for tile in kwargs['tile_all']:
            first_node_name = tile[0]
            device_name = place_info[first_node_name][0].device
            tile_name = '.'.join(device_name.split('.')[0:3])
            (npu_index, rsv, tile_index) = tile_name.split('.')
            for layer_name in tile:
                index = tile.index(layer_name)
                if ir.layers[layer_name].op.op_id in ['fused_conv2d'] and ir.layers[layer_name].op.pool != None:
                    if index == 0:
                        ir.devices[npu_index].devices[tile_index].info.pool0_en = 1
                    elif index == 1:
                        ir.devices[npu_index].devices[tile_index].info.pool1_en = 1
                    elif index == 2:
                        ir.devices[npu_index].devices[tile_index].info.pool2_en = 1
                    else:
                        ir.devices[npu_index].devices[tile_index].info.pool3_en = 1
            ir.devices[npu_index].devices[tile_index].info.op_list = tile
            tile_occupied_xb_list = tile_occupied_xb[tile_name]
            if tile_occupied_xb_list in [[2], [2, 2], [2, 2, 2], [2, 2, 2, 2]]:
                ir.devices[npu_index].devices[tile_index].info.tile_mode = 3
            elif tile_occupied_xb_list in [[4], [4, 4]]:
                ir.devices[npu_index].devices[tile_index].info.tile_mode = 1
            elif tile_occupied_xb_list in [[4, 2], [4, 2, 2], [2, 4]]:
                ir.devices[npu_index].devices[tile_index].info.tile_mode = 2
            else:
                raise ValueError(f'不支持 {tile_occupied_xb_list} 排布方式!!!')
    return ir

def make_device_ir(ir, device=None):
    
    if ir.devices != None:
        raise ValueError(f'已存在devices: {ir.devices.keys()} !!!')
    if device != None:
        if isinstance(device, list):
            for dev_ in device:
                dev_copy = copy.deepcopy(dev_)
                if 'num' in dev_.keys():
                    dev_copy.pop('name')
                    dev_copy.pop('kind')
                    dev_copy.pop('num')
                    ir.add_device(dev_['name'], dev_['kind'], number=dev_['num'], **dev_copy)
                else:
                    dev_copy.pop('name')
                    dev_copy.pop('kind')
                    ir.add_device(dev_['name'], dev_['kind'], **dev_copy)
        elif isinstance(device, dict):
            dev_copy = copy.deepcopy(device)
            dev_copy.pop('name')
            dev_copy.pop('kind')
            dev_copy.pop('num')
            ir.add_device(device['name'], device['kind'], number=device['num'], **dev_copy)
        else:
            raise TypeError(f'device type {type(device)} error!!!')
        return ir
    else:
        raise ValueError('无device信息！！！')

def make_onnx_ir(onnx_file, return_weight=False):
    
    t = ConvertONNX(onnx_file)
    if return_weight:
        return (t.ir, t.model_parser.weight_numpy)
    else:
        return t.ir

def make_node_id(split_nodes):
    node_id = {}
    for i in range(len(split_nodes)):
        if len(split_nodes[i]) != 1:
            raise ValueError(f'暂不支持多层放在同一个XB中！！！{split_nodes}')
        else:
            node_name = list(split_nodes[i][0].keys())[0]
            node_id[node_name] = i
    return node_id

def fuse_op(ir, relu_fuse=False, pool_fuse=False, split_fuse=False, silu_fuse=False):
    fused_op_all = {}
    next_layer_dict = get_next_layer(ir.layers)
    layers_info = ir.layers
    for (name, layer) in layers_info.items():
        can_fuse_relu = False
        can_fuse_pool = False
        can_fuse_split = False
        can_fuse_silu = False
        if layer.type == 'op' and layer.op.op_id in ['conv2d', 'matmul', 'linear', 'fc', 'add', 'concat', 'fused_add', 'fused_concat']:
            next_layers = next_layer_dict[name]
            if len(next_layers) == 1:
                nl = next_layers[0]
                if layers_info[nl].type == 'op':
                    if relu_fuse and layers_info[nl].op.op_id in ['relu']:
                        can_fuse_relu = True
                    if pool_fuse and layers_info[nl].op.op_id in ['max_pool2d', 'maxpool2d', 'global_avg_pool2d']:
                        can_fuse_pool = True
                    if split_fuse and layers_info[nl].op.op_id in ['split']:
                        if layer.op.op_id in ['fused_add', 'fused_concat'] and layer.op.split != None:
                            continue
                        can_fuse_split = True
                    if silu_fuse and layers_info[nl].op.op_id in ['silu']:
                        can_fuse_silu = True
            if can_fuse_pool or can_fuse_relu or can_fuse_split or can_fuse_silu:
                current_op_info = layer.op
                if current_op_info.op_id in ['conv2d', 'conv_transpose2d']:
                    kernel = current_op_info.kernel
                    in_channel = current_op_info.in_channel
                    out_channel = current_op_info.out_channel
                    stride = current_op_info.stride
                    padding = current_op_info.padding
                    bias = current_op_info.bias
                    fused_op_obj = fused_conv2d(kernel=kernel, in_channel=in_channel, out_channel=out_channel, stride=stride, padding=padding, bias=bias).clone()
                elif current_op_info.op_id in ['matmul', 'linear', 'fc']:
                    in_channel = current_op_info.in_channel
                    out_channel = current_op_info.out_channel
                    bias = current_op_info.bias
                    fused_op_obj = fused_fc(in_channel=in_channel, out_channel=out_channel, bias=bias).clone()
                elif current_op_info.op_id in ['add', 'fused_add']:
                    fused_op_obj = fused_add().clone()
                elif current_op_info.op_id in ['concat', 'fused_concat']:
                    attr_ = dict(axis=current_op_info.axis)
                    fused_op_obj = fused_concat(**attr_).clone()
                for nl in next_layers:
                    if nl in fused_op_all.keys():
                        raise ValueError(f'该层 {nl} 已被 {fused_op_all[nl]} 融合!!!')
                    next_layer_op_info = layers_info[nl].op
                    if current_op_info.op_id in ['add', 'concat'] and len(next_layers) == 1 and (next_layer_op_info.op_id == 'split'):
                        fused_op_obj.split = next_layer_op_info
                        fused_op_all[nl] = name
                    elif next_layer_op_info.op_id in ['relu']:
                        fused_op_obj.relu = next_layer_op_info
                        fused_op_all[nl] = name
                    elif next_layer_op_info.op_id in ['max_pool2d', 'maxpool2d', 'global_avg_pool2d']:
                        fused_op_obj.pool = next_layer_op_info
                        fused_op_all[nl] = name
                    elif current_op_info.op_id in ['conv2d', 'matmul', 'linear', 'fc'] and next_layer_op_info.op_id in ['silu']:
                        fused_op_obj.silu = next_layer_op_info
                        fused_op_all[nl] = name
                fused_layer_inputs = layer.inputs
                fused_layer_weights = layer.weights
                fused_layer_outputs = layers_info[nl].outputs
                fused_layer = make_layer(op=fused_op_obj, inputs=fused_layer_inputs, weights=fused_layer_weights, outputs=fused_layer_outputs)
                ir.layers[name] = fused_layer
    for (fused_op_name, replaced_op_name) in fused_op_all.items():
        next_layers = next_layer_dict[fused_op_name]
        for nl in next_layers:
            for i in ir.layers[nl].inputs:
                if ':' in i.ref:
                    ref_name = i.ref.split(':')
                    if ref_name[0] == fused_op_name:
                        i.ref = f'{replaced_op_name}:{ref_name[1]}'
                elif i.ref == fused_op_name:
                    i.ref = replaced_op_name
        ir.layers.pop(fused_op_name)
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    return ir

def get_device_ip(device):
    device_ip = {}
    if isinstance(device, list):
        for dev_ in device:
            assert isinstance(dev_, dict)
            if 'ip' in dev_.keys():
                device_ip[dev_['name']] = dev_['ip']
    elif isinstance(device, dict):
        if 'ip' in dev_.keys():
            device_ip[dev_['name']] = dev_['ip']
    else:
        raise ValueError(f'不支持device的类型 {type(device)}!!!')
    return device_ip

def gen_c200_mapping_info(placed_nodes, hardware_name, window_copy=False):
    node_mapping_info = {}
    for index in range(len(placed_nodes)):
        device_ref = hardware_name[index]
        for node_addr in placed_nodes[index]:
            key = list(node_addr.keys())[0]
            value = list(node_addr.values())[0]
            name_ = key.split('.')
            node_name = name_[0]
            if window_copy:
                index_ = [int(name_[1]), int(name_[2]), int(name_[3].split('_')[0])]
            else:
                index_ = [int(name_[1]), int(name_[2]), int(name_[3])]
            if node_name not in node_mapping_info.keys():
                node_mapping_info[node_name] = []
            mapping_info = C200DeviceMappingInfo(index=index_, device=device_ref, address=value)
            node_mapping_info[node_name].append(mapping_info)

def get_pre_layer(layers):
    
    prefix_layer = {}
    for (name, layer) in layers.items():
        if layer.type not in ['input']:
            prefix_layer[name] = []
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                continue
            for i in layer.inputs:
                if 'graph_input' not in i.ref:
                    ref = i.ref
                    if ':' in ref:
                        ref = ref.split(':')[0]
                    pre_layer = layers[ref]
                    if pre_layer.type == 'op' and pre_layer.op.op_id in ['flatten', 'reshape']:
                        for j in pre_layer.inputs:
                            prefix_layer[name].append(j.ref)
                    else:
                        prefix_layer[name].append(ref)
                else:
                    prefix_layer[name].append(i.ref)
    return prefix_layer

def get_next_layer(layers):
    
    next_layer = {}
    pre_layer = get_pre_layer(layers)
    for (k, v) in pre_layer.items():
        if layers[k].type == 'op' and layers[k].op.op_id in ['flatten']:
            continue
        for name in v:
            if name not in next_layer.keys():
                next_layer[name] = []
            next_layer[name].append(k)
    return next_layer

def draw_square_mesh(grid_size, square_size, arrow_size, text_dict, value_dict, dmac_info, save_fig=None):
    (fig, ax) = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    def draw_text_upon_dash(text, position, x, y, square_size, arrow_size):
        if position == 'top':
            text_x = x + square_size / 2
            text_y = y - arrow_size / 2
        elif position == 'bottom':
            text_x = x + square_size / 2
            text_y = y + square_size + arrow_size / 2
        elif position == 'left':
            text_x = x - arrow_size / 2
            text_y = y + square_size / 2
        elif position == 'right':
            text_x = x + square_size + arrow_size / 2
            text_y = y + square_size / 2
        if position == 'left':
            text_y = text_y - 10
        elif position == 'right':
            text_y = text_y + 10
        elif position == 'top':
            text_x = text_x - 10
        elif position == 'bottom':
            text_x = text_x + 10
        ax.text(text_x, text_y, text, fontsize=8, ha='center', va='center', color='purple')

    def draw_text_in_square(text, position, x, y, square_size):
        small_square_size = square_size / 4
        text_x = None
        text_y = None
        if position == 'top':
            text_x = x + square_size / 2
            text_y = y + square_size - small_square_size / 2
        elif position == 'bottom':
            text_x = x + square_size / 2
            text_y = y + small_square_size / 2
        elif position == 'left':
            text_x = x + small_square_size / 2
            text_y = y + square_size / 2
        elif position == 'right':
            text_x = x + square_size - small_square_size / 2
            text_y = y + square_size / 2
        if text_x is not None and text_y is not None:
            ax.text(text_x, text_y, text, fontsize=6, ha='center', va='center', color='red', fontdict={'weight': 'bold'})
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * (square_size + arrow_size)
            y = i * (square_size + arrow_size)
            ax.text(x + square_size / 2, y + square_size / 2, f'Node[{i}][{j}]', fontsize=7, ha='center', va='center')
            if (i, j) in dmac_info.keys():
                ln = dmac_info[i, j]
                ax.text(x + 25, y + square_size / 2 + 10, f'{ln}(DMAC)', fontsize=5, ha='center', va='center', color='red', fontdict={'weight': 'bold'})
            square = plt.Rectangle((x, y), square_size, square_size, linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(square)
            small_square_size = square_size / 4
            small_x = x + square_size / 2 - small_square_size / 2
            small_y_top = y + square_size - small_square_size
            small_y_bottom = y
            small_x_left = x
            small_x_right = x + square_size - small_square_size
            small_square_top = plt.Rectangle((small_x, small_y_top), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_top)
            small_square_bottom = plt.Rectangle((small_x, small_y_bottom), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_bottom)
            small_square_left = plt.Rectangle((small_x_left, y + square_size / 2 - small_square_size / 2), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_left)
            small_square_right = plt.Rectangle((small_x_right, y + square_size / 2 - small_square_size / 2), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_right)
            if j < grid_size - 1:
                right_arrow = plt.Arrow(x + square_size, y + square_size / 2 + 2, arrow_size, 0, color='black', width=arrow_size / 2)
                ax.add_patch(right_arrow)
                left_arrow = plt.Arrow(x + square_size + arrow_size, y + square_size / 2 - 2, -arrow_size, 0, color='black', width=arrow_size / 2)
                ax.add_patch(left_arrow)
            if i < grid_size - 1:
                up_arrow = plt.Arrow(x + square_size / 2 + 2, y + square_size, 0, arrow_size, color='black', width=arrow_size / 2)
                ax.add_patch(up_arrow)
                down_arrow = plt.Arrow(x + square_size / 2 - 2, y + square_size + arrow_size, 0, -arrow_size, color='black', width=arrow_size / 2)
                ax.add_patch(down_arrow)
            if (i, j) in text_dict:
                text_position = text_dict[i, j]
                for value in text_position:
                    (text, position) = value
                    draw_text_in_square(text, position, x, y, square_size)
            if (i, j) in value_dict:
                text_position = value_dict[i, j]
                for value in text_position:
                    (text, position) = value
                    draw_text_upon_dash(text, position, x, y, square_size, arrow_size)
    ax.set_xlim(0, grid_size * (square_size + arrow_size))
    ax.set_ylim(grid_size * (square_size + arrow_size), 0)
    plt.tight_layout()
    if save_fig:
        import os
        if '\\' in save_fig:
            path = save_fig.split('\\')
            path = '\\'.join(path[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
        plt.savefig(save_fig)
    plt.close()

def draw_mesh_fig(map, mesh_width=4, save_fig_path='1.svg'):
    record_io_workload = map.place.record_io_workload
    draw_info = {}
    dash_info = {}
    dmac_info = {}
    draw_loc = {0: 'bottom', 1: 'right', 2: 'top', 3: 'left'}
    node_mapping_info = map.place.node_mapping_info_list
    for (name, addr) in node_mapping_info.items():
        name_ = name.split('.')[0]
        node_id = int(addr.split('.')[1].split(':')[1])
        figure_id = (node_id // mesh_width, node_id % mesh_width)
        if 'dmac' in addr:
            dmac_info[figure_id] = name_
            continue
        if figure_id not in draw_info.keys():
            draw_info[figure_id] = []
        location = draw_loc[int(addr.split('.')[2].split(':')[1])]
        if '[' in addr.split('.')[-1] and ',' in addr.split('.')[-1]:
            name_ = 'PE_TASK'
            pe_number = addr.split('.')[3].split(':')[1]
            if '-' in pe_number:
                p1 = int(pe_number.split('-')[0])
                p2 = int(pe_number.split('-')[1])
                pe_num = p2 - p1 + 1
            else:
                pe_num = 1
        else:
            name_ = 'Others'
            pe_num = 1
        draw_info[figure_id].append((name_, location, pe_num))
    draw_info_pe_task_num = {}
    for (k, v) in draw_info.items():
        draw_info_pe_task_num[k] = []
        count_loc = {'bottom': 0, 'right': 0, 'top': 0, 'left': 0}
        for (n, l, pe_num) in v:
            if n == 'PE_TASK':
                count_loc[l] += pe_num
        for (d, val) in count_loc.items():
            draw_info_pe_task_num[k].append((val, d))
    value_list = []
    for (loc, value) in record_io_workload.items():
        t1 = list(loc.split('-')[0])
        node1_id = [int(t1[1]), int(t1[4])]
        t2 = list(loc.split('-')[1])
        node2_id = [int(t2[1]), int(t2[4])]
        figure_id = (int(t1[1]), int(t1[4]))
        position = None
        if node2_id[1] - node1_id[1] > 0:
            position = 'right'
        elif node2_id[1] - node1_id[1] < 0:
            position = 'left'
        elif node2_id[0] - node1_id[0] > 0:
            position = 'bottom'
        elif node2_id[0] - node1_id[0] < 0:
            position = 'top'
        if figure_id not in dash_info.keys():
            dash_info[figure_id] = []
        dash_info[figure_id].append((str(value), position))
        value_list.append(value)
    draw_square_mesh(mesh_width, 50, 10, draw_info_pe_task_num, dash_info, dmac_info, save_fig_path)

def replace_op(ir):
    layers = ir.layers
    next_layers_dict = get_next_layer(layers)
    layers_recurrent = copy.deepcopy(layers)
    for (layer_name, layer_info) in layers_recurrent.items():
        current_layer_info = layer_info
        next_sigmoid = False
        next_mul = False
        sigmoid_out_mul = False
        if current_layer_info.type == 'op' and current_layer_info.op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc']:
            next_layers = next_layers_dict[layer_name]
            for i in next_layers:
                if layers[i].type == 'op':
                    if layers[i].op.op_id == 'sigmoid':
                        next_sigmoid = True
                        sigmoid_next_layer = next_layers_dict[i]
                        if len(sigmoid_next_layer) == 1:
                            snl = sigmoid_next_layer[0]
                            if layers[snl].type == 'op' and layers[snl].op.op_id == 'mul':
                                sigmoid_out_mul = True
                    if layers[i].op.op_id == 'mul':
                        next_mul = True
            if next_sigmoid and next_mul and sigmoid_out_mul:
                op_ = make_op('silu')
                in_height = current_layer_info.outputs[0].height
                in_width = current_layer_info.outputs[0].width
                in_channel = current_layer_info.outputs[0].channel
                input_ = [dict(ref=layer_name, channel=in_channel, width=in_width, height=in_height)]
                silu_name = 'Silu_' + layer_name
                output_ = [dict(channel=in_channel, width=in_width, height=in_height)]
                ir.add_layer(silu_name, op=op_, inputs=input_, outputs=output_)
                for i in next_layers:
                    if layers[i].type == 'op':
                        if layers[i].op.op_id == 'sigmoid':
                            ir.layers.pop(i)
                        elif layers[i].op.op_id == 'mul':
                            mul_next_layer = next_layers_dict[i]
                            for mnl in mul_next_layer:
                                for in_ in ir.layers[mnl].inputs:
                                    if in_.ref == i:
                                        in_.ref = silu_name
                            ir.layers.pop(i)
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    return ir
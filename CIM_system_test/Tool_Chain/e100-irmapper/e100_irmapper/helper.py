import math
from .device.c200 import *
from .device.a111 import *
from .device.CIMA import *

from model2ir.onnx2ir.converter import ConvertONNX
import numpy as np
from .self_defined_op.fused_op import *
from .self_defined_op.cima_op import *

from e100_irtool.core.layer import make_layer, make_op, DataDef

import copy
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import warnings

def get_max_time_layer(layer_time):
    '''
    input: 
        layer_time: 字典形式，{'layer_name':layer_time}
    return: 
        时间最长的一层，字典形式。如果有多个层时间一样，则返回遍历到的第一层。
    '''
    
    a1 = sorted(layer_time.items(),key = lambda x:x[1],reverse = True)
    layer_name = a1[0][0]
    max_ = a1[0][1]
    return {layer_name : max_}

def split_node(node_shape,split_num):
    '''
    input: 
        node_shape: 字典形式，{'node_name':[w,h]}
        split_num: 拆分的份数，字典形式，{'node_name':[para_diff_array,para_same_array,w_num,h_num]},
                    列表的元素分别表示 不同阵列并行复制的次数，同一阵列并行复制的次数，列方向以及行方向拆分的次数
    return:
        根据拆分份数，返回拆分之后的字典{'node_name_new':[w_split,h_split]},'node_name_new' = 'node_name' + '.repeat_index' + '.h_index' + '.w_index'
    '''
    node_shape_split = {}
    for node_name in node_shape.keys():
        h = []
        w = []
        [W,H] = node_shape[node_name]
        [pda,psa,w_split,h_split] = split_num[node_name]
        h_i = h_split 
        w_i = w_split 
        _h = math.floor(H/h_i)
        _w = math.floor(W/w_i)
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
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)] = [w[j],h[i]]
    return node_shape_split

def split_node_window_duplicate(node_info,xb_size,split_num):
    '''
    input: 
        node_shape: 字典形式，{'node_name':[w,h]}
        split_num: 拆分的份数，字典形式，{'node_name':[parallel_num,copy_num,w_num,h_num]},列表的元素分别表示复制的次数，列方向以及行方向拆分的次数
    return:
        根据拆分份数，返回拆分之后的字典{'node_name_new':[w_split,h_split]},'node_name_new' = 'node_name' + '.parallel_index' + '.h_index' + '.w_index'
    '''
    node_shape_split = {}
    for node_name in node_info.keys():
        
        h = []
        w = []
        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            [para_num,repeat_num,w_split,h_split] = split_num[node_name]
            kz = node_info[node_name]['kernel_size']
            stride = node_info[node_name]['stride']
            in_channel = node_info[node_name]['in_channel']
            out_channel = node_info[node_name]['out_channel']
            # cc = node_info[node_name]['copy_constraint']
            W = out_channel * repeat_num
            H = ( kz  + (repeat_num - 1) * stride ) * in_channel * kz
            h_i =  math.ceil(H /  xb_size[1])
            w_i =  math.ceil(W /  xb_size[0])
            
            _h = math.floor(H/h_i)
            _w = math.floor(W/w_i)
            split_num[node_name] = [para_num,repeat_num,w_i,h_i]
            
        elif node_info[node_name]['op_type'] in ['matmul','fc','linear', 'fused_fc']:
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
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)+'_wd'] = [w[j],h[i]]
    
    return node_shape_split,split_num

def split_node_HWC(node_weight,node_info,para_num,XB_size,dmac_size=None,dmac_layer= None,device='rram-144k'):
    '''
    input: 
        node_shape: 字典形式，{'node_name':[w,h]}
        node_info: 字典形式，{'node_name':node_info}
        para_num: 字典形式，{'node_name':para_num}
        XB_Size: 列表形式[w,h]
    return:
        根据拆分份数，返回拆分之后的字典{'node_name_new':[w_split,h_split]},'node_name_new' = 'node_name' + '.repeat_index' + '.h_index' + '.w_index'
    '''
    
    node_shape_split = {}
    node_split_num = {}

    for node_name in node_weight.keys():
        
        h = []
        w = []
        [W, H] = node_weight[node_name]
        array_size = XB_size
        
        # 如果有一些层需要放在DMAC上，则需要根据DMAC的size来进行权重拆分
        if dmac_layer!= None and node_name in dmac_layer:
            assert dmac_size != None
            array_size = dmac_size
            
                
        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            kernel_size = node_info[node_name]['kernel_size']
            if 'a111-tile' in device and kernel_size not in [1, 3, 7]:
                warnings.warn(f"当前层 {node_name} 不适合A111计算 , 采用数字模块(cpu)计算 !!!")
                continue
            in_channel = node_info[node_name]['in_channel']
            if 'a111-tile' in device and in_channel not in [4, 8, 16, 32, 64, 128, 256, 512]:
                warnings.warn(f"当前层 {node_name} 不适合A111计算 , 采用数字模块(cpu)计算 !!!")
                continue
            out_channel = node_info[node_name]['out_channel']
            if 'a111-tile' in device and out_channel not in [8, 16, 32, 64, 128, 256, 512]:
                warnings.warn(f"当前层 {node_name} 不适合A111计算 , 采用数字模块(cpu)计算 !!!")
                continue
            assert (H % (kernel_size**(2) * in_channel) == 0)
            row_repeat_avg = H / (kernel_size**(2) * in_channel)
            if H <= array_size[1]:
                h.append(H)
            else:
                h_temp = H
                # t = 1
                split_ic = []
                # while h_temp > array_size[1]:
                #     if t == 1:
                #         t += 1
                #     else:
                #         t *= 2
                #     max_split_channel_num, split_ic = get_max_channel_split_num(in_channel,t)
                #     if np.array(split_ic).mean() != max_split_channel_num:
                #         warnings.warn(f'当前层 {node_name} 的输入通道拆分是非均匀的!!! 拆分后的通道数为: {split_ic} !!!')
                #     h_temp = max_split_channel_num * kernel_size * kernel_size * row_repeat_avg
                #     if t == in_channel:
                #         raise ValueError("暂不支持kernel_size平方超过array_size_H !!!")
                row_split_num = math.ceil(h_temp / array_size[1])
                while in_channel % row_split_num != 0:
                    row_split_num += 1
                max_split_channel_num, split_ic = get_max_channel_split_num(in_channel,row_split_num)
                if np.array(split_ic).mean() != max_split_channel_num:
                    warnings.warn(f'当前层 {node_name} 的输入通道拆分是非均匀的!!! 拆分后的通道数为: {split_ic} !!!')
                
                assert (split_ic != [])
                for ic_ in split_ic:
                    h.append(ic_ * kernel_size * kernel_size * row_repeat_avg )
        else:
            
            h_split = math.ceil(H /  array_size[1])
            h_i = h_split  
            _h = math.floor(H/h_i)
            for i in range(h_i):    
                if H - _h > 0:
                    h.append(_h)
                    H = H - _h
                else:
                    h.append(H)
                    
        w_split = math.ceil(W /  array_size[0]) 
        w_i = w_split
        _w = math.floor(W/w_i)
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
        # 并行repeat，diff_array
        repeat = 1
        if para_num != None:
            repeat = para_num[node_name][0]
        diff_array_repeat = repeat
        same_array_repeat = 1
        node_split_num[node_name] = [diff_array_repeat, same_array_repeat, len(w), len(h)]
        for k in range(repeat):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)] = [w[j],h[i]]
    
    return node_shape_split, node_split_num

def get_max_channel_split_num(ic,split_num):
    '''
    input:
        ic: 一个整数
        split_num: 将ic拆分的份数，使得每一份尽可能相同
    return:
        max_num:拆分之后，最大的份数
    '''
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

    return np.array(w).max(), w

def get_layer_ref(inputs, layer, ref):
    '''
    input:
        inputs: 算子的输入
        layer: 各层的info
    '''
    
    # MAX_count = 10
    for i in inputs:
        ref_name = i.ref
        # count = 0
        # while True:
        #     if count > MAX_count:
        #         raise ValueError(f'找寻 计算图中最近的一个 CIM友好的算子 (FC, Convolution) 失败，找到最近的算子为: {ref_name} !!!')
        if ':' in ref_name:
            ref_name = ref_name.split(':')[0]
        if 'graph_input' in ref_name:
            ref.append(ref_name)
        elif layer[ref_name].type == 'reuse':
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d', 'linear','matmul', 'fc', 'fused_fc']:
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['constant', 'split', 'add', 'fused_add', 'fused_concat', 'concat', 'mul']:
            ref.append(ref_name)
        else:
            get_layer_ref(layer[ref_name].inputs, layer, ref)
            
        # print(ref_name)
        # print(ref)
        # input()
    # return ref

def get_conv_shape(op_info):
    '''
    input:
        op_info: 算子的运算信息，op object形式，默认kernel是正方形的
    '''
    kernel_size = op_info.kernel
    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    # bias=op_info.bias
    # print(bias)
    # 默认bias放在片外做
    bias = False
    if bias:
        unroll_shape_h = kernel_size * kernel_size * in_channel + 1
    else:
        unroll_shape_h = kernel_size * kernel_size * in_channel
    unroll_shape_w = out_channel
    
    return [unroll_shape_w,unroll_shape_h]

def get_linear_shape(op_info):
    '''
    input:
        op_info: 算子的运算信息，op object形式
    '''
    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    # bias=op_info.bias
    # 默认bias放在片外做
    bias = False
    if bias:
        unroll_shape_h = in_channel + 1
    else:
        unroll_shape_h = in_channel
    unroll_shape_w = out_channel
    
    return [unroll_shape_w,unroll_shape_h]

def get_conv_info(layer):
    '''
    input:
        layer: layer object
    return:
        字典形式，{'in_channel':INT,'out_channel':INT,'kernel_size':INT,'calc_num':INT,'stride':INT,
                    'in_precision':INT,'out_precision':INT,'copy_constraint':INT}
    '''
    
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
    padding  = layer.op.padding
    
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
    
    return dict(op_type=op_type,in_channel=in_channel,out_channel=out_channel,kernel_size=kz,
                stride=stride,calc_num=calc_num,in_precision=intype,
                out_precision=outtype,copy_constraint=copy_const, in_data_len=in_data_len,
                out_data_len = out_data_len, input_shape=input_shape)

def get_linear_info(layer):
    '''
    input:
        layer: layer object
    return:
        字典形式，{'in_channel':INT,'out_channel':INT,'kernel_size':-1,'calc_num':INT,'stride':-1,
                    'in_precision':INT,'out_precision':INT,'copy_constraint':INT}
    '''
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
    
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, kernel_size=kz,
                stride=stride, calc_num=calc_num, in_precision=intype, out_precision=outtype,
                copy_constraint=copy_const, in_data_len = in_channel, out_data_len = out_channel)

def get_split_concat_info(layer):
    '''
    input:
        layer: layer object
    return:
        字典形式，{'in_channel': LSIT[INT], 'out_channel':LSIT[INT], 'axis': INT,}
    '''
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
    
    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, axis = axis, input_shape=input_shape)

def get_add_info(layer):
    '''
    input:
        layer: layer object
    return:
        字典形式，{'in_channel': LSIT[INT], 'out_channel':LSIT[INT], 'axis': INT,}
    '''
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
    '''
    input:
        列表，例如[1,2,3,4,5]
    output:
        反序的列表，[5,4,3,2,1]
    '''
    len_ = len(list_)
    reverse_list = []
    for i in range(len_-1,-1,-1):
        reverse_list.append(list_[i])
    return reverse_list

def make_mapped_ir(ir,split_info,place_info,copy_info=None,cpu_layer=None,
                   calc_info=None, device='rram-144k', runtime = 'simulation', **kwargs):
    '''
    add mapping info into IR
    input:
        ir: ir object 
        split_info: 字典形式，{'node_name':[r,w,h],...}
        place_info: 字典形式，{node_name: [Mapping info object],...}
    return:
        ir object with mapping info
    '''
    for name, layer in ir.iter_layers():
        if layer.type == 'op' :
            if cpu_layer != None and name in cpu_layer:
                # if calc_info != None: 
                #     if isinstance(calc_info, dict):
                #         if name not in calc_info.keys():
                #             Warning(f"layer{name}未设置运行参数，采用默认配置！！！")
                #             layer.c200_calc_info = C200CalcInfo().clone()
                #         else:
                #             layer.c200_calc_info = calc_info[name]
                #     else:
                #         layer.c200_calc_info = calc_info.clone()
                continue
            if layer.op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d'] or layer.op.op_id in ['linear', 'matmul', 'fc', 'fused_fc']:
                if 'rram-144k' in device:
                    if name not in split_info.keys():
                        continue
                    split_num = split_info[name]
                    if copy_info != None and name in copy_info.keys():
                        col_repeat_num = copy_info[name][1]
                        row_repeat_num = copy_info[name][0]
                    else:
                        col_repeat_num = 1
                        row_repeat_num = 1
                    # if len(split_num) == 3:
                    #     layer.mapping_info = MappingInfo(col_split_num=split_num[1],row_split_num=split_num[2],
                    #                                     col_repeat_num=col_repeat_num,row_repeat_num=row_repeat_num,
                    #                                     para_diff_array=split_num[0],
                    #                                     mappings=place_info[name])
                    # elif len(split_num) == 4:
                    assert(len(split_num) == 4)
                    if isinstance(runtime,dict):
                        if name in runtime.keys():
                            runtime_ = runtime[name]
                        else:
                            runtime_ = 'simulation'
                    elif isinstance(runtime,str):
                        runtime_ = runtime
                    else:
                        raise ValueError(f'暂不支持 runtime 类型 {type(runtime)} !!!')
                    layer.c200_mapping_info = C200MappingInfo(col_split_num=split_num[2],row_split_num=split_num[3],
                                                    col_repeat_num=col_repeat_num,row_repeat_num=row_repeat_num,
                                                    para_same_array=split_num[1],para_diff_array=split_num[0],
                                                    runtime=runtime_, mappings=place_info[name],)
                    if calc_info == None:
                        layer.c200_calc_info = C200CalcInfo().clone()
                    elif isinstance(calc_info,dict):
                        if name not in calc_info.keys():
                            if layer.c200_calc_info == None:
                                Warning(f"layer{name}未设置运行参数，采用默认配置！！！")
                                layer.c200_calc_info = C200CalcInfo().clone()
                            
                        else:
                            layer.c200_calc_info = calc_info[name]
                    else:
                        layer.c200_calc_info = calc_info.clone()
                elif 'a111-tile' in device :
                    
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
                    
                    layer.a111_mapping_info = A111MappingInfo(col_split_num=split_num[2],row_split_num=split_num[3],
                                                        col_repeat_num=col_repeat_num,row_repeat_num=row_repeat_num,
                                                        para_diff_array=split_num[1],input_buffer_addr = input_buffer_addr,
                                                        output_buffer_addr = output_buffer_addr,
                                                        in_buf_type = in_buf_type, out_buf_type= out_buf_type,
                                                        mappings=place_info[name])
                    if calc_info == None:
                        layer.a111_calc_info = A111CalcInfo().clone()
                    elif isinstance(calc_info,dict):
                        if name not in calc_info.keys():
                            Warning(f"layer{name}未设置运行参数，采用默认配置！！！")
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
                    # output_tile_buffer_addr = kwargs['output_tile_buffer_addr'][name]
                    # in_buf_type = kwargs['in_buf_type'][name]
                    # out_buf_type = kwargs['out_buf_type'][name]
                    credit_len = kwargs['credit_len'][name]
                    
                    layer.CIMA_mapping_info = CIMAMappingInfo(col_split_num=split_num[2],row_split_num=split_num[3],
                                                        col_repeat_num=col_repeat_num,row_repeat_num=row_repeat_num,
                                                        para_diff_array=split_num[0],in_line_buffer_addr = in_line_buffer_addr,
                                                        credit_len = credit_len,
                                                        mappings=place_info[name])
                    if calc_info == None:
                        layer.CIMA_calc_info = CIMACalcInfo().clone()
                    elif isinstance(calc_info,dict):
                        if name not in calc_info.keys():
                            Warning(f"layer{name}未设置运行参数，采用默认配置！！！")
                            layer.CIMA_calc_info = CIMACalcInfo().clone()
                        else:
                            layer.CIMA_calc_info = calc_info[name]
                    else:
                        layer.CIMA_calc_info = calc_info.clone()
                        
                    # dmac layer 更改data type
                    if kwargs['dmac_layer'] != None and name in kwargs['dmac_layer']:
                        layer.CIMA_calc_info.data_type = '8bit'
                        continue
                    elif kwargs['layer_data_type_dict'] != None and name in kwargs['layer_data_type_dict'].keys():
                        layer.CIMA_calc_info.data_type = kwargs['layer_data_type_dict'][name] 
                            
                else:
                    raise ValueError(f'暂不支持的device : {device} !!!')
                
            elif 'cima' in device and layer.op.op_id in ['add', 'maxpool2d', 'avgpool2d', 'concat', 'split', 'fused_add', 'fused_concat', 'identity',
                                                         'global_avg_pool2d', 'silu', 'resize', 'mul_add', 'pad', 'relu', 'type_conversion']:
                
                in_line_buffer_addr = kwargs['in_line_buffer_addr'][name]
                # output_tile_buffer_addr = kwargs['output_tile_buffer_addr'][name]
                # in_buf_type = kwargs['in_buf_type'][name]
                # out_buf_type = kwargs['out_buf_type'][name]
                credit_len = kwargs['credit_len'][name]
                
                layer.CIMA_mapping_info = CIMAMappingInfo(col_split_num=None,row_split_num=None,
                                                        col_repeat_num=None,row_repeat_num=None,
                                                        para_diff_array=None,in_line_buffer_addr = in_line_buffer_addr,
                                                        credit_len = credit_len,
                                                        mappings=place_info[name])
                    
                if calc_info == None:
                    layer.CIMA_calc_info = CIMACalcInfo().clone()
                elif isinstance(calc_info,dict):
                    if name not in calc_info.keys():
                        Warning(f"layer{name}未设置运行参数，采用默认配置！！！")
                        layer.CIMA_calc_info = CIMACalcInfo().clone()
                    else:
                        layer.CIMA_calc_info = calc_info[name]
                else:
                    layer.CIMA_calc_info = calc_info.clone()
                
                # 类型转换层单独处理
                if layer.op.op_id == 'type_conversion':
                    layer.CIMA_calc_info.data_type = layer.op.out_dtype
                    continue
                
                # dmac layer 的衍生层 也需要保持相同的数值精度
                if kwargs['dmac_layer'] != None :
                    IsDmacAppendLayer = False
                    for l in kwargs['dmac_layer']:
                        if l in name:
                            layer.CIMA_calc_info.data_type = ir.layers[name].CIMA_calc_info.data_type
                            IsDmacAppendLayer = True
                            break
                    if IsDmacAppendLayer:
                        continue
                
                # 根据传入的layer_data_type 指定各层的数据位宽
                if kwargs['layer_data_type_dict'] != None and name in kwargs['layer_data_type_dict'].keys():
                    # print(f"{name} data type: {kwargs['layer_data_type_dict'][name]} !!!")
                    layer.CIMA_calc_info.data_type = kwargs['layer_data_type_dict'][name]
                    continue
                
                    
                # 如果上述两个都没指定数据位宽，则根据ref层的数据位宽选择
                pre_layer_dict = get_pre_layer(ir.layers)
                pre_layer = pre_layer_dict[name][0]
                if 'graph_input' in pre_layer:
                    continue
                pre_layer_data_type = ir.layers[pre_layer].CIMA_calc_info.data_type
                layer.CIMA_calc_info.data_type = pre_layer_data_type
                    
    if 'a111-tile' in device:
        
        assert kwargs['tile_all'] != []
        tile_occupied_xb = kwargs['tile_occupied_xb']
        for tile in kwargs['tile_all']:
            first_node_name = tile[0]
            device_name = place_info[first_node_name][0].device
            tile_name = '.'.join(device_name.split('.')[0:3])
            npu_index, rsv, tile_index  = tile_name.split('.')
            
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
            # in_esram_addr = kwargs['in_esram_addr'][tile_name]
            # out_esram_addr = kwargs['out_esram_addr'][tile_name]
            ir.devices[npu_index].devices[tile_index].info.op_list = tile
            # ir.devices[npu_index].devices[tile_index].info.in_esram_addr = in_esram_addr
            # ir.devices[npu_index].devices[tile_index].info.out_esram_addr = out_esram_addr
            # ir.devices[npu_index].devices[tile_index].info.runtime = runtime_
            tile_occupied_xb_list = tile_occupied_xb[tile_name]
            if tile_occupied_xb_list in [[2,] , [2, 2] ,[2 , 2, 2], [2, 2, 2, 2]]:
                ir.devices[npu_index].devices[tile_index].info.tile_mode = 3
            elif tile_occupied_xb_list in [[4,], [4,4]]:
                ir.devices[npu_index].devices[tile_index].info.tile_mode = 1
            elif tile_occupied_xb_list in [[4,2], [4, 2, 2], [2, 4]]:
                ir.devices[npu_index].devices[tile_index].info.tile_mode = 2
            else:
                raise ValueError(f'不支持 {tile_occupied_xb_list} 排布方式!!!')
    
    return ir

def make_device_ir(ir,device=None):
    '''
    add device info into IR
    input:
        ir: ir object 
        device: 在ir中指定的mapping的硬件信息，字典形式，device = {'name':STR,'kind':STR,'num':INT}
                name:device的名字，kind：为IR DEVICE中的硬件名称，num: 设备数量。
    return:
        ir object with device info
    
    '''
    if ir.devices != None:
        raise ValueError(f'已存在devices: { ir.devices.keys() } !!!')

    if device != None:
        if isinstance(device,list):
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
        elif isinstance(device,dict):
            dev_copy = copy.deepcopy(device)
            dev_copy.pop('name')
            dev_copy.pop('kind')
            dev_copy.pop('num')
            ir.add_device(device['name'], device['kind'], number=device['num'], **dev_copy)
        else:
            raise TypeError(f'device type {type(device)} error!!!')
        return ir
    else:
        raise ValueError('无device信息 !!!')

def make_onnx_ir(onnx_file,return_weight=False):
    '''
    convert onnx into ir object
    input:
        onnx_file: onnx model name
        return_weight: boolean ; 
    return:
        case1 : ir object and weight value when return_weight is True
        case2 : ir object
    '''
    t = ConvertONNX(onnx_file)
    if return_weight:
        return t.ir,t.model_parser.weight_numpy
    else:
        return t.ir

def make_node_id(split_nodes):
    node_id = {}
    for i in range(len(split_nodes)):
        if len(split_nodes[i]) != 1:
            raise ValueError(f"暂不支持多层放在同一个XB中 !!! {split_nodes}")
        else:
            node_name = list(split_nodes[i][0].keys())[0]
            node_id[node_name] = i
    return node_id

def fuse_op(ir, relu_fuse =False, pool_fuse=False, split_fuse = False, silu_fuse = False, conv_mul_add_fuse = False):
    
    # 融合op
    
    # 被融合层与融合的第一层的对应关系
    fused_op_all = {}
    
    next_layer_dict = get_next_layer(ir.layers)
    layers_info = ir.layers
    
    for name, layer in layers_info.items():
        
        # 判断是否可以进行融合
        can_fuse_relu = False
        can_fuse_pool = False
        can_fuse_split = False
        can_fuse_silu = False
        can_fuse_mul_add = False
        
        # 
        if layer.type == 'op' and layer.op.op_id in ['conv2d']:
            next_layers = next_layer_dict[name]
            current_op_info = layer.op
            # 第一级算子类型判断
            if len(next_layers) == 1 and layers_info[next_layers[0]].type == 'op' and layers_info[next_layers[0]].op.op_id in ['mul']:
                nl = next_layers[0]
                nl_info = layers_info[nl]
                #  第一级维度条件判断
                nl_input_2_info = layers_info[nl_info.inputs[1].ref]
                if len(nl_info.inputs) == 2 and nl_input_2_info.type == 'op' and nl_input_2_info.op.op_id in ['constant'] and \
                    nl_info.inputs[0].channel == nl_info.inputs[1].channel:
                    
                    # 第二级算子类型判断
                    next_layers_2 = next_layer_dict[nl]
                    if len(next_layers_2) == 1 and layers_info[next_layers_2[0]].type == 'op' and layers_info[next_layers_2[0]].op.op_id in ['add']:
                        nl_nl = next_layers_2[0]
                        nl_nl_info = layers_info[nl_nl]
                        # 第二级维度条件判断
                        nl_nl_input_2_info = layers_info[nl_nl_info.inputs[1].ref]
                        if len(nl_nl_info.inputs) == 2 and nl_nl_input_2_info.type == 'op' and nl_nl_input_2_info.op.op_id in ['constant'] and \
                            nl_nl_info.inputs[0].channel == nl_nl_info.inputs[1].channel:
                            can_fuse_mul_add = True
            
            if can_fuse_mul_add:
                # 
                kernel = current_op_info.kernel
                in_channel = current_op_info.in_channel
                out_channel = current_op_info.out_channel
                stride = current_op_info.stride
                padding = current_op_info.padding
                bias = current_op_info.bias
                fused_op_obj = fused_conv2d(kernel=kernel, in_channel = in_channel,
                                        out_channel = out_channel, stride= stride,
                                        padding= padding, bias= bias).clone()
                
                # next_layer_op_info = layers_info[nl]
                fused_op_obj.mul = layers_info[nl]
                fused_op_obj.add = layers_info[nl_nl]
                # 
                fused_layer_inputs = layer.inputs
                fused_layer_weights = layer.weights
                # 融合的最后一层 的outputs 作为 融合层的outputs
                fused_layer_outputs = layers_info[nl].outputs
                
                fused_layer = make_layer(op= fused_op_obj, inputs = fused_layer_inputs,
                                        weights = fused_layer_weights,
                                        outputs = fused_layer_outputs)
                ir.layers[name] = fused_layer
                # 
                fused_op_all[nl] = name
                fused_op_all[nl_nl] = name
                
        if layer.type == 'op' and layer.op.op_id in ['conv2d', 'matmul', 'linear', 'fc', 'add', 'concat', 'fused_add', 'fused_concat']:
        # if layer.type == 'op' and layer.op.op_id in ['add', 'concat', 'fused_add', 'fused_concat'] and name not in fused_op_all.keys():
            next_layers = next_layer_dict[name]
            
            # 只支持下一层只有一层的情况
            if len(next_layers) == 1:
                nl = next_layers[0]
                if layer.op.op_id in ['conv2d', 'conv_transpose2d', 'matmul']:
                    if layers_info[nl].type == 'op':
                        if relu_fuse and layers_info[nl].op.op_id in ['relu']:
                            can_fuse_relu = True
                else:
                    if layers_info[nl].type == 'op':
                        if relu_fuse and layers_info[nl].op.op_id in ['relu']:
                            can_fuse_relu = True
                        if pool_fuse and layers_info[nl].op.op_id in ['max_pool2d', 'maxpool2d','global_avg_pool2d']:
                            can_fuse_pool = True
                        if split_fuse and layers_info[nl].op.op_id in ['split'] :
                            if layer.op.op_id in ['fused_add', 'fused_concat'] and layer.op.split != None:
                                continue
                            can_fuse_split = True
                        if silu_fuse and layers_info[nl].op.op_id in ['silu']:
                            can_fuse_silu = True
            
            # 可以融合
            if can_fuse_pool or can_fuse_relu or can_fuse_split or can_fuse_silu:
                
                current_op_info = layer.op
                if current_op_info.op_id in ['conv2d', 'conv_transpose2d']:
                    kernel = current_op_info.kernel
                    in_channel = current_op_info.in_channel
                    out_channel = current_op_info.out_channel
                    stride = current_op_info.stride
                    padding = current_op_info.padding
                    bias = current_op_info.bias
                    fused_op_obj = fused_conv2d(kernel=kernel, in_channel = in_channel,
                                            out_channel = out_channel, stride= stride,
                                            padding= padding, bias= bias).clone()

                elif current_op_info.op_id in ['matmul']:
                    in_channel = current_op_info.in_channel
                    out_channel = current_op_info.out_channel
                    bias = current_op_info.bias
                    fused_op_obj = fused_fc(in_channel = in_channel, out_channel = out_channel,
                                        bias=bias).clone()
                    
                if current_op_info.op_id in ['add']:
                    fused_op_obj = fused_add().clone()
                elif current_op_info.op_id in ['fused_add']:
                    fused_op_obj = layer.op.clone()
                        
                elif current_op_info.op_id in ['concat']:
                    attr_ = dict(axis = current_op_info.axis)
                    fused_op_obj = fused_concat(**attr_).clone()
                elif current_op_info.op_id in ['fused_concat']:
                    fused_op_obj = layer.op.clone()

                for nl in next_layers:
                    
                    if nl in fused_op_all.keys():
                        raise ValueError(f'该层 {nl} 已被 {fused_op_all[nl]} 融合!!!')
                    
                    # 将下一层的op 添加 到融合层 中
                    next_layer_op_info = layers_info[nl].op
                    if current_op_info.op_id in ['add', 'concat', 'fused_concat', 'fused_add'] and len(next_layers) == 1 and next_layer_op_info.op_id in ['split']:
                        # add/concat 与 split 融合，CIMA 架构下适用
                        fused_op_obj.split = next_layer_op_info
                        fused_op_all[nl] = name
                    elif next_layer_op_info.op_id in ['relu']:
                        fused_op_obj.relu = next_layer_op_info
                        fused_op_all[nl] = name
                    elif next_layer_op_info.op_id in ['max_pool2d', 'maxpool2d','global_avg_pool2d']:
                        fused_op_obj.pool = next_layer_op_info
                        fused_op_all[nl] = name
                    # elif current_op_info.op_id in ['conv2d', 'matmul', 'linear', 'fc', 'add', 'concat'] and next_layer_op_info.op_id in ['silu']:
                    elif current_op_info.op_id in ['add', 'concat'] and next_layer_op_info.op_id in ['silu']:
                        fused_op_obj.silu = next_layer_op_info
                        fused_op_all[nl] = name                    
                
                fused_layer_inputs = layer.inputs
                fused_layer_weights = layer.weights
                # 融合的最后一层 的outputs 作为 融合层的outputs
                fused_layer_outputs = layers_info[nl].outputs
                
                fused_layer = make_layer(op= fused_op_obj, inputs = fused_layer_inputs,
                                        weights = fused_layer_weights,
                                        outputs = fused_layer_outputs)
                ir.layers[name] = fused_layer
                # ir.add_layer(name=name, layer=fused_layer)
    
    # 删除所有被融合的层，并更改其下一层的ref name
    for fused_op_name, replaced_op_name in fused_op_all.items():
        # 替换
        next_layers = next_layer_dict[fused_op_name]
        for nl in next_layers:
            for i in ir.layers[nl].inputs:
                if ":" in i.ref:
                    ref_name = i.ref.split(':')
                    if ref_name[0] == fused_op_name:
                        i.ref = f'{replaced_op_name}:{ref_name[1]}'
                elif i.ref == fused_op_name:
                    i.ref = replaced_op_name
        # 删除
        ir.layers.pop(fused_op_name)
        
    # sorted layer, layer 层排序
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    
    return ir, fused_op_all

def insert_mul_add_op(ir, mul_add_op = None):
    
    # 获取前层信息
    pre_layer_dict = get_pre_layer(ir.layers)
    layers_info = ir.layers
    
    # 记录插入层的名称对应关系
    insert_op_name_dict = {}
    # 插入乘加算子
    if mul_add_op != None:
        for (name, index_list) in mul_add_op:
            insert_op_name_dict[name] = []

            if name not in pre_layer_dict.keys():
                raise ValueError(f'未在IR中找到层 {name} !!!')
            pre_layers = pre_layer_dict[name]
            
            for index in index_list:
                assert index <= len(pre_layers) - 1
                # 获取前一层 信息
                inserted_pre_layer_name = pre_layers[index]
                inserted_pre_layer = layers_info[inserted_pre_layer_name]
                channel = inserted_pre_layer.outputs[0].channel
                width = inserted_pre_layer.outputs[0].width
                height = inserted_pre_layer.outputs[0].height
                # 插入层信息
                insert_op_obj = MulAddOp().clone()
                inserted_layer_name = f'{inserted_pre_layer_name}_mul_add'
                inserted_layer_inputs = [dict(ref=inserted_pre_layer_name, channel=channel,height=height,width=width)]
                inserted_layer_outputs = inserted_pre_layer.outputs
                
                inserted_layer = make_layer(op= insert_op_obj, inputs = inserted_layer_inputs,
                                        outputs = inserted_layer_outputs)
                ir.layers[inserted_layer_name] = inserted_layer

                # 替换
                for i in ir.layers[name].inputs:
                    if ":" in i.ref:
                        ref_name = i.ref.split(':')
                        if ref_name[0] == inserted_pre_layer_name:
                            i.ref = f'{inserted_layer_name}:{ref_name[1]}'
                    elif i.ref == inserted_pre_layer_name:
                        i.ref = inserted_layer_name
                
                # 记录插入层的名称对应关系
                insert_op_name_dict[name].append((inserted_layer_name, index))
        # sorted layer, layer 层排序
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    
    return ir, insert_op_name_dict

def get_power2_num_list_less_than_8(num, reference_num = 8):
    '''
    输入为正整数 num ,输出为正整数集合M, 输出集合M满足以下几个特点: 
    1. 集合M中任意一个元素都小于reference_num, 
    2. 集合M及其任意一个子集合的长度都小于8, 
    3. 每个集合和子集合的长度都为2的幂次,
    4. 所有元素之和为num
    '''
    num_list = []
    while num > reference_num:
        num_list.append(reference_num)
        num -= reference_num
    if num > 0:
        bin_str = bin(num)[2:]
        for i in range(len(bin_str)):
            if bin_str[i] == '1':
                num_list.append(2**(len(bin_str)-i-1))
    # 按照每组长度最大为8进行分组
    num_list_group = []
    while True:
        num_list_group_temp = []
        index = 0
        for i in num_list:
            if len(num_list_group_temp) < 8:
                num_list_group_temp.append(i)
            else:
                num_list_group.append(num_list_group_temp)
                num_list_group_temp = []
                num_list_group_temp.append(i)
            if index == len(num_list)-1:
                num_list_group.append(num_list_group_temp)
            index += 1
        if len(num_list_group) == 1:
            num_list_group = num_list_group[0]
        if len(num_list_group) <= 8:
            if len(num_list_group) > 1:
                # 按照2的幂次进行分组
                len_ = len(num_list_group)
                bin_str_len = bin(len_)[2:]
                len_list = []
                for i in range(len(bin_str_len)):
                    if bin_str_len[i] == '1':
                        len_list.append(2**(len(bin_str_len)-i-1))
                # num_list_group
                num_list_group_all = []
                index = 0
                for i in len_list:
                    num_list_group_all.append(num_list_group[index:index+i])
                    index += i
                if len(num_list_group_all) == 1:
                    num_list_group_all = num_list_group_all[0]
            else:
                num_list_group_all = num_list_group
            break
        else:
            num_list = num_list_group
            num_list_group = []
    
    return num_list_group_all

def insert_transition_op(ir):
    '''
    处理CIMA架构下,由于rdc\concat的输入源数量以及mcast\seg的输出源数量有上限限制(目前最大为8), 因此需要插入相应层来分级处理, 减少单个节点的源数量 
    '''
    max_source_num = 8
    # 获取后层信息
    next_layer_dict = get_next_layer(ir.layers)
    output_modified_layers = {}
    for k,v in ir.layers.items():
        if v.type == 'op' and len(v.outputs) > max_source_num:
            assert len(v.outputs) <= 64, f'节点 {k} 输出源数量超过64 !!!'
            output_modified_layers[k] = len(v.outputs)
            
    if output_modified_layers != {}:
        for ln, ln_num in output_modified_layers.items():
            # 获取当前层信息
            current_layer = ir.layers[ln]
            group_num = get_power2_num_list_less_than_8(ln_num, reference_num=max_source_num)
            # assert ln_num % max_source_num == 0, f'节点 {ln} 输出源数量 {ln_num} 不是8的整数倍 !!!'
            # group_num = ln_num // max_source_num
            # group_num_list = [max_source_num] * group_num
            if current_layer.type == 'op':
                # seg情况
                if (current_layer.op.op_id in ['fused_concat', 'fused_add'] and current_layer.op.split != None) \
                        or current_layer.op.op_id == 'split':
                    assert ln_num % max_source_num == 0, f'节点 {ln} 输出源数量 {ln_num} 不是 {max_source_num} 的整数倍 !!!'
                    # 考虑芯片的限制, 这里最多出现两级分级
                    assert isinstance(group_num[0], int)
                    # 插入第一级分级层
                    level_0_seg_name = ln
                    level_0_seg_num = len(group_num)
                    current_layer_outputs_channel = current_layer.outputs[0].channel * len(current_layer.outputs)
                    assert current_layer_outputs_channel % level_0_seg_num == 0, f'{k}, {v}'
                    axis = 1
                    split = []
                    split_output = []
                    level_0_output_channel = current_layer_outputs_channel // level_0_seg_num
                    for i in range(level_0_seg_num):
                        # 默认输入通道是均匀拆分的
                        split.append(level_0_output_channel)
                        split_output.append(to_cls_obj({'channel': level_0_output_channel,
                                             'width': current_layer.inputs[0].width,
                                             'height': current_layer.inputs[0].height}, DataDef))
                    # op_ = make_op('split', axis=axis, split=split)
                    # split_input = current_layer.inputs
                    # ir.add_layer(level_0_seg_name, op=op_, inputs=split_input, outputs=split_output)
                    if 'fused' in current_layer.op.op_id:
                        ir.layers[ln].op.split.split = split
                    else:
                        ir.layers[ln].op.split = split
                    ir.layers[ln].outputs = split_output
                    # 插入第二级分级层
                    c_one = 0
                    # 
                    two_level_seg_name = {}
                    c_two = 0
                    for level_1_seg_num in group_num:
                        split = []
                        split_output = []
                        level_1_output_channel = level_0_output_channel // level_1_seg_num
                        level_1_seg_name = f'{ln}_split_level_1_{c_one}'
                        for i in range(level_1_seg_num):
                            # 默认输入通道是均匀拆分的
                            split.append(level_1_output_channel)
                            split_output.append({'channel': level_1_output_channel,
                                                'width': current_layer.inputs[0].width,
                                                'height': current_layer.inputs[0].height})
                        
                        op_ = make_op('split', axis=axis, split=split)
                        split_input = [{'ref': f'{level_0_seg_name}:{c_one}', 
                                        'channel': level_0_output_channel,
                                        'height': current_layer.inputs[0].height,
                                        'width': current_layer.inputs[0].width}]
                        ir.add_layer(level_1_seg_name, op=op_, inputs=split_input, outputs=split_output)
                        c_one += 1
                        #
                        c_three = 0
                        for seg_num in range(level_1_seg_num):
                            two_level_seg_name[f'{ln}:{c_two}'] = f'{level_1_seg_name}:{c_three}'
                            c_two += 1
                            c_three += 1       
                    # 修改原始后级节点的ref
                    for nl in next_layer_dict[ln]:
                        for in_ in ir.layers[nl].inputs:
                            if in_.ref in two_level_seg_name.keys():
                                in_.ref = two_level_seg_name[in_.ref]
                    
                # mcast 情况
                else:
                    pass
    
    # 获取前层信息
    # pre_layer_dict = get_pre_layer(ir.layers)
    input_modified_layers = {}
    for k,v in ir.layers.items():
        if v.type == 'op' and v.op.op_id not in ['constant'] and len(v.inputs) > max_source_num:
            assert len(v.inputs) <= 64, f'节点 {k} 输出源数量超过64 !!!'
            input_modified_layers[k] = len(v.inputs)                         
    
    if input_modified_layers != {}:
        for ln, ln_num in input_modified_layers.items():
            # 获取当前层信息
            current_layer = ir.layers[ln]
            group_num = get_power2_num_list_less_than_8(ln_num, reference_num=max_source_num)
            # assert ln_num % max_source_num == 0, f'节点 {ln} 输出源数量 {ln_num} 不是8的整数倍 !!!'
            # group_num = ln_num // max_source_num
            # group_num_list = [max_source_num] * group_num
            if current_layer.type == 'op':
                # rdc 情况
                if current_layer.op.op_id in ['fused_add', 'add']:
                    assert ln_num % max_source_num == 0, f'节点 {ln} 输出源数量 {ln_num} 不是 {max_source_num} 的整数倍 !!!'
                    # 考虑芯片的限制, 这里最多出现两级分级
                    assert isinstance(group_num[0], int)
                    # 插入第一级分级层
                    index = 0
                    c = 0
                    current_layer_inputs_list = current_layer.inputs
                    current_layer_outputs_list = current_layer.outputs
                    # 
                    level_1_add_input_list = []
                    for level_0_add_num in group_num:
                        level_0_add_name = f'{ln}_add_level_0_{c}'
                        op_ = make_op('add')
                        add_inputs_list = current_layer_inputs_list[index:index+level_0_add_num]
                        ir.add_layer(level_0_add_name, op=op_, inputs=add_inputs_list, outputs=current_layer_outputs_list)
                        index += level_0_add_num
                        c += 1
                        # 记录第二级分级层的名称
                        level_1_add_input_list.append(to_cls_obj({'ref': level_0_add_name,
                                                   'channel': current_layer_inputs_list[0].channel,
                                                   'height': current_layer_inputs_list[0].height,
                                                   'width': current_layer_inputs_list[0].width}, DataDef))
                    # 插入第二级分级层
                    ir.layers[ln].inputs = level_1_add_input_list
                    
                # concat 情况
                elif current_layer.op.op_id in ['fused_concat', 'concat']:
                    assert ln_num % max_source_num == 0, f'节点 {ln} 输出源数量 {ln_num} 不是 {max_source_num} 的整数倍 !!!'
                    # 考虑芯片的限制, 这里最多出现两级分级
                    assert isinstance(group_num[0], int)
                    # 插入第一级分级层
                    index = 0
                    c = 0
                    current_layer_inputs_list = current_layer.inputs
                    current_layer_outputs_list = current_layer.outputs
                    concat_axis = current_layer.op.axis
                    # 记录第一级分级层的名称
                    level_1_concat_input_list = []
                    for level_0_concat_num in group_num:
                        # 计算第一级的输出通道数
                        level_0_concat_output_channel = current_layer_inputs_list[0].channel * level_0_concat_num
                        level_0_concat_name = f'{ln}_concat_level_0_{c}'
                        op_ = make_op('concat', axis=concat_axis)
                        add_inputs_list = current_layer_inputs_list[index:index+level_0_concat_num]
                        current_layer_outputs_list[0].channel = level_0_concat_output_channel
                        ir.add_layer(level_0_concat_name, op=op_, inputs=add_inputs_list, outputs=current_layer_outputs_list)
                        # 
                        index += level_0_concat_num
                        c += 1
                        # 记录第二级分级层的名称
                        level_1_concat_input_list.append(to_cls_obj({'ref': level_0_concat_name,
                                                   'channel': current_layer_outputs_list[0].channel,
                                                   'height': current_layer_outputs_list[0].height,
                                                   'width': current_layer_outputs_list[0].width}, DataDef))
                    # 插入第二级分级层
                    ir.layers[ln].inputs = level_1_concat_input_list
    
    ir.dump_json(file=f'test.yaml')
    
    # sorted layer, layer 层排序
    if output_modified_layers != {} or input_modified_layers != {} :
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    
    return ir

                    
def insert_identity_op(ir):
    
    # 获取前层信息
    next_layer_dict = get_next_layer(ir.layers)
    layers_info = ir.layers
    
    # 获取需要插入identity层的名称
    need_insert_layer = {}
    for k,v in next_layer_dict.items():
        if not math.log2(len(v)).is_integer():
            need_insert_layer[k] = len(v)
    
    # 插入identity算子
    if need_insert_layer != {}:
        for layer_name, output_node_num in need_insert_layer.items():
            # 根据当前层输出的个数，确定每个输出对应几个输出node
            layer_output_num = len(layers_info[layer_name].outputs)
            assert output_node_num % layer_output_num == 0, f'输出节点数:{output_node_num}, 层输出节点数:{layer_output_num} 不匹配！！！'
            
            # 拆分节点数
            split_num = output_node_num // layer_output_num
            # 将split_num 拆分为2的幂次方的求和
            split_num_list = bin(split_num)[2:]
            # 
            insert_node_num_list = []
            for i in range(len(split_num_list)):
                if split_num_list[i] == '1':
                    insert_node_num_list.append(2**(len(split_num_list)-i-1))
            # 
            if  layer_output_num != 1:
                assert math.log2(layer_output_num).is_integer(), f'输出个数不是2的幂次方!!!'
                # 根据原来ref为layer_name:X的层进行分类
                next_layer_sort_dict = {}
                for nl in next_layer_dict[layer_name]:
                    for i in ir.layers[nl].inputs:
                        if layer_name in i.ref:
                            ref_name = i.ref.split(':')
                            if ref_name[1] not in next_layer_sort_dict.keys():
                                next_layer_sort_dict[ref_name[1]] = []
                            next_layer_sort_dict[ref_name[1]].append(nl)
                
                # 根据insert node num 列表，插入identity算子
                index = 0
                for i in range(len(insert_node_num_list)):
                    # 替换掉原来输出层的ref
                    for k,v in next_layer_sort_dict.items():
                        # 插入层信息
                        identity_name = f'{layer_name}_identity_mcast_{i}_seg_{k}'
                        op_ = make_op('identity')
                        ref_layer_name = f'{layer_name}:{i}'
                        input_shape = layers_info[layer_name].outputs[0]
                        inputs_ = [dict(ref=ref_layer_name,channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                        outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                        ir.add_layer(identity_name,op=op_,inputs=inputs_,outputs=outputs_)
                        for l in v[index:index+insert_node_num_list[i]]:
                            for j in ir.layers[l].inputs:
                                if layer_name in j.ref:
                                    j.ref = identity_name
                    index += insert_node_num_list[i]
            else:   
                # 根据insert node num 列表，插入identity算子
                index = 0
                for i in range(len(insert_node_num_list)):
                    # 插入层信息
                    identity_name = f'{layer_name}_identity_{i}'
                    op_ = make_op('identity')
                    ref_layer_name = f'{layer_name}'
                    input_shape = layers_info[layer_name].inputs[0]
                    inputs_ = [dict(ref=ref_layer_name,channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                    outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                    ir.add_layer(identity_name,op=op_,inputs=inputs_,outputs=outputs_)
                    # 替换掉原来输出层的ref
                    nl_list = next_layer_dict[layer_name]
                    for l in nl_list[index:index+insert_node_num_list[i]]:
                        for j in ir.layers[l].inputs:
                            if layer_name in j.ref:
                                j.ref = identity_name
                    index += insert_node_num_list[i]
            
        # sorted layer, layer 层排序
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    
    return ir


def insert_type_conversion_op(ir, type_conversion_list = None):
    
    # 获取前层信息
    layers_info = ir.layers
    
    for (name, in_index, conversion_type) in type_conversion_list:
        
        assert len(in_index) == len(conversion_type)
        
        # 获取当前层信息
        inserted_original_layer = layers_info[name]
        c = 0
        for id in in_index:
            assert conversion_type[c][0] != conversion_type[c][1], f'插入层 {name} 输入数据类型应与输出数据类型不一致!!!' 
            in_info = inserted_original_layer.inputs[id]
            
            # 插入层信息
            insert_op_obj = TypeConversionOp().clone()
            insert_op_obj.in_dtype = conversion_type[c][0]
            insert_op_obj.out_dtype = conversion_type[c][1]
            inserted_layer_name = f'{name}_in_type_conversion_{id}'
            inserted_layer_inputs = [copy.deepcopy(in_info)]
            inserted_layer_outputs = [dict(channel=in_info.channel, width=in_info.width, height=in_info.height)]
            
            inserted_layer = make_layer(op= insert_op_obj, inputs = inserted_layer_inputs,
                                    outputs = inserted_layer_outputs)
            ir.layers[inserted_layer_name] = inserted_layer

            # 更新当前层的ref
            ir.layers[name].inputs[id].ref = inserted_layer_name
            
            c += 1
    
    # sorted layer, layer 层排序
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
                index_ = [int(name_[1]),int(name_[2]),int(name_[3].split('_')[0])]
            else:
                index_ = [int(name_[1]),int(name_[2]),int(name_[3])]
            if node_name not in node_mapping_info.keys():
                node_mapping_info[node_name] = []
            mapping_info = C200DeviceMappingInfo(index = index_, device=device_ref, address=value)
            node_mapping_info[node_name].append(mapping_info)


def get_pre_layer(layers):
    ''''
    获取当前层名称与前一层的名称的对应字典
    input: {layer_name:layer_object}
    return: {current_layer_name: pre_layer_name}
    '''
    prefix_layer = {}
    for name, layer in layers.items():
        if layer.type not in ['input']:
            prefix_layer[name] =  []
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                continue
            # get_layer_ref(layers[name].inputs, layers, prefix_layer[name])
            for i in layer.inputs:
                if 'graph_input' not in i.ref:
                    #判断前一层是不是flatten，reshape等算子，如果是，则跳过
                    ref = i.ref
                    if ':' in ref:
                        ref = ref.split(':')[0]
                    pre_layer = layers[ref]
                    if pre_layer.type == 'op' and pre_layer.op.op_id in ['flatten','reshape']:
                        for j in pre_layer.inputs:
                            prefix_layer[name].append(j.ref)
                    else:
                        prefix_layer[name].append(ref)
                else:        
                    prefix_layer[name].append(i.ref)
    # print(prefix_layer)
    # input()
    return prefix_layer   


def get_next_layer(layers):
    '''
    获取当前层名称与下一层的名称的对应字典
    input: {layer_name:layer_object}
    return: {current_layer_name: next_layer_name}
    '''
    next_layer = {}
    pre_layer = get_pre_layer(layers)  
    
    for k,v in pre_layer.items():
        # if layers[k].type == 'op' and layers[k].op.op_id not in ['flatten']:
        #     for name in v:
        #         if name not in next_layer.keys():
        #             next_layer[name] = []
        #         next_layer[name].append(k)
        if layers[k].type == 'op' and layers[k].op.op_id in ['flatten']:
            continue
    
        for name in v:
            if name not in next_layer.keys():
                next_layer[name] = []
            next_layer[name].append(k)
                
    return next_layer

def draw_square_mesh(grid_size, square_size, arrow_size, text_dict, value_dict, dmac_info,
                     save_fig = None):
    # 创建一个无刻度的画布
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # 在指定位置写入字体，并设置字体大小和颜色
    def draw_text_upon_dash(text, position, x, y, square_size, arrow_size):
        # text_x = x + square_size / 2
        # text_y = y + square_size / 2
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
        elif position == 'right' :
            text_y = text_y + 10
        elif position == 'top':
            text_x = text_x - 10
        elif position == 'bottom':
            text_x = text_x + 10
            
        ax.text(text_x, text_y, text, fontsize = 8, ha='center', va='center', color='purple')
    
    # 在指定位置写入字体，并设置字体大小
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
            ax.text(text_x, text_y, text, fontsize = 6, ha='center', va='center', color='red', fontdict={'weight':'bold'})
        
    # 绘制方块网格和箭头
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = j * (square_size + arrow_size)
            y = i * (square_size + arrow_size)
            ax.text(x  + square_size / 2, y + square_size / 2, f'Node\n[{i}][{j}]', fontsize=7, ha='center', va='center')
            
            # 写入 DMAC layer name
            if (i, j) in dmac_info.keys():
                ln = dmac_info[(i ,j)]
                ax.text(x + 25 , y + square_size / 2 + 10, f'{ln}(DMAC)', fontsize=5, ha='center', 
                        va='center', color='red', fontdict={'weight':'bold'})
            
            # 创建方块对象
            square = plt.Rectangle((x, y), square_size, square_size, linewidth=2, edgecolor='black', facecolor='none')

            # 添加方块到图形中
            ax.add_patch(square)
            
            # 添加小格子
            small_square_size = square_size / 4
            small_x = x + square_size / 2 - small_square_size / 2
            small_y_top = y + square_size - small_square_size
            small_y_bottom = y
            small_x_left = x
            small_x_right = x + square_size - small_square_size

            # 添加上部居中的小格子
            small_square_top = plt.Rectangle((small_x, small_y_top), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_top)

            # 添加底部居中的小格子
            small_square_bottom = plt.Rectangle((small_x, small_y_bottom), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_bottom)

            # 添加左边居中的小格子
            small_square_left = plt.Rectangle((small_x_left, y + square_size / 2 - small_square_size / 2), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_left)

            #添加右边居中的小格子
            small_square_right = plt.Rectangle((small_x_right, y + square_size / 2 - small_square_size / 2), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_right)
            
            # 添加箭头
            if j < grid_size[1] - 1:
                # 添加向右的箭头
                right_arrow = plt.Arrow(x + square_size, y + square_size / 2 + 2, arrow_size, 0, color='black', width=arrow_size/2)
                ax.add_patch(right_arrow)
                # 添加向左的箭头
                left_arrow = plt.Arrow(x + square_size + arrow_size, y + square_size / 2 - 2, -arrow_size, 0, color='black', width=arrow_size/2)
                ax.add_patch(left_arrow)
            if i < grid_size[0] - 1:
                # 添加向上的箭头
                up_arrow = plt.Arrow(x + square_size / 2 + 2, y + square_size, 0, arrow_size, color='black', width=arrow_size/2)
                ax.add_patch(up_arrow)
                # 添加向下的箭头
                down_arrow = plt.Arrow(x + square_size / 2 - 2, y + square_size + arrow_size, 0, -arrow_size, color='black', width=arrow_size/2)
                ax.add_patch(down_arrow)
            
            # 在指定小格子上写入字体
            if (i, j) in text_dict:
                text_position = text_dict[(i, j)]
                for value in text_position:
                    text, position = value
                    draw_text_in_square(text, position, x, y, square_size)
            
            # 在指定的两个大格子至今写入字体
            if (i, j) in value_dict:
                text_position = value_dict[(i,j)]
                for value in text_position:
                    text, position = value
                    draw_text_upon_dash(text, position, x, y, square_size, arrow_size)
                
    
    # 设置坐标轴范围
    ax.set_xlim(0, grid_size[1] * (square_size + arrow_size))
    ax.set_ylim(grid_size[0] * (square_size + arrow_size), 0)

    plt.tight_layout()
    # 显示图形
    if save_fig:
        import os
        if '\\' in save_fig:
            path = save_fig.split('\\')
            path = '\\'.join(path[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
        plt.savefig(save_fig)
        
    plt.close()
    # plt.show()


def draw_mesh_fig(record_io_workload, node_mapping_info, mesh = [6, 6], save_fig_path = '1.svg'):
    
    # record_io_workload = map.place.record_io_workload
    
    draw_info = {} # RRAM xb layer
    dash_info = {} # Data communication
    dmac_info = {} # DMAC layer
    
    draw_loc = {0:'bottom', 1:'right', 2:'top', 3:'left'}

    # node_mapping_info = map.place.node_mapping_info_list
    
    for name, addr in node_mapping_info.items():
        name_ = name.split('.')[0]
        
        node_id = int(addr.split('.')[1].split(':')[1])
        figure_id = (node_id // mesh[1], node_id % mesh[1])
        
        if 'dmac' in addr: 
            dmac_info[figure_id] = name_
            continue
         
        if figure_id not in draw_info.keys():
            draw_info[figure_id] = []
        location = draw_loc[int(addr.split('.')[2].split(':')[1])]
        if '[' in addr.split('.')[-1] and ',' in addr.split('.')[-1]:
            name_ = 'PE_TASK'
            pe_number = addr.split('.')[3].split(':')[1]
            # if '-' in pe_number:
            #     p1 = int(pe_number.split('-')[0])
            #     p2 = int(pe_number.split('-')[1])
            #     pe_num = p2 - p1 + 1
            # else:
            #     pe_num = 1
            pe_num = 1
        else:
            name_ = 'Others'
            pe_num = 1
        draw_info[figure_id].append((name_, location, pe_num))
    
    # 统计 单个方向上的 PE任务数量
    draw_info_pe_task_num = {}
    pe_num_list = []
    for k,v in draw_info.items():
        draw_info_pe_task_num[k] = []
        count_loc = {'bottom':0, 'right':0, 'top':0, 'left':0}
        for n, l, pe_num in v:
            if n == 'PE_TASK':
                count_loc[l] += pe_num
        for d, val in count_loc.items():
            draw_info_pe_task_num[k].append((val, d))
            if val > 2:
                print(k)
                print(l)
                print(val)
                
            pe_num_list.append(val)
            
    print(f'Max PE Thread Num: {np.array(pe_num_list).max()}')
    
    value_list = []
    
    for loc, value in record_io_workload.items():
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
       
    draw_square_mesh(mesh, 50, 10, draw_info_pe_task_num, dash_info, dmac_info, save_fig_path)

def replace_op(ir):
    
    # ================================================================================
    # 将 ir 中的相邻operator进行等价算子替换，可以针对特定的硬件平台进行优化，从而减少运算量
    # ================================================================================
    
    # pattern 1: Conv/FC --> sigmoid -->mul --> y    ==>   Conv/FC --> Silu ---> y 
    #               |                    | 
    #               --------------------->   
    
    layers = ir.layers
    next_layers_dict = get_next_layer(layers)
    
    layers_recurrent = copy.deepcopy(layers)
    
    for layer_name, layer_info in layers_recurrent.items():
        
        current_layer_info = layer_info
        # 首先判断当前层的op id 是否是 Conv 或者 FC 或者 Add 或者 Concat
        # 然后判断下一层是否为 sigmoid 以及 mul
        # 最后判断sigmoid 的输出是否为 mul
        next_sigmoid = False
        next_mul = False
        sigmoid_out_mul = False
        
        if current_layer_info.type == 'op' and current_layer_info.op.op_id in ['conv2d','fc','linear','matmul', 'fused_conv2d', 'fused_fc',
                                                                               'fused_add', 'add', 'concat', 'fused_concat', 'batch_norm2d']:
            next_layers = next_layers_dict[layer_name]
            
            for i in next_layers:
                if layers[i].type == 'op':
                    if layers[i].op.op_id == 'sigmoid':
                        next_sigmoid = True
                        sigmoid_next_layer = next_layers_dict[i]
                        # sigmoid 的输出只能是mul 才能进行合并
                        if len(sigmoid_next_layer) == 1:
                            snl = sigmoid_next_layer[0]
                            if layers[snl].type == 'op' and layers[snl].op.op_id == 'mul':
                                sigmoid_out_mul = True
                    if layers[i].op.op_id == 'mul':
                        next_mul = True
            
            # 如果三个条件同时满足, 插入silu 算子，删除sigmoid 与 mul 算子
            if next_sigmoid and next_mul and sigmoid_out_mul:
                # 插入 silu
                op_ = make_op('silu')
                in_height = current_layer_info.outputs[0].height
                in_width = current_layer_info.outputs[0].width
                in_channel = current_layer_info.outputs[0].channel
                input_ = [dict(ref=layer_name, channel= in_channel, width=in_width, height=in_height)]
                silu_name = 'Silu_' + layer_name
                # 激活函数不改变图片形状
                output_ = [dict(channel= in_channel, width=in_width, height=in_height)]
                ir.add_layer(silu_name,op=op_,inputs=input_,outputs=output_)
                
                # 删除 mul 和 sigmoid ，并更新 mul 层的下一层的输入ref name
                for i in next_layers:
                    if layers[i].type == 'op':
                        if layers[i].op.op_id == 'sigmoid':
                            ir.layers.pop(i)
                        elif layers[i].op.op_id == 'mul':
                            # 更改 mul 层的下一层的 ref name
                            mul_next_layer = next_layers_dict[i]
                            for mnl in mul_next_layer:
                                for in_ in ir.layers[mnl].inputs:
                                    if in_.ref == i:
                                        in_.ref = silu_name
                            ir.layers.pop(i)
                            
    # ir layers 排序
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    
    return ir

def remove_flatten_op(ir):
    layers = ir.layers
    layers_recurrent = copy.deepcopy(layers)
    
    # 移除flatten 算子
    for n, l in layers_recurrent.items():
        if l.type == 'op' and l.op.op_id == 'flatten':
            current_layer_ref = l.inputs[0].ref
            # 更新 所有输入为flatten的层
            for n_,l_ in ir.layers.items():
                for in_ in l_.inputs:
                    if in_.ref == n:
                        in_.ref = current_layer_ref
            ir.layers.pop(n)
    # ir.layers = layers
    return ir
    
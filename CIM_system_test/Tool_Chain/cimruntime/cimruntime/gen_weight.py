from e100_irtool.core.ir import load_ir, BaseIR
from e100_irtool.tools import flatten_layers  # noqa
import pickle
import numpy as np
from .quant import *
import warnings

def pickle_load(file, **kwargs):
    with open(file, 'rb') as f:
        return pickle.load(f, **kwargs)

def get_addr_record(mappings):
    row_start_addr_record = {}
    col_start_addr_record = {}
    for k,v in mappings.items():
        r_index, h_index, w_index = v.index
        if h_index in row_start_addr_record.keys():
            if h_index != 0:
                assert ((v.address[2] + row_start_addr_record[h_index-1]) == row_start_addr_record[h_index])
        else:
            if h_index == 0:
                row_start_addr_record[h_index] = 0
            else:
                row_start_addr_record[h_index] = v.address[2] + row_start_addr_record[h_index-1]
        if w_index in col_start_addr_record.keys():
            if w_index != 0:
                assert ((v.address[3] + col_start_addr_record[w_index-1]) == col_start_addr_record[w_index])
        else:
            if w_index == 0:
                col_start_addr_record[w_index] = 0
            else:
                col_start_addr_record[w_index] = v.address[3] + col_start_addr_record[w_index-1]
    return row_start_addr_record,col_start_addr_record

# 给定pt形式的数组，制作rram格式的数组，即pt顺序的格式转为rram4i+j的格式
def pt_sequence_2_rram_discretization(pt_sequence):
    pt_sequence_row, pt_sequence_colum = pt_sequence.shape
    rram_discretization = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum[:, :pt_sequence_colum] = pt_sequence
    # 遍历127次，对应索引为：pt0-rram0,pt1-rram4,pt31-rram124,pt32-rram1,pt126-rram123,pt127-rram127
    for rram_colum in range(127):
        mapping_index = (4 * rram_colum) % 127
        rram_discretization[:, mapping_index] = pt_sequence_128colum[:, rram_colum]
    # 最后一次需要单独赋值，pt127-rram127
    rram_discretization[:, 127] = pt_sequence_128colum[:, 127]
    return rram_discretization

# 给定pt形式的权重，转为rram需要的权重，包括具体值大小和格式
def trans_pt_weight_2_rram(pt_weight, bias_flag=False, pos_sa = 5, neg_sa = 5):
    # bias_flag是一个标志位，为了将权重移到下面，为bias的8行留出一个空间出来
    # 对于pt的3值权重，映射到rram上需要改变具体值，也就是rram = pt x pos_sa 或者rram = pt x neg_sa
    bias_row_num = 8*2

    row, colum = pt_weight.shape
    # 转换原始pt权重为2T2R权重
    rram_weight = np.zeros([row * 2, colum])
    pos_weight = np.zeros_like(pt_weight)
    neg_weight = np.zeros_like(pt_weight)
    flag = pt_weight > 0
    pos_weight[flag] = pos_sa
    flag = pt_weight < 0
    neg_weight[flag] = neg_sa
    rram_weight[::2, :] = pos_weight
    rram_weight[1::2, :] = neg_weight
    # 根据芯片mapping策略，重构rram权重（每隔4列存一个数据，满列操作，即128列都用）
    sub_mapping_weight = pt_sequence_2_rram_discretization(rram_weight)
    # 补全其余行的数据，最终芯片mapping的权重需求为640x128的矩阵
    mapping_weight = np.zeros([640, 128])
    if bias_flag:
        mapping_weight[bias_row_num : bias_row_num + rram_weight.shape[0]] = sub_mapping_weight
    else:
        mapping_weight[:rram_weight.shape[0]] = sub_mapping_weight
    mapping_weight = mapping_weight.astype(np.uint8)
    return mapping_weight

def gen_array_weight(ir,weight_file=None,format='CHW', device_shape=(576,128), device='C200', **kwargs):
    # 判断是否为ir object
    if isinstance(ir,str):
        ir = load_ir(ir)
    elif isinstance(ir, BaseIR):
        ir = ir
    else:
        raise ValueError(f"暂不支持ir类型:{type(ir)}")
    # 判断weight是否为字典，目前只支持字典格式
    # {'layer_name':tensor, }
    weight = None
    if isinstance(weight_file,str):
        weight = pickle_load(weight_file)
    elif isinstance(weight_file, dict):
        weight = weight_file
    else:
        raise ValueError(f"暂不支持weight 数据类型:{type(weight_file)}")
    
    array_data = {}
    systemc_weight_data = {}
    
    # onnx_weight_HWC = {}

    # layer info
    layers = ir.flatten_layers()
    
    # 声明a111 mapping 权重的sa 值
    if device.lower() == 'a111':
        pos_sa = 5
        neg_sa = 5
        if 'pos_sa' in kwargs.keys():
            pos_sa = kwargs['pos_sa']
        if 'neg_sa' in kwargs.keys():
            neg_sa = kwargs['neg_sa']
    
    for name, layer in layers.items():
        if layer.type in ['input', 'output', 'reuse']:
            continue
        
        # 获取mapping_info
        mapping_info = None
        if device.lower() =='c200':
            mapping_info = layer.c200_mapping_info
        elif device.lower() == 'a111':
            mapping_info = layer.a111_mapping_info
        else:
            raise ValueError(f"暂不支持设备 {device} ！！！")
    
        op_id = layer.op.op_id
        # 判断是否需要在片上算
        if op_id in ['matmul','fc','linear','conv2d', 'conv_transpose2d'] and mapping_info != None  :
            weight_name = name +'.weight'
            assert weight_name in weight.keys(), f'{weight_name} 不存在 {weight.keys()} 中 !!!'
            col_repeat_num =  mapping_info.col_repeat_num
            row_repeat_num = mapping_info.row_repeat_num
            wd = weight[weight_name]
            # print(wd.shape)
            # conv 权重一般为四维 tensor
            if op_id in ['conv2d', 'conv_transpose2d']:
                
                if len(wd.shape) == 2:
                    # 默认权重的输出通道在前，因此这里需要转置将输出通道放在后面
                    wd = wd.transpose(1,0)
                    
                elif len(wd.shape) == 4:
                    if op_id == 'conv_transpose2d':
                        # 水平，竖直方向镜像翻转
                        wd = np.flip(wd, axis=(-1, -2))
                        # 转置卷积的输出通道在第二维， 输入通道在第一维
                        wd = wd.transpose(1, 0, 2, 3)
                        
                    # 默认权重是按照CHW排布
                    if format == 'HWC':
                        wd = wd.transpose(0,2,3,1)
                        # # 记录原始的权重维度变为channel last 之后的数据
                        # onnx_weight_HWC[weight_name] = wd
                        
                        wd = wd.reshape(wd.shape[0],-1,wd.shape[3])
                    elif format == 'CHW':
                        
                        wd = wd.reshape(wd.shape[0],-1)
                        wd = wd.transpose(1,0)
                        wd = np.tile(wd,[row_repeat_num,col_repeat_num])
                    else:
                        raise ValueError(f"暂不支持数据格式{format}")
                    # # 判断是否有bias
                    # if layer.op.bias:
                    #     bias_name = name +'.bias'
                    #     assert bias_name in weight.keys()
                    #     bias = weight[bias_name]
                    #     if len(bias.shape) == 1:
                    #         bias = bias.reshape(1,bias.shape[0])
                    #     wd = np.concatenate([wd,bias],axis=0)
                else:
                    raise ValueError(f'暂不支持维度 {wd.shape} !!!')
            elif op_id in ['matmul','fc','linear']:
                
                if format == 'HWC':
                    # 判断前一层是否通过flatten或者reshape进行降维，如果format==‘HWC’，则需要把权重进行重排，
                    # 因为pytorch默认的输入是按照‘CHW’的方式进行降维（flatten、reshape）
                    if (len(layer.inputs) != 1):
                        raise ValueError('全连接层目前只支持一个动态的输入值，权重为静态！！！')
                    former_layer_name = layer.inputs[0].ref
                    former_layer =  layers[former_layer_name]
                    if former_layer.op.op_id in ['reshape','flatten']:
                        in_channel = former_layer.inputs[0].channel
                        in_h = former_layer.inputs[0].height
                        in_w = former_layer.inputs[0].width
                        assert(wd.shape[1] == in_channel * in_h * in_w)
                        out_d = wd.shape[0]
                        wd = wd.reshape(out_d,in_channel,in_h,in_w)
                        wd = wd.transpose(0,2,3,1)
                        wd = wd.reshape(out_d,-1)
                        # # 记录原始的权重维度变为channel last 之后的数据
                        # onnx_weight_HWC[weight_name] = wd
                    # 判断前一层是否是Concat，concat是否是通过flatten或者reshape进行降维，如果format==‘HWC’，则需要把权重进行重排，
                    # 因为pytorch默认的输入是按照‘CHW’的方式进行降维（flatten、reshape）
                    # 此时则需要将当前层的权重按照concat的方式进行分割，分割之后再进行重排,重排之后重组
                    elif former_layer.op.op_id == 'concat':
                        current_input_row_start = 0
                        transformed_fc_weight = []
                        for in_ in former_layer.inputs:
                            former_former_layer = layers[in_.ref]
                            if former_former_layer.op.op_id in ['reshape','flatten']:
                                # 默认reshape和flatten只有一个输入
                                assert (len(former_former_layer.inputs) == 1)
                                in_channel = former_former_layer.inputs[0].channel
                                in_h = former_former_layer.inputs[0].height
                                in_w = former_former_layer.inputs[0].width
                                row_num = in_channel * in_h * in_w
                                current_input_row_end = current_input_row_start + row_num
                                current_layer_fc_weight = wd[:,current_input_row_start:current_input_row_end] + 0
                                out_d = current_layer_fc_weight.shape[0]
                                current_layer_fc_weight = current_layer_fc_weight.reshape(out_d,in_channel,in_h,in_w)
                                current_layer_fc_weight = current_layer_fc_weight.transpose(0,2,3,1)
                                current_layer_fc_weight = current_layer_fc_weight.reshape(out_d,-1)
                                transformed_fc_weight.append(current_layer_fc_weight)
                                # 输入起始递增
                                current_input_row_start = current_input_row_end
                            else:
                                in_channel = in_.channel
                                in_h = in_.height
                                in_w = in_.width
                                row_num = in_channel * in_h * in_w
                                current_input_row_end = current_input_row_start + row_num
                                current_layer_fc_weight = wd[:,current_input_row_start:current_input_row_end] + 0
                                transformed_fc_weight.append(current_layer_fc_weight)
                                # 输入起始递增
                                current_input_row_start = current_input_row_end
                           
                        # 拼接所有的权重
                        transformed_fc_weight = np.concatenate(transformed_fc_weight,axis=1)
                        assert (transformed_fc_weight.shape == wd.shape)
                        wd = transformed_fc_weight
                # 默认的权重数据输出通道在前,转置之后为输出通道在后符合阵列的排布
                wd = wd.transpose(1,0)
                wd = np.tile(wd,[row_repeat_num,col_repeat_num])  
            
            row_record,col_record = get_addr_record(mapping_info.mappings)
            
            systemc_id = 0
            for k,v in mapping_info.mappings.items():
                r_index, h_index, w_index = v.index
                if h_index == 0:
                    input_row_start = 0
                else:
                    input_row_start = row_record[h_index]
                if w_index == 0:
                    input_col_start = 0
                else:
                    input_col_start = col_record[w_index]
                # print(v.device)
                # input()
                # array_id = int(v.device.split(":")[-1])
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
                    # conv2d 按照channel进行拆分，
                    # 因此current_row_num 和 input_row_start必须要是kernel_size * kernel_size * row_repeat_num的倍数
                    kernel_size = layer.op.kernel
                    assert(current_row_num % ((kernel_size**2) * row_repeat_num) == 0)
                    assert(input_row_start % ((kernel_size**2) * row_repeat_num) == 0)
                    assert((kernel_size**2) == wd.shape[1])
                    current_channel_num = int(current_row_num / ((kernel_size**2) * row_repeat_num))
                    input_channel_start = int(input_row_start / ((kernel_size**2) * row_repeat_num))
                    input_channel_end = input_channel_start + current_channel_num
                    # print(f'当前层的名字：{name}')
                    # print(f'index:{systemc_id}')
                    # print(f'权重维度：{wd.shape}')
                    # print(f'input 行起始地址：{input_row_start}')
                    # print(f'input 行结束地址：{input_row_end}')
                    # print(f'input channel起始：{input_channel_start}')
                    # print(f'input channel结束：{input_channel_end}')
                    # print(f'output channel起始：{input_col_start}')
                    # print(f'output channel结束：{input_col_end}')
                    # input()
                    
                    # 得到原始矩阵中的channel起始值和数量
                    current_wd = wd[:,:,input_channel_start:input_channel_end] + 0
                    # reshape 为二维矩阵
                    current_wd = current_wd.reshape(current_wd.shape[0],-1)
                    # 转置一下，以行为第一维度，列为第二维度
                    current_wd = current_wd.transpose(1,0)
                    # 按照row_repeat_num,col_repeat_num复制一下，
                    current_wd = np.tile(current_wd,[row_repeat_num,col_repeat_num])
                    ## 列方向先复制再拆分
                    if device.lower() == 'a111':
                        array_data[array_id] = trans_pt_weight_2_rram(current_wd[:,input_col_start:input_col_end], bias_flag=True, pos_sa=pos_sa, neg_sa=neg_sa)
                    elif device.lower() == 'c200':
                        array_data[array_id][array_row_start:array_row_end,array_col_start:array_col_end] = current_wd[:,input_col_start:input_col_end]
                    # systemc weight 需要transpose一下方便计算
                    systemc_weight_data[name+f":{systemc_id}"] = current_wd[:,input_col_start:input_col_end].transpose(1,0)
                    
                else:
                    if device.lower() == 'a111':
                        array_data[array_id] = trans_pt_weight_2_rram(wd[input_row_start:input_row_end,input_col_start:input_col_end], pos_sa=pos_sa, neg_sa=neg_sa)
                    elif device.lower() == 'c200':
                        array_data[array_id][array_row_start:array_row_end,array_col_start:array_col_end] = wd[input_row_start:input_row_end,input_col_start:input_col_end]
                    # systemc weight 需要transpose一下方便计算
                    systemc_weight_data[name+f":{systemc_id}"] = wd[input_row_start:input_row_end,input_col_start:input_col_end].transpose(1,0)
                systemc_id += 1

    return array_data,systemc_weight_data

def Hardware_adaptive_split_weight(onnx_weight, array_size=[576, 128], bn_split_layer_dict=None, split_method='uniform'):
    
    array_data = {}
    split_layer = []
    for k,v in onnx_weight.items():
        
        layer_name = k.split('.')[0] 
        data_shape = v.shape
        IsNeedSplit = False
        
        # 跳过batchnorm层
        if 'bn' in k or 'BatchNormalization' in k:
            if f'{layer_name}_bn.weight' not in onnx_weight.keys():
                array_data[k] = v
            if layer_name in bn_split_layer_dict.keys():
                output_split_num = 0
                for name in bn_split_layer_dict[layer_name]:
                    osn = int(name.split('_')[-1])
                    if osn > output_split_num:
                        output_split_num = osn
                output_split_num += 1
                # 拆分权重
                bn_weight = onnx_weight[f'{layer_name}.weight']
                bn_bias = onnx_weight[f'{layer_name}.bias']
                bn_mean = onnx_weight[f'{layer_name}.running_mean']
                bn_var = onnx_weight[f'{layer_name}.running_var']
                assert bn_weight.shape[0] % output_split_num == 0
                oc = bn_weight.shape[0] // output_split_num
                for name in bn_split_layer_dict[layer_name]:
                    osn = int(name.split('_')[-1])
                    array_data[f'{name}.weight'] = bn_weight[osn*oc:(osn+1)*oc]
                    array_data[f'{name}.bias'] = bn_bias[osn*oc:(osn+1)*oc]
                    array_data[f'{name}.mean'] = bn_mean[osn*oc:(osn+1)*oc]
                    array_data[f'{name}.var'] = bn_var[osn*oc:(osn+1)*oc]
                
            continue
        
        # constant
        if 'x_Constant' in k:
            array_data[k] = v
            continue
        
        # layernorm层
        if 'LayerNormalization' in k:
            array_data[k] = v
            continue
        
        # 跳过激活层
        if 'quantizer' in k and 'Silu' not in k:
            continue
        
        if 'weight' in k:
            in_row = 0
            if len(data_shape) == 4:
                [oc, ic, h1, h2] = data_shape
                in_row = ic * h1 * h2
                if in_row > array_size[0] or oc > array_size[1]:
                    IsNeedSplit = True
            elif len(data_shape) == 2:
                [oc, ic] = data_shape
                in_row = ic
                if ic > array_size[0] or oc > array_size[1]:
                    IsNeedSplit = True
            else:
                raise ValueError(f'权重维度 {data_shape} 不支持 !!!')
            
            if IsNeedSplit:
                if split_method == 'uniform':
                    row_split_num = math.ceil(in_row / array_size[0])
                    col_split_num = math.ceil(oc / array_size[1])
                    # assert in_row % row_split_num == 0
                    # assert oc % col_split_num == 0
                    _, row_value = get_split_num(ic, row_split_num)
                    # if layer_name == 'Conv_863':
                    #     print(data_shape)
                    #     print(in_row)
                    #     print(row_split_num)
                    #     print(row_value)
                    #     input()
                    _, col_value = get_split_num(oc, col_split_num)
                    max_row = np.array(row_value).max()
                    max_col = np.array(col_value).max()
                    
                    for rn in range(row_split_num):
                        for cn in range(col_split_num):
                            start_row = int(np.sum(np.array(row_value[:rn])))
                            end_row =  int(start_row + row_value[rn])
                            start_col =  int(np.sum(np.array(col_value[:cn])))
                            end_col =  int(start_col + col_value[cn])
                            
                            if len(data_shape) == 4:
                                array_data[f'{layer_name}_{rn}_{cn}.weight'] = onnx_weight[k][start_col:end_col,start_row:end_row,:,:]
                            elif len(data_shape) == 2:
                                array_data[f'{layer_name}_{rn}_{cn}.weight'] = onnx_weight[k][start_col:end_col,start_row:end_row]
                            
                            if rn == row_split_num - 1:    
                                if f'{layer_name}.bias' in onnx_weight.keys():
                                    array_data[f'{layer_name}_{rn}_{cn}.bias'] = onnx_weight[f'{layer_name}.bias'][start_col:end_col]
                            else:
                                if f'{layer_name}.bias' in onnx_weight.keys():
                                    array_data[f'{layer_name}_{rn}_{cn}.bias'] = np.zeros(col_value[cn])
                            
                            if len(data_shape) == 4:
                                # 如果切分之后的通道数不均衡, 则进行补0操作, 以通道数最大的对齐
                                if (end_row - start_row) < max_row:
                                    w_oc, w_ic , w_h, w_w = array_data[f'{layer_name}_{rn}_{cn}.weight'].shape
                                    diff_row = max_row - (end_row - start_row)
                                    diff_data = np.zeros((w_oc, diff_row, w_h, w_w))
                                    array_data[f'{layer_name}_{rn}_{cn}.weight'] = np.concatenate([array_data[f'{layer_name}_{rn}_{cn}.weight'], diff_data], axis=1)
                                
                                if (end_col - start_col) < max_col:
                                    w_oc, w_ic , w_h, w_w = array_data[f'{layer_name}_{rn}_{cn}.weight'].shape
                                    diff_col = max_col - (end_col - start_col)
                                    diff_data = np.zeros((diff_col, w_ic, w_h, w_w))
                                    array_data[f'{layer_name}_{rn}_{cn}.weight'] = np.concatenate([array_data[f'{layer_name}_{rn}_{cn}.weight'], diff_data], axis=0)
                            else:
                                raise ValueError(f'暂不支持 二维权重补零操作 {data_shape} !!!')
                            
                            # 拆分batchnorm层参数
                            if f'{layer_name}_bn.weight' in onnx_weight.keys():
                                for ln in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                                    if f'{layer_name}_bn.{ln}' not in onnx_weight.keys():
                                        warnings.warn(f'缺少 {layer_name}_bn.{ln} 权重参数')
                                        continue
                                    if ln == 'num_batches_tracked':
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}']
                                    elif ln in ['bias', 'running_mean']:
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}'][start_col:end_col] / row_split_num
                                    else:
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}'][start_col:end_col]
                            
                            # 拆分scale参数
                            if f'{layer_name}.a_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}.a_quantizer.s'] = onnx_weight[f'{layer_name}.a_quantizer.s']
                            if f'{layer_name}.w_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}.w_quantizer.s'] = onnx_weight[f'{layer_name}.w_quantizer.s']
                            if f'{layer_name}.a_out_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}.a_out_quantizer.s'] = np.array(onnx_weight[f'{layer_name}.a_out_quantizer.s'] / row_split_num)
                            if f'{layer_name}_bn.a_out_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}_bn.a_out_quantizer.s'] = np.array(onnx_weight[f'{layer_name}_bn.a_out_quantizer.s'] / row_split_num)
                            
                else:
                    raise ValueError(f'暂不支持 拆分方式 {split_method} !!!')
                split_layer.append(layer_name)
            else:
                array_data[k] = v
                # 不拆分batchnorm层参数
                if f'{layer_name}_bn.weight' in onnx_weight.keys():
                    for ln in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        if f'{layer_name}_bn.{ln}' not in onnx_weight.keys():
                            warnings.warn(f'缺少 {layer_name}_bn.{ln} 权重参数')
                            continue
                        array_data[f'{layer_name}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}']

                # 不拆分scale参数
                if f'{layer_name}.a_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}.a_quantizer.s'] = onnx_weight[f'{layer_name}.a_quantizer.s']
                if f'{layer_name}.w_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}.w_quantizer.s'] = onnx_weight[f'{layer_name}.w_quantizer.s']
                if f'{layer_name}.a_out_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}.a_out_quantizer.s'] = onnx_weight[f'{layer_name}.a_out_quantizer.s'] 
                if f'{layer_name}_bn.a_out_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}_bn.a_out_quantizer.s'] = onnx_weight[f'{layer_name}_bn.a_out_quantizer.s']
                
        elif 'bias' in k:
            if layer_name not in split_layer:
                array_data[k] = v
        elif 'Silu' in k:
            array_data[k] = v
        else:
            raise ValueError(f'暂无法解析 数据类型{k}!!!')
    return array_data

def get_split_num(ic, split_num):
    '''
    input:
        ic: 一个整数
        split_num: 将ic拆分的份数，使得每一份尽可能相同
    return:
        max_num:拆分之后，最大的份数
        num: 拆分之后的各个整数
    '''
    t = int(math.ceil(ic / split_num))
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
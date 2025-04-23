from onnx import numpy_helper, helper
import pickle
import numpy as np
import onnx
import math

def data_quantization_sym(data_float, half_level=15, data_range=None, isint=1, clamp_std=0, boundary_refine=False, reg_shift_mode=False, reg_shift_bits=None):
    data_float = data_float + 0
    if half_level <= 0:
        return (data_float, 1)
    if boundary_refine:
        pass
    if clamp_std:
        std = data_float.std()
        data_float[data_float < clamp_std * -std] = clamp_std * -std
        data_float[data_float > clamp_std * std] = clamp_std * std
    if data_range == None or data_range == 0:
        data_range = abs(data_float).max()
    if data_range == 0:
        return (data_float, 1)
    if reg_shift_mode:
        if reg_shift_bits != None:
            quant_scale = 2 ** reg_shift_bits
        else:
            shift_bits = round(math.log(1 / data_range * half_level, 2))
            quant_scale = 2 ** shift_bits
        data_quantized = (data_float * quant_scale).astype(np.int)
    else:
        data_quantized = np.clip(data_float / data_range * half_level, -half_level, half_level).round()
        quant_scale = 1 / data_range * half_level
    if isint == 0:
        data_quantized = data_quantized * data_range / half_level
        quant_scale = 1
    return (data_quantized, quant_scale)

def pickle_save(data, file, **kwargs):
    with open(file, 'wb') as f:
        pickle.dump(data, f, **kwargs)

def get_hardware_shape(device_name, device):
    chip_name = device_name[0] + '[0]'
    device_shape = device.devices[chip_name]['attributes']['shape']
    k = device_shape[0]
    device_shape[0] = device_shape[1]
    device_shape[1] = k
    return device_shape

def get_hardware_name(device_):
    max = 1
    device_list = list(device_.all_devices)
    for device in device_list:
        name_list = device.split('.')
        if len(name_list) > max:
            max = len(name_list)
            max_list = name_list
    max_list.pop(0)
    device_name = []
    for name in max_list:
        device_name.append(name.split('[')[0])
    return device_name

def fix_node_name(model):
    
    new_node_list = []
    type_index = {}
    for node in model.graph.node:
        if node.op_type not in type_index.keys():
            type_index[node.op_type] = 0
        node_name = node.op_type + '_' + str(type_index[node.op_type])
        type_index[node.op_type] += 1
        attr_dict = {}
        for a in node.attribute:
            if node.op_type == 'Concat':
                attr_dict[a.name] = a.i
            elif a.name in ['group', 'ceil_mode', 'axis']:
                attr_dict[a.name] = a.i
            elif a.type == 7:
                attr_dict[a.name] = a.ints
            elif a.type == 4:
                attr_dict[a.name] = a.t
            elif a.type == 1:
                attr_dict[a.name] = a.f
            elif a.type == 3:
                attr_dict[a.name] = a.s
        new_node = helper.make_node(node.op_type, node.input, node.output, node_name, doc_string=None, **attr_dict)
        new_node_list.append(new_node)
    model_graph = helper.make_graph(new_node_list, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer, doc_string=None, value_info=model.graph.value_info)
    model = helper.make_model(model_graph)
    return model

def add_value_info_for_constants(model: onnx.ModelProto):
    

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            if init.name in inputs:
                continue
            elem_type = init.data_type
            shape = init.dims
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField('shape'):
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim
        for node in graph.node:
            for attr in node.attribute:
                if attr.ref_attr_name != '':
                    continue
                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)
        return helper.make_model(graph)
    return add_const_value_infos_to_graph(model.graph)

def save_onnx_model(model, filename):
    
    onnx.save(model, filename)

def get_npu_compatible_model(model, model_config, output_npu_model_filename):
    
    graph_input_name = model_config['input'][0]
    if type(graph_input_name) == list:
        if len(graph_input_name) > 1:
            raise ValueError('暂只支持一个输入标签')
    graph_output_name = model_config['output'][0]
    if type(graph_input_name) == list:
        if len(graph_output_name) > 1:
            raise ValueError('暂只支持一个输出标签')
    inputs = []
    outputs = []
    for n in model.graph.node:
        for p in n.input:
            inputs.append(p)
        for l in n.output:
            outputs.append(l)
    if graph_input_name not in inputs:
        raise ValueError(f'不存在输入名称为：{graph_input_name} 的节点')
    if graph_output_name not in outputs:
        raise ValueError(f'不存在输出名称为：{graph_output_name} 的节点')
    multi_input_to_out_dict = {}
    npu_compatible_value_info_list = []
    npu_compatible_node_list = []
    input_value_info = []
    output_value_info = []
    initializer_list = []
    input_name = model_config['input'][0]
    for i in model.graph.node:
        if i.op_type != 'Constant' and i.input[0] == graph_input_name:
            if len(in_to_out(input_name, model)[input_name]) > 1:
                multi_input_to_out_dict = {**in_to_out(input_name, model), **multi_input_to_out_dict}
            if i.output[0] == graph_output_name:
                break
            input_name = i.output[0]
            continue
    value_info = {}
    for v in model.graph.value_info:
        value_info[v.name] = v
    for i in model.graph.node:
        if i.op_type != 'Constant' and i.input[0] == graph_input_name:
            if i.input[0] not in value_info.keys():
                input_value_info.append(model.graph.input[0])
                npu_compatible_node_list.append(i)
                for t in range(len(i.input) - 1):
                    npu_compatible_value_info_list.append(value_info[i.input[t + 1]])
            else:
                input_value_info.append(value_info[i.input[0]])
                npu_compatible_node_list.append(i)
                for t in range(len(i.input) - 1):
                    npu_compatible_value_info_list.append(value_info[i.input[t + 1]])
        if i.op_type != 'Constant' and i.input[0] in multi_input_to_out_dict.keys():
            for n in model.graph.node:
                if n.output[0] in multi_input_to_out_dict[i.input[0]]:
                    npu_compatible_node_list.append(n)
                    for k in model.graph.value_info:
                        if k.name in n.input:
                            if k not in npu_compatible_value_info_list:
                                npu_compatible_value_info_list.append(k)
        elif graph_input_name not in i.input and npu_compatible_node_list != []:
            npu_compatible_node_list.append(i)
            for j in i.input:
                npu_compatible_value_info_list.append(value_info[j])
        if i.op_type == 'Constant':
            npu_compatible_node_list.append(i)
        if i.output[0] == graph_output_name:
            if i.output[0] != model.graph.output[0].name:
                output_value_info.append(value_info[i.output[0]])
            else:
                output_value_info.append(model.graph.output[0])
            break
    for node in npu_compatible_node_list:
        if node.op_type in ['Conv', 'MatMul', 'Gemm', 'Reshape', 'ConvTranspose', 'Mul', 'PRelu']:
            for tensor in model.graph.initializer:
                if tensor.name in node.input:
                    initializer_list.append(tensor)
    npu_compatible_graph = helper.make_graph(npu_compatible_node_list, model.graph.name, input_value_info, output_value_info, initializer_list, doc_string=None, value_info=npu_compatible_value_info_list)
    npu_model = helper.make_model(npu_compatible_graph)
    if output_npu_model_filename:
        save_onnx_model(npu_model, output_npu_model_filename)
    return npu_model

def value_info_has_shape(value_info):
    
    if list(value_info.type.tensor_type.shape.dim) != []:
        return True
    else:
        return False

def in_to_out(input_name, model):
    
    in_out_dict = {}
    out_list = []
    for i in model.graph.node:
        if i.op_type != 'Constant' and i.input[0] == input_name:
            out_list.append(i.output[0])
    in_out_dict[input_name] = out_list
    return in_out_dict

def get_model_constant_node(model):
    
    nodes_output_dict = {}
    for i in model.graph.node:
        if i.op_type == 'Constant':
            nodes_output_dict[i.output[0]] = i
    return nodes_output_dict

def in_to_node(output_name, model):
    node_list = []
    constant_node = get_model_constant_node(model)
    initializer_output_name = get_model_initializer_value(model)
    sp_output_name_list = []
    for i in model.graph.node:
        if i.op_type not in ['Constant', 'Shape']:
            for k in i.input:
                if k in output_name:
                    node_list.append(i)
                    if len(i.input) > 1:
                        for j in i.input:
                            if j in constant_node.keys():
                                node_list.append(constant_node[j])
                            elif j not in initializer_output_name.keys() and j not in output_name:
                                sp_output_name_list.append(j)
    return (node_list, sp_output_name_list)

def dim_to_list(dim):
    
    dim_list = []
    for i in dim:
        dim_list.append(i.dim_value)
    return dim_list

def sort_index(node_list):
    
    for i in range(len(node_list) - 1):
        for j in range(len(node_list) - i - 1):
            if int(node_list[j]) > int(node_list[j + 1]):
                (node_list[j], node_list[j + 1]) = (node_list[j + 1], node_list[j])
    return node_list

def get_conv_node_attr(attr):
    
    has_stride = False
    has_pad = False
    for i in attr:
        if i.name == 'strides':
            stride = i.ints
            has_stride = True
        if i.name == 'pads':
            pad = i.ints
            has_pad = True
        if i.name == 'kernel_shape':
            kernel = i.ints
    if has_pad and has_stride:
        return (stride, pad, kernel)
    elif has_pad:
        return ([0, 0], pad, kernel)
    elif has_stride:
        return (stride, [0, 0], kernel)
    else:
        return ([0, 0], [0, 0], kernel)

def get_split(node):
    
    for attr in node.attribute:
        if attr.name == 'split':
            return attr.ints

def get_axis(node):
    
    for attr in node.attribute:
        if attr.name == 'axis':
            return attr.i
    return 0

def get_new_axis(node):
    
    for attr in node.attribute:
        if attr.name == 'new_axis':
            return attr.i

def get_fmod(node):
    
    for attr in node.attribute:
        if attr.name == 'fmod':
            return attr.i

def get_attr_max(node):
    
    for attr in node.attribute:
        if attr.name == 'max':
            return attr.f

def get_attr_min(node):
    
    for attr in node.attribute:
        if attr.name == 'min':
            return attr.f

def get_alpha(node):
    
    for attr in node.attribute:
        if attr.name == 'alpha':
            return attr.f

def get_perm(node):
    
    for attr in node.attribute:
        if attr.name == 'perm':
            return list(attr.ints)

def get_axes(node):
    
    for attr in node.attribute:
        if attr.name == 'axes':
            return attr.ints
    return []

def get_keepdims(node):
    
    for attr in node.attribute:
        if attr.name == 'keepdims':
            return attr.i
    return 1

def get_select_last_index(node):
    
    for attr in node.attribute:
        if attr.name == 'select_last_index':
            return attr.ints[0]
    return 0

def get_conv_dilations(attr):
    
    for i in attr:
        if i.name == 'dilations':
            dilations = i.ints
    return dilations

def get_conv_group(node, value_info):
    
    node_input_channel = dim_to_list(value_info[node.input[0]].type.tensor_type.shape.dim)[1]
    node_weight_channel = dim_to_list(value_info[node.input[1]].type.tensor_type.shape.dim)[1]
    group = node_input_channel // node_weight_channel
    assert node_input_channel % node_weight_channel == 0
    return group

def get_allowzero(node):
    
    allowzero = 0
    for i in node.attribute:
        if i.name == 'allowzero':
            allowzero = i.ints[0]
    return allowzero

def get_pad_mode(node):
    
    mode = 'constant'
    for i in node.attribute:
        if i.name == 'mode':
            mode = i.s
    return mode

def get_pad_value(node):
    
    value = 0
    for i in node.attribute:
        if i.name == 'value':
            value = i.f
    return value

def get_direction(node):
    
    for i in node.attribute:
        if i.name == 'direction':
            direction = i.ints[0]
    return direction

def get_model_value_info(model):
    
    value_info = {}
    for v in model.graph.value_info:
        value_info[v.name] = v
    for v in model.graph.input:
        value_info[v.name] = v
    for v in model.graph.output:
        value_info[v.name] = v
    return value_info

def get_model_initializer_value(model, value_type='TensorProto'):
    
    init_dict = {}
    if value_type == 'Numpy':
        for init in model.graph.initializer:
            init_dict[init.name] = numpy_helper.to_array(init)
    elif value_type == 'TensorProto':
        for init in model.graph.initializer:
            init_dict[init.name] = init
    else:
        raise ValueError('Only support two data type: Numpy or TensorProto!')
    return init_dict

def get_model_constant_value(model, OutputNode=False):
    
    constant_dict = {}
    for c in model.graph.node:
        if c.op_type == 'Constant':
            for attr in c.attribute:
                if OutputNode:
                    constant_dict[c.output[0]] = numpy_helper.to_array(attr.t)
                else:
                    constant_dict[c.name] = numpy_helper.to_array(attr.t)
    return constant_dict

def get_node_pads(node):
    
    pads = []
    for i in node.attribute:
        if i.name == 'pads':
            pads = i.ints
    return pads

def get_node_epsilon(node):
    
    pads = 0
    for i in node.attribute:
        if i.name == 'epsilon':
            pads = i.f
    return pads

def get_pool_attr(node):
    
    kernel_shape = []
    pads = []
    strides = []
    for i in node.attribute:
        if i.name == 'kernel_shape':
            kernel_shape = i.ints
        if i.name == 'pads':
            pads = i.ints
        if i.name == 'strides':
            strides = i.ints
    return (kernel_shape, pads, strides)

def hardmax(x, axis=-1):
    x_argmax = np.argmax(x, axis=axis)
    y = np.zeros_like(x)
    np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
    return y

def logsoftmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return x - x_max - np.log(s)

def get_input_info(input_shape, name):
    
    t = {}
    if len(input_shape) == 4:
        t = dict(ref=name, channel=input_shape[1], height=input_shape[2], width=input_shape[3])
    elif len(input_shape) == 2 or len(input_shape) == 1:
        t = dict(ref=name, channel=input_shape[-1], height=1, width=1)
    else:
        t = dict(ref=name, shape=input_shape)
    return t

def get_output_info(out_shape):
    
    outputs_ = []
    if len(out_shape) == 4:
        outputs_ = [{'channel': out_shape[1], 'height': out_shape[2], 'width': out_shape[3]}]
    elif len(out_shape) == 2:
        outputs_ = [{'channel': out_shape[1], 'height': 1, 'width': 1}]
    elif len(out_shape) == 1:
        outputs_ = [{'channel': out_shape[0], 'height': 1, 'width': 1}]
    else:
        outputs_ = [{'shape': out_shape}]
    return outputs_

def get_weight_info(weight_shape, bias_shape=None):
    
    weights_ = {}
    if bias_shape != None:
        weights_['bias'] = {'shape': bias_shape}
    weights_['weight'] = {'shape': weight_shape}
    return weights_

def get_reshape_shape(shape):
    
    shape_ = []
    for i in shape.int64_data:
        shape_.append(i)
    return shape_

def get_hidden_size(node):
    
    hidden_size = 0
    for i in node.attribute:
        if i.name == 'hidden_size':
            hidden_size = i.i
    return hidden_size
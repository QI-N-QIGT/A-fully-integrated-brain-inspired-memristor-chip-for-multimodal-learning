import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import numpy as np
from dummy_npu.data_io import pickle_load

def make_tensor(value_name, value):
    d_ = str(value.dtype).upper()
    if d_ == 'FLOAT32':
        d_ = 'FLOAT'
    dtype = TensorProto.DataType.Value(d_)
    tensor = helper.make_tensor(name=value_name, data_type=dtype, dims=value.shape, vals=value.reshape(-1))
    return tensor

def quantization(data, q_bit, percent=1, max_val=None):
    if max_val == None:
        min_val = data.min()
        max_val = data.max()
    else:
        min_val = -max_val
        max_val = max_val
    max_val_abs = max(abs(min_val), abs(max_val))
    qmax = 2.0 ** q_bit - 1.0
    scale = max_val_abs * percent / qmax
    q_x = data / scale
    q_x = np.clip(q_x, a_min=-qmax, a_max=qmax)
    q_x = q_x.round()
    q_x = scale * q_x
    return q_x
model = onnx.load('model_with_fixed_name.onnx')
new_dict = []
node_list = list(model.graph.node)
weight = pickle_load('weight_202203111252.pkl')
value_conv1 = quantization(weight['conv1.weight'], 3, max_val=1)
value_conv2 = quantization(weight['conv2.weight'], 3, max_val=1)
value_fc = quantization(weight['13'], 3, max_val=1)
for i in model.graph.initializer:
    if 'conv1.weight' in i.name:
        value = value_conv1
        tensor = make_tensor(i.name, value)
    elif 'conv2.weight' in i.name:
        value = value_conv2
        tensor = make_tensor(i.name, value)
    elif '14' in i.name:
        value = value_fc.transpose(1, 0)
        tensor = make_tensor(i.name, value)
    else:
        tensor = i
    new_dict.append(tensor)
npu_compatible_graph = helper.make_graph(node_list, model.graph.name, model.graph.input, model.graph.output, new_dict, doc_string=None, value_info=model.graph.value_info)
npu_model = helper.make_model(npu_compatible_graph)
npu_model.ir_version = 6
npu_model.opset_import[0].version = 11
onnx.save(npu_model, 'cnn_0628_modified.onnx')
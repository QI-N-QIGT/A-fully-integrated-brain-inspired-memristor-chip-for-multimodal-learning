import time
import numpy as np
from model2ir.onnx2ir.converter import ConvertONNX
from tqdm import tqdm
use_torch = False
try:
    import torch
    from cimruntime.A111.simulation.a111_sim_rt import A111TorchRT
    use_torch = True
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    rt = A111TorchRT(device=device_str)
except:
    print()
    from cimruntime.c200.c200_rt import C200NumpyRT
    from cimruntime.cimtensor import CIMNumpyTensor
    rt = C200NumpyRT(multi_batch=True)

def trans_pt_layer_info2onnx_layer_info(quant_info, onnx_key2pt_key):
    pt_key2_onnx_key = {}
    for (k, v) in onnx_key2pt_key.items():
        if v == 'onnx::Conv_84':
            v = 'conv1'
        elif v == 'onnx::MatMul_86':
            v = 'fc'
        pt_key2_onnx_key[v.rsplit('.weight', 1)[0]] = k
    layers_info = {}
    for (k, v) in quant_info.items():
        ks = k.rsplit('.', 1)
        (pt_key, attr) = ks
        onnx_key = pt_key2_onnx_key[pt_key]
        if onnx_key not in layers_info:
            layers_info[onnx_key] = {}
        layers_info[onnx_key][attr] = v
    return layers_info

def twn_n(s, max_k=8):
    assert s >= 1, "twn_n func's input s need tobe >= 1. "
    for k in range(0, max_k):
        if 2 ** k <= s < 2 ** (k + 1):
            if s / 2 ** k <= 2 ** (k + 1) / s:
                return k
            else:
                return k + 1
    return max_k
care_name_input_shape = {}

def callback(name, layer, inputs, weights, attrs, outputs, layer_quant_info=None, **kwargs):
    if outputs == None:
        if 'Conv_' in name or 'MatMul_' in name:
            data = inputs[0]
            if isinstance(data, tuple):
                data = data[0]
            care_name_input_shape[name] = list(data.data.shape)

def get_output_from_specified_node(onnx_obj, input_node_and_data, output_node, onnx_layer_info, func=None, save_path=None, batch_size=64, paras=None):
    begin = time.time()
    onnx_ir = onnx_obj.ir
    onnx_weight_data = onnx_obj.model_parser.weight_numpy
    if use_torch:
        onnx_weight_data = {k: torch.from_numpy(v.copy().astype(np.float32)).to(device) for (k, v) in onnx_weight_data.items()}
    if isinstance(output_node, str):
        output_node = [output_node]
    elif isinstance(output_node, (tuple, list, set)):
        output_node = set(output_node)
        output_node = list(output_node)
    temp_d = list(input_node_and_data.values())[0]
    if isinstance(temp_d, dict):
        temp_d = list(temp_d.values())[0]
    sample_num = len(temp_d)
    iters = int((sample_num - 1) // batch_size) + 1
    data_ = {n: {'data': [], 's': 0} for n in output_node}
    for i in range(iters):
        input_1 = {}
        for (key, value) in input_node_and_data.items():
            if value is not None:
                if use_torch:
                    if isinstance(value, dict):
                        data = value['data']
                        data_s = value['s']
                        input_1_data = torch.from_numpy(data[i * batch_size:(i + 1) * batch_size, ...]).to(device)
                        input_1[key] = (input_1_data, torch.tensor([data_s]).to(device))
                    else:
                        input_1[key] = torch.from_numpy(value[i * batch_size:(i + 1) * batch_size, ...]).to(device)
                else:
                    input_1[key] = CIMNumpyTensor.to_cimtensor(data=value[i * batch_size:(i + 1) * batch_size], multi_batch=True)
            else:
                input_1[key] = ''
        if use_torch:
            with torch.no_grad():
                output = rt.run_ir(onnx_ir, input_1, onnx_weight_data, outputs=output_node, callback=func, layer_quant_info=onnx_layer_info, **paras)
        else:
            output = rt.run_ir(onnx_ir, input_1, onnx_weight_data, outputs=output_node, callback=func, layer_quant_info=onnx_layer_info, **paras)
        for n in output_node:
            s = 1
            if use_torch:
                temp = output[n]
                if isinstance(temp, tuple):
                    s = temp[1].item() if isinstance(temp[1], torch.Tensor) else temp[1]
                    temp = temp[0]
                re = temp.detach().cpu().numpy()
            else:
                re = CIMNumpyTensor.scale_recover(output[n]).data
            data_[n]['data'].append(re)
            data_[n]['s'] = s
    node_output = {n: {} for n in output_node}
    for n in output_node:
        node_output[n]['data'] = np.concatenate(data_[n]['data'], axis=0)
        node_output[n]['s'] = data_[n]['s']
    data = node_output[output_node[0]] if len(output_node) == 1 else node_output
    del node_output, data_
    if save_path is not None:
        np.save(save_path, data)
    end = time.time()
    return data

def get_sim_acc(onnx_obj, onnx_layer_info, pt_layer1_input, pt_label):
    input_node_name = 'MaxPool_0'
    output_node_name = 'MatMul_0'
    input_data = pt_layer1_input
    input_node_and_data = {input_node_name: input_data}
    matmul_50_output = get_output_from_specified_node(onnx_obj, input_node_and_data, output_node_name, onnx_layer_info, callback, paras=onnx_layer_info)
    (premax_num, top1, top5) = cal_accuracy_use_numpy(pt_label, matmul_50_output)
    print()

def get_onnx_obj(model_onnx_path):
    onnx_obj = ConvertONNX(model_onnx_path, fix_layer_name=True)
    if not use_torch:
        rt.init_rpc(onnx_obj.ir, simulation=True)
    return onnx_obj

def cal_accuracy_use_pytorch(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        (values, pred) = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        premax_num = (output == values).sum()
        print()
        return res

def cal_accuracy_use_pytorch1(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        (values, pred) = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        premax_num = (output == values).sum()
        res += [premax_num.item()]
        return (res[0].item() / 100, res[1])

def cal_accuracy_use_numpy(targets, rram_output):
    argsort = np.argsort(-rram_output, axis=1)[:, :rram_output.shape[-1]]
    predictions = argsort == targets[:, None].repeat(rram_output.shape[-1], axis=1)
    top1 = float(predictions[:, 0].sum()) / len(predictions)
    top5 = float(predictions.sum()) / len(predictions)
    premax = np.max(rram_output, axis=1)
    premax_num = (rram_output == premax[:, None]).sum()
    return (premax_num, top1, top5)
from cimruntime.A111.simulation.a111_sim_rt import A111TorchRT as ATT
import torch.nn.functional as F
import torch
from model2ir.onnx2ir.converter import ConvertONNX
from dummy_npu.data_io import pickle_load
import time
import numpy as np
paras = pickle_load('new_example\\quant_args_input_7_15_adc_range_16_128_hard_scale_method1.pickle')
model = 'new_example\\model_with_fixed_name.onnx'
onnx_obj = ConvertONNX(model, fix_layer_name=False)
onnx_ir = onnx_obj.ir
onnx_ir.dump_json(file='new_example\\1.yaml')
onnx_weight_data = onnx_obj.model_parser.weight_numpy
for i in onnx_weight_data.keys():
    onnx_weight_data[i] = torch.from_numpy(onnx_weight_data[i].copy())
input_int = np.load('new_example\\features_save\\soft_ware_int_batch_0.npy')
input_scale = np.load('new_example\\features_save\\soft_ware_scale.npy')
input_ = (torch.from_numpy(input_int), torch.from_numpy(input_scale))
output_ = np.load('new_example\\features_save\\model_output_batch_0.npy')
rt = ATT()
batch_size = 10
batch_num = 1
data_re = []
time1 = time.time()
for i in range(batch_num):
    input_1 = {'Relu_8': input_}
    output = rt.run_ir(onnx_ir, input_1, onnx_weight_data, outputs=['MatMul_0'], **paras)
    re = output['MatMul_0']
    if isinstance(re, tuple) and len(re) == 2:
        re = re[0] * re[1]
    data_re.append(re)
time2 = time.time()
print()
data_re = torch.cat(data_re, axis=0)
print()
from matplotlib import pyplot as plt
data_re = data_re.cpu().detach().numpy()
print()
plt.scatter(data_re.flatten(), output_[0:batch_size].flatten())
plt.xlabel('A111 Simulator')
plt.ylabel('@NaMin Results')
plt.show()
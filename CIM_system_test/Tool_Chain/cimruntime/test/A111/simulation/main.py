from cimruntime.A111.simulation.a111_sim_rt import A111TorchRT as ATT 
import torch.nn.functional as F  
import torch
from model2ir.onnx2ir.converter import ConvertONNX
from dummy_npu.data_io import pickle_load
import time
import numpy as np

# 加载运行参数 来自算法 @NaMin
paras = pickle_load('new_example\\quant_args_input_7_15_adc_range_16_128_hard_scale_method1.pickle')

# 加载模型
# model = 'data\\model_float.onnx'
model = 'new_example\\model_with_fixed_name.onnx'

# onnx转ir
onnx_obj = ConvertONNX(model, fix_layer_name=False)
onnx_ir = onnx_obj.ir
onnx_ir.dump_json(file='new_example\\1.yaml')
onnx_weight_data = onnx_obj.model_parser.weight_numpy

# 转换权重为 torch.tensor
for i in onnx_weight_data.keys():
    onnx_weight_data[i] = torch.from_numpy(onnx_weight_data[i].copy())

# 加载 input
input_int = np.load('new_example\\features_save\\soft_ware_int_batch_0.npy')
input_scale = np.load('new_example\\features_save\\soft_ware_scale.npy')

input_ = (torch.from_numpy(input_int), torch.from_numpy(input_scale))
# input_ = torch.from_numpy(input_int * input_scale)

# input_int = np.load('new_example\\features_save\\fc_soft_ware_int_batch_0.npy')
# input_scale = np.load('new_example\\features_save\\fc_soft_ware_scale.npy')
# input_ = (torch.from_numpy(input_int), torch.from_numpy(input_scale))
# input_ = (torch.from_numpy(input_int * input_scale), 

# 输出 output
# output_int = np.load('new_example\\features_save\\layer3.1.conv1_soft_ware_int_batch_0.npy')
# output_scale = np.load('new_example\\features_save\\layer3.1.conv1_soft_ware_scale.npy')
# output_ = output_int * output_scale

# output_ = output_int

# if isinstance(output_, tuple):
#     output_ = output_[0] * output_[1]
output_ = np.load('new_example\\features_save\\model_output_batch_0.npy')

# runtime
rt = ATT()
batch_size = 10
batch_num = 1
data_re = []
time1 = time.time()
for i in range(batch_num):
    input_1 = {'Relu_8': input_}
    # input_1 = {'Flatten_0': input_}
    # outputs= [] 输出层名字
    output = rt.run_ir(onnx_ir, input_1, onnx_weight_data, outputs=["MatMul_0"], **paras)
    # 保存结果
    re = output["MatMul_0"]
    if isinstance(re, tuple) and len(re) == 2:
        re = re[0] * re[1]
    data_re.append(re)
    
time2 = time.time()
print(f'calculation time: {time2 - time1} s')
data_re = torch.cat(data_re, axis=0)
# torch.save(data_re, 'simulation_results_1images.pth')
print(data_re.shape)

# compare results
from matplotlib import pyplot as plt

data_re = data_re.cpu().detach().numpy()
# output_ = output_.detach().numpy()
print(f'Abs Max Error : {(data_re - output_[0:batch_size]).max() / data_re.max() * 100} %')
plt.scatter(data_re.flatten(), output_[0:batch_size].flatten())
plt.xlabel('A111 Simulator')
plt.ylabel('@NaMin Results')
plt.show()
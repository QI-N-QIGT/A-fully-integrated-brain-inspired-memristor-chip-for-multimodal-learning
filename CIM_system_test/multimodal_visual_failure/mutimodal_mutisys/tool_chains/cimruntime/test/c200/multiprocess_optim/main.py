from model2ir.onnx2ir.converter import ConvertONNX
from cimruntime.c200.c200_rt import C200NumpyRT
from cimruntime.cimtensor import CIMNumpyTensor as CNT
from cimruntime.gen_weight import gen_array_weight
from e100_irmapper.mapper import mapper
from e100_irmapper.device.c200 import C200CalcInfo
from e100_irmapper.device.c200 import MappedLayer
from e100_irmapper.placement import LowestLevelAlgorithm, OneOnOne
import numpy as np
from e100_irtool.core.ir import load_ir
from dummy_npu.data_io import pickle_save, pickle_load
from e100_iroptimizer.c200.optimizer import AccOptimzier
import time
import multiprocessing

def load_csv(fn, dtype='int32'):
    assert fn
    return np.loadtxt(fn, dtype=dtype, delimiter=',', ndmin=2)

def save_csv(fn, data):
    assert fn
    np.savetxt(fn, data, delimiter=',', fmt='%d')

def txt2numpy(fn):
    with open(fn, 'r') as f:
        data = f.readline()
        data_all = []
        while data:
            data = data.strip()
            data1 = data.split(',')
            if data1[-1] == '':
                data1.remove(data1[-1])
            new_data = []
            for i in data1:
                new_data.append(int(i))
            data = f.readline()
            data_all.append(new_data)
    return np.array(data_all)
if __name__ == '__main__':
    multiprocessing.freeze_support()
    cpu_layer = None
    device = [{'name': 'c200-0', 'kind': 'rram-144k-cluster', 'num': 58}]
    model_path = f'F:\\NE文章\\review\\model_zoo\\mnist\\3_layer_cnn\\'
    code_path = f'F:\\NE文章\\review\\code\\3_layer_cnn\\optimize_time\\'
    model = model_path + 'cnn.onnx'
    onnx_obj = ConvertONNX(model, weight_half_level=6)
    onnx_ir = onnx_obj.ir
    onnx_weight_data = onnx_obj.model_parser.weight_numpy
    onnx_weight_data_quant = onnx_obj.model_parser.weight_numpy_quant
    copy_para = None
    calc_info = C200CalcInfo(shift_expansion_mode='bit_pulse', output_half_level=31, adc_clamp=False, adc_quant=False, noise_scale=0.0)
    map = mapper(ir=onnx_ir, device=device, cpu_layer=cpu_layer, calc_info=calc_info, place_strategy=OneOnOne, average_copy=copy_para, runtime='simulation')
    map.run()
    mapped_ir = map.ir
    mapped_ir.layers = dict(mapped_ir.iter_layers(deep=False, sorted=True))
    (array_data, _) = gen_array_weight(mapped_ir, onnx_weight_data)
    test_input = pickle_load(model_path + 'DataSet\\input_test_binary.pkl')['input'][0:1000]
    device_adc_scale_data = {}
    for i in range(58):
        device_adc_scale_data[f'c200-0.rram-144k-cluster:{i}'] = 1
    rt = C200NumpyRT(mapped_array_data=array_data, device_adc_scale_data=device_adc_scale_data, quant_all=True, multi_batch=True, half_level=31)
    rt.init_rpc(mapped_ir, simulation=True)
    calc = True
    if calc:
        for batch_size in [1, 10, 100, 1000, 10000]:
            data_output = []
            time1 = time.time()
            for i in range(1):
                input_1 = CNT.to_cimtensor(data=test_input[i * batch_size:(i + 1) * batch_size, :, :, :], multi_batch=True)
                output = rt.run_ir(mapped_ir, input_1, onnx_weight_data, outputs=['MatMul_7'])
                re1 = CNT.scale_recover(output['MatMul_7']).data
                data_output.append(re1)
            time2 = time.time()
            print()
            data_output = np.concatenate(data_output, axis=0)
            len_ = data_output.shape[0]
            label = pickle_load(model_path + 'DataSet\\label_test_binary.pkl')['label'][0:len_]
            rram_re = np.argmax(data_output, axis=1)
            sum_ = np.sum(rram_re == label)
            re = round(sum_ / label.shape[0] * 100, 4)
        print(f'Final Accuracy: {re} %')
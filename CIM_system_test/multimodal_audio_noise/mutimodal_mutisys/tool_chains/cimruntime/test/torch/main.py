from model2ir.onnx2ir.converter import ConvertONNX
from cimruntime.torch import TorchRuntime as TRT
import time
import torch
if __name__ == '__main__':
    model = 'model.onnx'
    onnx_obj = ConvertONNX(model, store_intermediate_model=True)
    onnx_ir = onnx_obj.ir
    onnx_weight_data = onnx_obj.model_parser.weight_numpy
    onnx_weight_data_quant = onnx_obj.model_parser.weight_numpy_quant
    for i in onnx_weight_data.keys():
        onnx_weight_data[i] = torch.from_numpy(onnx_weight_data[i].copy()).to('cuda')
    test_input = torch.randn(size=(1, 3, 224, 224)).to('cuda')
    calc = True
    if calc:
        rt = TRT()
        time1 = time.time()
        batch_size = 30
        batch_num = 1
        data_Relu_41 = []
        for i in range(batch_num):
            input_1 = test_input
            output = rt.run_ir(onnx_ir, input_1, onnx_weight_data, outputs=['Relu_41'])
            re = output['Relu_41']
            data_Relu_41.append(re)
        time2 = time.time()
        print()
        data_Relu_41 = torch.cat(data_Relu_41, axis=0)
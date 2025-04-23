import torch
import os
from .helper import convert_module_to_object

def pt2onnx(pytorch_script_path, pytorch_model_name, input_shape, onnx_model_name=None):
    # 模型解析
    pt_model = convert_module_to_object(pytorch_script_path, pytorch_model_name)
    if input_shape == None:
        raise ValueError("缺少输入的维度，请输入模型的输入维度！！！")
    x = torch.randn(input_shape)
    file_path = os.getcwd() + '\\onnx_model\\'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if onnx_model_name == None:
        onnx_model_name = f"temp.onnx"
    torch.onnx.export(pt_model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                file_path + onnx_model_name,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                input_names = ['input'],   # the model's input names
                output_names = ['output'])
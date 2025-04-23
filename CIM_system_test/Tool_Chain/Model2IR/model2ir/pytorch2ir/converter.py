from pathlib import Path
from importlib import import_module
from .parser import PytorchModuleParser
from ..onnx2ir.helper import *
import torch
from ..onnx2ir.passop import *
from ..onnx2ir.shape_operation import *

class ConvertPytorchModule(object):
    '''
    Foo: 将pytorch 的模型首先转换成 ONNX.
    '''
    def __init__(self, pytorch_module = None, weight_file=None, ir_file=None, input_shape=None):
        self.pytorch_module = pytorch_module  
        self.ir = {}
        self.weight_file = weight_file
        self.ir_file = ir_file
        self.input_shape = input_shape
        
        self._convert()
        
    def _convert(self):
        
        if self.pytorch_module == None:
            raise ValueError("缺少输入，请指定pytorch module！！！")
        
        # 模型解析
        # module = import_module(self.pytorch_module)
        pt_model = self.pytorch_module
        if self.input_shape == None:
            raise ValueError("缺少输入的维度，请输入模型的输入维度！！！")
        x = torch.randn(self.input_shape, requires_grad=True)
        file_path = os.getcwd() + '\\temp\\'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.onnx.export(pt_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  file_path + f"temp.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])
         
        model = onnx.load(file_path + "temp.onnx")
        model = load_onnx_model(model)
        self.model_parser = PytorchModuleParser(model)
        
        # 保存权重
        if self.weight_file != None:
            pickle_save(self.model_parser.weight_numpy,self.weight_file)

        # IR生成
        self.ir = make_ir()
        # 输入层
        g_inputs = []
        for input_name in self.model_parser.inputs:
            input_value_info = self.model_parser.value_infos[input_name]
            input_shape = dim_to_list(input_value_info.type.tensor_type.shape.dim)
            if len(input_shape) == 4:
                in_channel,in_height,in_width = input_shape[1:]
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            elif len(input_shape) == 2:
                in_channel = input_shape[1]
                in_height,in_width = 1,1
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            else:
                raise ValueError(f"暂不支持该维度{input_shape}的数据格式！！！")
            g_inputs.append(temp_d)
        self.ir.add_layer('graph_input',type='input',inputs=g_inputs)
        
        # 中间的层 
        MakeIR = MakeIROp()
        for node_name in self.model_parser.nodes.keys():
            node = self.model_parser.nodes[node_name]
            op_type = node.op_type
            func = getattr(MakeIR,op_type,None)
            if func == None:
                raise ValueError(f'暂不支持的op : {op_type} !!!')
            func(self.ir,self.model_parser,node_name)
        
        # 输出层
        g_outputs = []
        for out_name in self.model_parser.outputs:
            out_value_info = self.model_parser.value_infos[out_name]
            out_shape = dim_to_list(out_value_info.type.tensor_type.shape.dim)
            ref_name = self.model_parser.predecessors[out_name][0].name
            if len(out_shape) == 4:
                out_channel,out_height,out_width = out_shape[1:]
                temp_d = dict(ref=ref_name,channel=out_channel,height=out_height,width=out_width,channel_last=True)
            elif len(out_shape) == 2:
                out_channel = out_shape[1]
                temp_d = dict(ref=ref_name,channel=out_channel,height=1,width=1,channel_last=True)
            else:
                raise ValueError(f"暂不支持该维度{out_shape}的数据格式！！！")
            g_outputs.append(temp_d)
        self.ir.add_layer('graph_output',type='output',inputs=g_outputs)
        
    
    def dump(self):
        if self.ir_file == None:
            # 如果没有输出文件路径，则默认输出到当前运行的文件路径下，文件名称与onnx模型文件名一致
            file_path = os.getcwd()
            ir_file = Path(file_path +'\\'+ self.pytorch_module.split('\\')[-1].split('.')[0] + '.yaml')
        else:
            ir_file =  self.ir_file
            
        self.ir.dump(file=ir_file)
from model2ir.pytorch2ir.converter import ConvertPytorchModule
from model2ir.pytorch2ir.helper import convert_module_to_object
path = 'test\\vgg8\\'
output_file = path + '\\' + 'nn_1.py'
pt_module = convert_module_to_object(output_file, 'MappedNet')
pt_obj = ConvertPytorchModule(pt_module, input_shape=[1, 3, 32, 32])
pt_ir = pt_obj.ir
pt_ir.dump_json(file=path + 'vgg8_split_layer.yaml')
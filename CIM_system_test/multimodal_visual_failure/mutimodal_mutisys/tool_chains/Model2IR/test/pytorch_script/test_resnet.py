from model2ir.pytorch2ir.converter import ConvertPytorchModule
from model2ir.pytorch2ir.helper import convert_module_to_object
module_path = 'test\\pytorch_script\\nn_0.py'
model_name = 'LBLNet0'
model = convert_module_to_object(module_path, model_name)
pt_obj = ConvertPytorchModule(model, input_shape=[1, 3, 32, 32])
ir = pt_obj.ir
ir.dump_json(file='test\\pytorch_script\\test_resnet.yaml')
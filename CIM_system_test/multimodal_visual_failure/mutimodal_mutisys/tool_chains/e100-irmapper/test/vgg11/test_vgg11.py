from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
model = 'test\\vgg11\\VGG11.onnx'
device2 = {'name': 'c200-0', 'kind': 'rram-144k-cluster', 'num': 200}
onnx_ir = make_onnx_ir(model)
device_ir = make_device_ir(onnx_ir, device2)
parser = IrParser(device_ir)
place = Base(parser.node_info, parser.node_weight, parser.hardware_config, place_strategy=OneOnOne)
place_info = place.node_mapping_info
split_info = place.split_num
copy_info = place.average_copy
mapped_ir = make_mapped_ir(device_ir, split_info, place_info, copy_info=copy_info)
mapped_ir.dump_json(file='test\\vgg11\\VGG11_mapped_ir.yaml')
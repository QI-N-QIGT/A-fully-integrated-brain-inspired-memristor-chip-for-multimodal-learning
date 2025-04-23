from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *

model = 'lenet.onnx'
device1 = {
    'name':'a111',
    'kind':'a111-npu',
    'num':1
}
device2 = {
    'name':'c200',
    'kind':'rram-144k',
    'num':20
}
# onnx_ir,model_parser = make_onnx_ir(model,return_weight=True)
t = ConvertONNX(model)
onnx_ir = t.ir
model_parser = t.model_parser

onnx_ir.dump_json(file='onnx_ir.yaml')
device_ir = make_device_ir(onnx_ir,device1)
device_ir.dump_json(file='device_ir.yaml')
parser = IrParser(device_ir)
place = Base(parser.node_info,parser.node_weight,parser.hardware_config,place_strategy=OneOnOne)

place_info = place.node_mapping_info
split_info = place.split_num
placed_node = place.placed_nodes

node_id = make_node_id(placed_node)
mapped_ir = make_mapped_ir(device_ir,split_info,place_info)
mapped_ir.dump_json(file='mapped_ir.yaml')

convert_weight_idealarch(model_parser,split_info,node_id,weight_file='weight.txt')
print(place_info)
print(node_id)
input()

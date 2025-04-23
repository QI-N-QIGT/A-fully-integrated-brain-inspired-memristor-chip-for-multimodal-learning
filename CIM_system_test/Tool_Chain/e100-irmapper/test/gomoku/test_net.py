from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *

model = 'net.onnx'
device1 = {
    'name':'a111',
    'kind':'a111-npu',
    'num':1
}
device2 = {
    'name':'c200',
    'kind':'rram-144k',
    'num':10
}
onnx_ir = make_onnx_ir(model)
device_ir = make_device_ir(onnx_ir,device2)
parser = IrParser(device_ir)
place = GreedySearch(parser.node_info,parser.node_weight,parser.hardware_config,place_strategy=DiagnanolPlacement)
place_info = place.node_mapping_info
split_info = place.split_num

mapped_ir = make_mapped_ir(device_ir,split_info,place_info)
mapped_ir.dump_json(file='net.yaml')

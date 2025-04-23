from numpy import average
from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *

from e100_irtool.core.ir import load_ir
device2 = [
    {
        'name':'c200-0',
        'kind':'rram-144k-cluster',
        'num':8,
        'ip':'192.168.2.99',
        'profile':{
            # 'in_channel': 320,
            # 'out_channel': 128,
            'in_bits': 7,
            'out_bits': 8,
            # 'weight_bits': 4,
            # 'signed': True
        }
    }
]
onnx_ir = load_ir(file="test\\base\\ir_conv.yaml")
device_ir = make_device_ir(onnx_ir,device2)
device_ir.dump_json(file=f'test\\base\\ir_conv_device_1.yaml')

# parser = IrParser(device_ir)
# average_copy = {"conv":[3,3]}
# para_ = {"conv":[2,1]}
# place = Base(parser.node_info,parser.node_weight,parser.hardware_config,average_copy=average_copy,specify_para_num=para_,place_strategy=OneOnOne)
# place_info = place.node_mapping_info
# split_info = place.split_num
# copy_info = place.average_copy
 
# mapped_ir = make_mapped_ir(device_ir,split_info,place_info,copy_info)
# mapped_ir.dump_json(file=f'test\\base\\ir_conv_mapped.yaml')

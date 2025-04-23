from numpy import average
from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
from e100_irtool.core.ir import load_ir
device2 = [{'name': 'c200-0', 'kind': 'rram-144k-cluster', 'num': 8, 'ip': '192.168.2.99', 'profile': {'in_bits': 7, 'out_bits': 8}}]
onnx_ir = load_ir(file='test\\base\\ir_conv.yaml')
device_ir = make_device_ir(onnx_ir, device2)
device_ir.dump_json(file=f'test\\base\\ir_conv_device_1.yaml')
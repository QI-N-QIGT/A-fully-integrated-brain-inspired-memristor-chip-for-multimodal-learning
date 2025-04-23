from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
from e100_irmapper.mapper import *
model = 'test\\unet\\unet.onnx'
device = [{'name': 'c200-0', 'kind': 'rram-144k-cluster', 'num': 22, 'ip': '192.168.2.98'}]
calc_info = C200CalcInfo(shift_expansion_mode='bit_shift', output_half_level=31)
onnx_obj = ConvertONNX(model, weight_half_level=6)
onnx_ir = onnx_obj.ir
onnx_weight_data = onnx_obj.model_parser.weight_numpy
map = mapper(ir=onnx_ir, device=device, calc_info=calc_info, place_strategy=OneOnOne, runtime='simulation')
mapped_ir = map.ir
ir = mapped_ir
ir.dump_json(file='test\\unet\\unet_calc_info_ir.yaml')
from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
from e100_irmapper.mapper import *
from e100_irtool.core.ir import load_ir
from e100_irmapper.device.a111 import A111CalcInfo

# tile_num = 6
# devices = []
# for i in range(tile_num):
#     devices.append({
#         'name':'a111-0,
#         'kind':'a111-npu',
#         'ip':'192.168.2.5'
#     })
devices = []
devices.append({
        'name':'a111-0',
        'kind':'a111-npu',
        'ip':'192.168.2.5'})  
onnx_ir = load_ir(file="test\\a111\\ir_fc_twolayer.yaml")

calc_info = A111CalcInfo(weight_scale = 1,
                assigned_output_quant_scale = 1,
                adc_range = 1,
                relu_threshold = 0,
                shift_num = 0)

map = mapper(ir=onnx_ir,device=devices,
                cpu_layer=None,
                calc_info=calc_info,
                weight_format='HWC',
                place_strategy=A111Placement,
                relu_fuse= True,
                pool_fuse= True,
                target_device='a111')

mapped_ir = map.ir

mapped_ir.dump_json(file=f'test\\a111\\ir_fc_twolayer_mapped_a111.yaml')

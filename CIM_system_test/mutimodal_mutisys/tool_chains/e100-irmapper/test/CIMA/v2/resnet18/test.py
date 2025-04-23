from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
from e100_irmapper.mapper import *
from e100_irtool.core.ir import load_ir
devices = {'name': 'cima-0', 'kind': 'cima-node', 'num': 36, 'height': 6, 'width': 6, 'task_num': 8}
model = 'resnet18.onnx'
onnx_obj = ConvertONNX(model)
onnx_ir = onnx_obj.ir
onnx_ir.dump_json(file=f'resnet18_ir.yaml')
map = mapper(ir=onnx_ir, device=devices, weight_format='HWC', place_strategy=CIMAPlacement, relu_fuse=True, pool_fuse=False, masked_device_id_list=[(3, 0), (0, 5), (1, 5), (4, 5), (5, 5)], target_device='cima')
map.run()
mapped_ir = map.ir
mapped_ir.dump_json(file=f'resnet18_mapped_ir_relu_fused_a_search.yaml')
draw_mesh_fig(map, mesh_width=6, save_fig_path='resnet_18_a_search.svg')
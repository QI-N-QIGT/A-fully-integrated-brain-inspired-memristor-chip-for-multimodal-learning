from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
import time
model = 'VGG11.onnx'
device1 = {'name': 'a111', 'kind': 'a111-npu', 'num': 1}
onnx_ir = make_onnx_ir(model)
device_ir = make_device_ir(onnx_ir, device1)
parser = IrParser(device_ir)
start = time.perf_counter()
place = Base(parser.node_info, parser.node_weight, parser.hardware_config, place_strategy=OneOnOne)
end = time.perf_counter()
print()
split_info = place.split_num
placed_node = place.placed_nodes
print()
print(split_info)
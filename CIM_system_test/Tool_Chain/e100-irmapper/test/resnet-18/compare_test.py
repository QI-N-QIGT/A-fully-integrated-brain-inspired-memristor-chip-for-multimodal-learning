from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
import time
model = 'resnet18_0.onnx'
device1 = {
    'name':'a111',
    'kind':'a111-npu',
    'num':1
}
onnx_ir = make_onnx_ir(model)
device_ir = make_device_ir(onnx_ir,device1)
parser = IrParser(device_ir)
start = time.perf_counter()
# place = Base(parser.node_info,parser.node_weight,parser.hardware_config,place_strategy=OneOnOne)
# place = GreedySearch(parser.node_info,parser.node_weight,parser.hardware_config,place_strategy=OneOnOne,window_copy=True)
ub_1 = [8] * 40
place = GeneticAlgorithm(parser.node_info,parser.node_weight,parser.hardware_config,ub=ub_1,lb=[1] * 40,precision=[1]*40,
                         size_pop=100,max_iter=100,place_strategy=OneOnOne,window_copy=True)
end = time.perf_counter()
print(f"time:{end - start}")
split_info = place.split_num
placed_node = place.placed_nodes
print(placed_node)
print(split_info)
# input()
from model2ir.onnx2ir.converter import ConvertONNX
from e100_irmapper.mapper import mapper
from e100_irmapper.device.a111 import MappedLayer #noqa
from e100_irmapper.placement import A111Placement
from cimruntime.A111.helper import get_a111_weight_mapping_info
    
if __name__ == "__main__":
    
    # 声明设备
    device = [{
        'name':'a111-0',
        'kind':'a111-npu',
        'num': 1
    }]
    
    model_path = f'test\\a111\\resblock\\'
    model = model_path + f'model_with_fixed_name.onnx'
        
    # onnx转ir
    onnx_obj = ConvertONNX(model, weight_half_level = 7, store_intermediate_model=False)
    onnx_ir = onnx_obj.ir
    onnx_weight_data = onnx_obj.model_parser.weight_numpy
    onnx_weight_data_quant = onnx_obj.model_parser.weight_numpy_quant
    
    # mapping
    map1 = mapper(ir=onnx_ir,device=device,
                weight_format='HWC',
                place_strategy = A111Placement,
                masked_device_id_list= ['Tile:1-XB:2'],
                target_device='a111'
                )
    map1.run()
    mapped_ir = map1.ir
    # mapped_ir.layers = dict(mapped_ir.iter_layers(deep=False, sorted=True)) 
    # mapped_ir.dump_json(file= model_path + f'resbblock_mapped_a111.yaml')
    
    # mapping info
    weight_info = get_a111_weight_mapping_info(mapped_ir, onnx_weight_data_quant, pos_sa=4, neg_sa=4)
    
    with open(model_path + 'log.txt', 'w') as f:
        print(weight_info,file=f)
from e100_irtool.core.ir import load_ir, BaseIR
from cimruntime.gen_weight import gen_array_weight

def get_a111_weight_mapping_info(mapped_ir, onnx_weight_data_quant, pos_sa=5, neg_sa=5):
    (array_data, _) = gen_array_weight(mapped_ir, onnx_weight_data_quant, format='HWC', device='a111', pos_sa=pos_sa, neg_sa=neg_sa)
    if isinstance(mapped_ir, str):
        ir = load_ir(mapped_ir)
    elif isinstance(mapped_ir, BaseIR):
        ir = mapped_ir
    else:
        raise ValueError(f'暂不支持ir类型:{type(ir)}')
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
    weight_info = {}
    for (name, layer) in ir.layers.items():
        if layer.type == 'op':
            mapping_info = layer.a111_mapping_info
            if mapping_info != None:
                op_id = layer.op.op_id
                if op_id in ['conv2d', 'conv_transpose2d', 'matmul', 'fc', 'linear']:
                    weight_info[name] = {}
                    input_shape = [layer.inputs[0].channel, layer.inputs[0].height, layer.inputs[0].width]
                    weight_data = []
                    hd_tile_id = []
                    hd_xb_id = []
                    addr_ = []
                    for (k, v) in mapping_info.mappings.items():
                        addr_.append(v.device)
                        tile_id = int(v.device.split('.')[2].split(':')[-1])
                        xb_id_ = int(v.device.split('.')[3].split(':')[-1])
                        hd_tile_id.append(tile_id % 6)
                        xb_id_ = 4 * (tile_id // 6) + xb_id_
                        hd_xb_id.append(xb_id_)
                    assert all((x == hd_tile_id[0] for x in hd_tile_id))
                    for addr in addr_:
                        assert addr in array_data.keys(), f'层{name}所在的芯片位置{addr} 不在 array_data {array_data.keys()}中!!!'
                        weight_data.append(array_data[addr])
                    num_column = layer.outputs[0].channel
                    weight_info[name]['input_shape'] = input_shape
                    weight_info[name]['weight_data'] = weight_data
                    weight_info[name]['tile'] = hd_tile_id[0]
                    weight_info[name]['xb'] = hd_xb_id
                    weight_info[name]['num_column'] = num_column
                    if op_id in ['conv2d', 'conv_transpose2d']:
                        kernel = layer.op.kernel
                        stride = layer.op.stride
                        padding = layer.op.padding
                        weight_info[name]['kernel'] = kernel
                        weight_info[name]['stride'] = stride
                        weight_info[name]['padding'] = padding
    return weight_info
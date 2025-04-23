

def Convert_simrt_to_chiprt(ir, specify_layer=None):
    
    for name, layer in ir.layers.items():
        if layer.type != 'op' :
            continue
        
        if layer.op.op_id in ['conv2d', 'fc', 'matmul', 'linear', 'conv_transpose2d']:
            if specify_layer != None and name in specify_layer:
                assert layer.c200_mapping_info != None
                layer.c200_mapping_info.runtime = 'c200'
            elif specify_layer == None:
                assert layer.c200_mapping_info != None
                layer.c200_mapping_info.runtime = 'c200'
       
    return ir

def Convert_chiprt_to_simrt(ir, specify_layer=None):
    
    for name, layer in ir.layers.items():
        if layer.type != 'op' :
            continue
        
        if layer.op.op_id in ['conv2d', 'fc', 'matmul', 'linear', 'conv_transpose2d']:
            if specify_layer != None and name in specify_layer:
                assert layer.c200_mapping_info != None
                layer.c200_mapping_info.runtime = 'simulation'
            elif specify_layer == None:
                assert layer.c200_mapping_info != None
                layer.c200_mapping_info.runtime = 'simulation'
                
    return ir
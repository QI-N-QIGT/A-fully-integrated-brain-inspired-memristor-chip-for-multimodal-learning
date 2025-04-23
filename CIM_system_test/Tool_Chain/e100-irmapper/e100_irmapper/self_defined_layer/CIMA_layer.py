from ..device.CIMA import *
from e100_irtool.core import make_ir, make_op, make_layer
from ..self_defined_op.fused_op import *

class CIMAComputeBaseLayer:
    
    def __init__(self, layer_name, core_id=[0, 0], in_channel=0, out_channel=0, credit_len = 0):
        
        self.layer_name = layer_name
        self.core_id = core_id
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.credit_len = credit_len
        self.ref_name = []
        self.input_image_size = []
        self.output_image_size = []
        
    def set_ref_name(self, ref_name):
        self.ref_name.append(ref_name)
    
    def set_input_image_size(self, image_size):
        self.input_image_size.append(image_size)
    
    def set_output_image_size(self, image_size):
        self.output_image_size.append(image_size)
    
    def set_in_buffer_addr(self, buffer_addr):
        self.in_buffer_addr = buffer_addr
    
    def set_credit_len(self, credit_len):
        self.credit_len = credit_len
        
class CIMAConvLayer(CIMAComputeBaseLayer):
    
    def __init__(self, layer_name, *, core_id= [0, 0], pe_cluster_id=0,  pe_xb_id=0, in_channel=0, 
                       out_channel = 0, relu=False, kernel_size = 1, stride = 1,
                       padding = 0, silu=False, credit_len = 0):
        '''
        pe_cluster_id: # N:0,  E:1, S:2, W:3
        pe_xb_id: [0, 15]
        '''
        super().__init__(layer_name, core_id, in_channel, out_channel, credit_len)
        self.pe_cluster_id = pe_cluster_id
        self.pe_xb_id = pe_xb_id
        self.kernel_size = kernel_size
        self.stride = stride
        self.relu = relu
        self.silu = silu
        self.padding = padding
        
    def set_weight_shape(self, weight_shape):
        self.weight_shape = weight_shape
    
class CIMAFCLayer(CIMAComputeBaseLayer):
    
    def __init__(self, layer_name, *, core_id= [0, 0], pe_cluster_id=0,  pe_xb_id=0, in_channel=0, 
                       out_channel = 0, relu=False,  silu=False, credit_len = 0):
        
        '''
        pe_cluster_id: # N:0,  E:1, S:2, W:3
        pe_xb_id: [0, 15]
        '''
        super().__init__(layer_name, core_id, in_channel, out_channel, credit_len)
        self.pe_cluster_id = pe_cluster_id
        self.pe_xb_id = pe_xb_id
        self.relu = relu
        self.silu = silu
    
    def set_weight_shape(self, weight_shape):
        self.weight_shape = weight_shape
    
class CIMAInputLayer:
    
    def __init__(self, layer_name, input_image_size):
        self.layer_name = layer_name
        self.input_image_size = input_image_size

class CIMAOutputLayer:
    
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.output_image_size = []
        self.ref_name = []
    
    def set_ref_name(self, ref_name):
        self.ref_name.append(ref_name) 
    
    def set_output_image_size(self, output_image_size):
        self.output_image_size.append(output_image_size)

class CIMAAddLayer(CIMAComputeBaseLayer):
    
    def __init__(self, layer_name, *, core_id= [0, 0], in_channel=0, out_channel = 0, credit_len=0, split = None):
        '''
        pe_cluster_id: # N:0,  E:1, S:2, W:3
        pe_xb_id: [0, 15]
        '''
        super().__init__(layer_name, core_id, in_channel, out_channel, credit_len)
        self.split = split
        
class CIMAConcatLayer(CIMAComputeBaseLayer):
    
    def __init__(self, layer_name, *, core_id= [0, 0], in_channel=0, out_channel=0, credit_len=0, split=None):
        '''
        pe_cluster_id: # N:0,  E:1, S:2, W:3
        pe_xb_id: [0, 15]
        '''
        super().__init__(layer_name, core_id, in_channel, out_channel, credit_len)
        self.split = split
    
class CIMASplitLayer(CIMAComputeBaseLayer):
    
    def __init__(self, layer_name, *, core_id= [0, 0], in_channel=0, out_channel = 0, credit_len=0):
        '''
        pe_cluster_id: # N:0,  E:1, S:2, W:3
        pe_xb_id: [0, 15]
        '''
        super().__init__(layer_name, core_id, in_channel, out_channel, credit_len)
    
class CIMAPoolingLayer(CIMAComputeBaseLayer):
    
    def __init__(self, layer_name, *, core_id= [0, 0], in_channel=0, out_channel = 0, 
                 pool_kernel=2, pool_mode='max', pool_stride=2, credit_len = 0):
        '''
        pe_cluster_id: # N:0,  E:1, S:2, W:3
        pe_xb_id: [0, 15]
        '''
        super().__init__(layer_name, core_id, in_channel, out_channel, credit_len)
        self.pool_kernel = pool_kernel
        self.pool_mode = pool_mode
        self.pool_stride = pool_stride
        
    
class CIMAIdentityLayer(CIMAComputeBaseLayer):
    
    def __init__(self, layer_name, *, core_id= [0, 0], in_channel=0, out_channel = 0, credit_len = 0):
        '''
        pe_cluster_id: # N:0,  E:1, S:2, W:3
        pe_xb_id: [0, 15]
        '''
        super().__init__(layer_name, core_id, in_channel, out_channel, credit_len)
    
class CIMAPipeGraph:
    
    def __init__(self, layer_info_list, layer_graph):
        self.layer_info_list = layer_info_list
        self.layer_graph = layer_graph
        # 各节点的存储保存位置
        self.dmem_size = {}
        self.layer_info_infer()
        
    def layer_info_infer(self):
        
        # 获取各层的名称以及对应的属性
        self.layer_info = {}
        for l in self.layer_info_list:
            if isinstance(l, CIMAInputLayer):
                self.layer_info['graph_input'] = l
            elif isinstance(l, CIMAOutputLayer):
                self.layer_info['graph_output'] = l
            else:
                self.layer_info[l.layer_name] = l
        
        # 更新各层的ref_name, 以及参数
        assert isinstance(self.layer_graph, dict), f'layer graph参数需要是 字典(dict) 类型!!!'
        for pre_layer_name, current_layer_name in self.layer_graph.items():
            assert isinstance(current_layer_name, list), f'dict的value需要是 列别(list)类型!!!'
            # 获取输入的图像大小
            pre_layer_info = self.layer_info[pre_layer_name]
            in_image_size = pre_layer_info.input_image_size
            if not isinstance(pre_layer_info, CIMAInputLayer):
                out_image_size = pre_layer_info.output_image_size
            # 
            index = 0
            for nl in current_layer_name:
                current_layer_info = self.layer_info[nl]
                # 更新current_layer_name中的ref name
                current_layer_info.set_ref_name(pre_layer_name)
                # 更新current_layer_name的输入图像大小
                current_image_size = []
                if isinstance(pre_layer_info, CIMAInputLayer):
                    current_image_size = in_image_size
                else:
                    if len(out_image_size) > 1:
                        current_image_size = out_image_size[index]
                    else:
                        current_image_size = out_image_size[0]
                if isinstance(current_layer_info, CIMAOutputLayer):
                    current_layer_info.set_output_image_size(current_image_size)
                    continue
                current_layer_info.set_input_image_size(current_image_size)
                # 获取当前层的channel
                in_channel = current_layer_info.in_channel
                out_channel = current_layer_info.out_channel
                # 初始化dmem_size
                core_id = current_layer_info.core_id
                if f'{core_id}' not in self.dmem_size.keys():
                    self.dmem_size[f'{core_id}'] = [0, 0] # start addr, end addr
                # 根据卷积与全连接推算当前层的图像大小
                if isinstance(current_layer_info, CIMAConvLayer):
                    # 更新当前层的输出图像大小
                    kernel = current_layer_info.kernel_size
                    stride = current_layer_info.stride
                    padding = current_layer_info.padding
                    out_h = (current_image_size[0] - kernel + 2*padding) // stride + 1
                    out_w = (current_image_size[1] - kernel + 2*padding) // stride + 1
                    current_layer_info.set_output_image_size([out_h, out_w])
                    # 更新权重维度
                    current_layer_info.set_weight_shape([out_channel, in_channel, kernel, kernel])
                    # 分配dmem地址
                    mem_length =  in_channel * current_image_size[1] * kernel
                    current_layer_info.set_in_buffer_addr([self.dmem_size[f'{core_id}'][1], mem_length])
                    self.dmem_size[f'{core_id}'][1] += mem_length
                    # 设置 credit value
                    if current_layer_info.credit_len == 0:
                        credit_len = current_image_size[1]
                        current_layer_info.set_credit_len(credit_len)
                elif isinstance(current_layer_info, CIMAFCLayer):
                    # 更新权重
                    current_layer_info.set_weight_shape([current_layer_info.out_channel, current_layer_info.in_channel])
                    # 更新权重
                    current_layer_info.set_weight_shape([current_layer_info.out_channel, current_layer_info.in_channel])
                    # 更新输入、输出图像大小都为 [1，1]
                    current_layer_info.set_input_image_size([1, 1])
                    current_layer_info.set_output_image_size([1, 1])
                    # 分配dmem地址
                    mem_length =  in_channel
                    current_layer_info.set_in_buffer_addr([self.dmem_size[f'{core_id}'][1], mem_length])
                    self.dmem_size[f'{core_id}'][1] += mem_length
                    # 设置 credit value
                    if current_layer_info.credit_len == 0:
                        credit_len = in_channel
                        current_layer_info.set_credit_len(credit_len)
                elif isinstance(current_layer_info, CIMAConcatLayer) or isinstance(current_layer_info, CIMAAddLayer) or \
                     isinstance(current_layer_info, CIMASplitLayer) or isinstance(current_layer_info, CIMAIdentityLayer):
                    
                    # 更新输入、输出图像大小
                    current_layer_info.set_input_image_size(current_image_size)
                    current_layer_info.set_output_image_size(current_image_size)
                    # 分配dmem地址
                    if isinstance(in_channel, list):
                        mem_length =  in_channel[0] * current_image_size[0]
                    else:
                        mem_length =  in_channel * current_image_size[0]
                    current_layer_info.set_in_buffer_addr([self.dmem_size[f'{core_id}'][1], mem_length])
                    self.dmem_size[f'{core_id}'][1] += mem_length
                    # 设置 credit value
                    if current_layer_info.credit_len == 0:
                        credit_len = current_image_size[0]
                        current_layer_info.set_credit_len(credit_len)
                    
                else:
                    raise ValueError(f'暂不支持 {type(current_layer_info)} 类型 !!!')
          
    def to_ir(self):
        
        self.ir = make_ir()        
        
        # 输入层
        inputs = []
        for nl in self.layer_graph['graph_input']:
            in_layer = self.layer_info[nl]
            inputs_dict = dict(channel=in_layer.in_channel, height=in_layer.input_image_size[0][0], width=in_layer.input_image_size[0][1])
            inputs.append(inputs_dict)
        self.ir.add_layer('graph_input',type='input',inputs=inputs)
        
        graph_input_count = 0
        
        # 中间层
        for l in self.layer_info_list:
            if isinstance(l, CIMAInputLayer) or isinstance(l, CIMAOutputLayer):
                continue
            elif isinstance(l, CIMAConvLayer):
                op_layer = l
                op_id = 'conv2d'
                if op_layer.relu or op_layer.avgpool:
                    op_id = 'fused_conv2d'
                    relu = None
                    silu = None
                    if op_layer.relu:
                        relu = dict(op_id='relu')
                    elif op_layer.silu:
                        silu = dict(op_id='silu')
                    op_ = make_op(op_id, in_channel=op_layer.in_channel,
                                out_channel=op_layer.out_channel,
                                kernel=op_layer.kernel_size,
                                stride=op_layer.stride, 
                                padding=op_layer.padding, relu=relu, silu= silu)
                else:
                    op_ = make_op(op_id, in_channel=op_layer.in_channel,
                                out_channel=op_layer.out_channel,
                                kernel=op_layer.kernel_size,
                                stride=op_layer.stride, 
                                padding=op_layer.padding)
                in_info = []
                index = 0
                for nl in op_layer.ref_name:
                    # ref_layer = self.layer_info[nl]
                    if 'graph_input' in nl:
                        nl = f'graph_input:{graph_input_count}'
                        graph_input_count += 1
                    in_info.append(dict(ref=nl, channel=op_layer.in_channel, height=op_layer.input_image_size[index][0],
                                        width=op_layer.input_image_size[index][1]))
                    index += 1
                weight_info = dict(weight=dict(shape=op_layer.weight_shape))
                out_info = [dict(channel=op_layer.out_channel, height=op_layer.output_image_size[0][0], 
                                width=op_layer.output_image_size[0][1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info, weights=weight_info)
                
                # 设置当前的 mapping info
                
                mapping_info = []
                c = 0
                addr_h = op_layer.in_channel * op_layer.kernel_size**(2)
                addr_w = op_layer.out_channel
                # for xb in op_layer.xb_id_list:
                core_index = op_layer.core_id[0] * 6 + op_layer.core_id[1]
                pe_cluster_id = op_layer.pe_cluster_id
                xb_id = op_layer.pe_xb_id
                device_ref = f'cima-0.cima-node:{core_index}.cima-pe-cluster:{pe_cluster_id}.cima-xb:{xb_id}'
                addr_value = [0, 0, addr_h, addr_w]
                mapping_info.append(CIMADeviceMappingInfo(index = [0, c, 0], device=device_ref, address = addr_value))
                # c += 1
                CIMA_mapping_info = CIMAMappingInfo(in_line_buffer_addr=[[hex(op_layer.in_buffer_addr[0]), hex(op_layer.in_buffer_addr[1])]], 
                                                    credit_len=[op_layer.credit_len],
                                                    mappings=mapping_info)
                self.ir.add_layer(op_layer.layer_name, current_layer, CIMA_mapping_info=CIMA_mapping_info)
                
            elif isinstance(l, CIMAFCLayer):
                
                op_layer = l
                op_id = 'matmul'
                if op_layer.relu:
                    op_id = 'fused_fc'
                    relu = dict(op_id='relu')
                    op_ = make_op(op_id, in_channel=op_layer.in_channel, out_channel=op_layer.out_channel, relu=relu)
                else:
                   op_ = make_op(op_id, in_channel=op_layer.in_channel, out_channel=op_layer.out_channel)
                in_info = []
                index = 0
                for nl in op_layer.ref_name:
                    # ref_layer = self.layer_info[nl]
                    in_info.append(dict(ref=nl, channel=op_layer.in_channel, height=op_layer.input_image_size[index][0],
                                        width=op_layer.input_image_size[index][1]))
                    index += 1
                weight_info = dict(weight=dict(shape=op_layer.weight_shape))
                out_info = [dict(channel=op_layer.out_channel, height=op_layer.output_image_size[0][0], 
                                width=op_layer.output_image_size[0][1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info, weights=weight_info)
                # 设置当前的 mapping info
                mapping_info = []
                c = 0
                core_index = op_layer.core_id[0] * 6 + op_layer.core_id[1]
                pe_cluster_id = op_layer.pe_cluster_id
                xb_id = op_layer.pe_xb_id
                device_ref = f'cima-0.cima-node:{core_index}.cima-pe-cluster:{pe_cluster_id}.cima-xb:{xb_id}'
                addr_h = op_layer.in_channel
                addr_w = op_layer.out_channel
                addr_value = [0, 0, addr_h, addr_w]
                mapping_info.append(CIMADeviceMappingInfo(index = [0, c, 0], device=device_ref, address = addr_value))
                # c += 1
                CIMA_mapping_info = CIMAMappingInfo(in_line_buffer_addr=[[hex(op_layer.in_buffer_addr[0]), hex(op_layer.in_buffer_addr[1])]], 
                                                    credit_len=[op_layer.credit_len],
                                                    mappings = mapping_info)
                self.ir.add_layer(op_layer.layer_name, current_layer, CIMA_mapping_info=CIMA_mapping_info,)
            elif isinstance(l, CIMAConcatLayer):
                
                op_layer = l
                op_id = 'concat'
                if op_layer.split != None:
                    op_id = 'fused_concat'
                    fused_split = {'op_id':'split', 'axis':1, 'split':op_layer.split}
                    op_ = make_op(op_id, axis=1, split=fused_split)
                else:
                    op_ = make_op(op_id, axis=1)
                in_info = []
                index = 0
                assert isinstance(op_layer.in_channel, list) 
                for nl in op_layer.ref_name:
                    ic = op_layer.in_channel[index]
                    in_info.append(dict(ref=nl, channel=ic, height=op_layer.input_image_size[index][0],
                                        width=op_layer.input_image_size[index][1]))
                    index += 1
                if op_id == 'fused_concat':
                    out_info = []
                    for oc in op_layer.out_channel:
                        out_info.append(dict( channel=oc, height=op_layer.input_image_size[index][0],
                                            width=op_layer.input_image_size[index][1]))
                        index += 1
                else:
                    out_info = [dict(channel=op_layer.out_channel, height=op_layer.output_image_size[0][0], 
                                    width=op_layer.output_image_size[0][1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info)
                # 设置当前的 mapping info
                mapping_info = []
                core_index = op_layer.core_id[0] * 6 + op_layer.core_id[1]
                device_ref = f'cima-0.cima-node:{core_index}'
                addr_value = 0
                mapping_info.append(CIMADeviceMappingInfo(index = [0, 0, 0], device=device_ref, address = addr_value))
                CIMA_mapping_info = CIMAMappingInfo(in_line_buffer_addr=[[hex(op_layer.in_buffer_addr[0]), hex(op_layer.in_buffer_addr[1])]], 
                                                    credit_len=[op_layer.credit_len],
                                                    mappings = mapping_info)
                self.ir.add_layer(op_layer.layer_name, current_layer, CIMA_mapping_info=CIMA_mapping_info,)
                
            elif isinstance(l, CIMAIdentityLayer):
                
                op_layer = l
                op_id = 'identity'
                op_ = make_op(op_id)
                in_info = []
                assert len(op_layer.ref_name) == 1, f'identity op 的ref name应该只有一个!!! {op_layer.ref_name}'
                ref_name = op_layer.ref_name[0]
                if 'graph_input' in ref_name:
                    ref_name = 'graph_input:0'
                in_info.append(dict(ref=ref_name, channel=op_layer.in_channel, height=op_layer.input_image_size[0][0],
                                    width=op_layer.input_image_size[0][1]))
                out_info = [dict(channel=op_layer.out_channel, height=op_layer.output_image_size[0][0], 
                                width=op_layer.output_image_size[0][1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info)
                # 设置当前的 mapping info
                mapping_info = []
                core_index = op_layer.core_id[0] * 6 + op_layer.core_id[1]
                device_ref = f'cima-0.cima-node:{core_index}'
                addr_value = 0
                mapping_info.append(CIMADeviceMappingInfo(index = [0, 0, 0], device=device_ref, address = addr_value))
                CIMA_mapping_info = CIMAMappingInfo(in_line_buffer_addr=[[hex(op_layer.in_buffer_addr[0]), hex(op_layer.in_buffer_addr[1])]], 
                                                    credit_len=[op_layer.credit_len],
                                                    mappings = mapping_info)
                self.ir.add_layer(op_layer.layer_name, current_layer, CIMA_mapping_info=CIMA_mapping_info,)
            
            elif isinstance(l, CIMASplitLayer):
                
                op_layer = l
                op_id = 'split'
                assert isinstance(op_layer.out_channel, list)
                op_ = make_op(op_id, axis=1, split=op_layer.out_channel)
                in_info = []
                assert len(op_layer.ref_name) == 1, f'identity op 的ref name应该只有一个!!! {op_layer.ref_name}'
                in_info.append(dict(ref=nl, channel=op_layer.in_channel, height=op_layer.input_image_size[0][0],
                                    width=op_layer.input_image_size[0][1]))
                for oc in op_layer.out_channel:
                    out_info = [dict(channel=oc, height=op_layer.output_image_size[0][0], 
                                    width=op_layer.output_image_size[0][1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info)
                # 设置当前的 mapping info
                mapping_info = []
                core_index = op_layer.core_id[0] * 6 + op_layer.core_id[1]
                device_ref = f'cima-0.cima-node:{core_index}'
                addr_value = 0
                mapping_info.append(CIMADeviceMappingInfo(index = [0, 0, 0], device=device_ref, address = addr_value))
                CIMA_mapping_info = CIMAMappingInfo(in_line_buffer_addr=[[hex(op_layer.in_buffer_addr[0]), hex(op_layer.in_buffer_addr[1])]], 
                                                    credit_len=[op_layer.credit_len],
                                                    mappings = mapping_info)
                self.ir.add_layer(op_layer.layer_name, current_layer, CIMA_mapping_info=CIMA_mapping_info,)
            
            elif isinstance(l, CIMAAddLayer):
                
                op_layer = l
                op_id = 'add'
                if op_layer.split != None:
                    op_id = 'fused_add'
                    fused_split = {'op_id':'split', 'axis':1, 'split':op_layer.split}
                    op_ = make_op(op_id, axis=1, split=fused_split)
                else:
                    op_ = make_op(op_id, axis=1)
                in_info = []
                index = 0
                assert isinstance(op_layer.in_channel, list) 
                for nl in op_layer.ref_name:
                    ic = op_layer.in_channel[index]
                    in_info.append(dict(ref=nl, channel=ic, height=op_layer.input_image_size[index][0],
                                        width=op_layer.input_image_size[index][1]))
                    index += 1
                if op_id == 'fused_add':
                    out_info = []
                    for oc in op_layer.out_channel:
                        in_info.append(dict( channel=oc, height=op_layer.input_image_size[index][0],
                                            width=op_layer.input_image_size[index][1]))
                        index += 1
                else:
                    out_info = [dict(channel=op_layer.out_channel, height=op_layer.output_image_size[0][0], 
                                    width=op_layer.output_image_size[0][1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info)
                # 设置当前的 mapping info
                mapping_info = []
                core_index = op_layer.core_id[0] * 6 + op_layer.core_id[1]
                device_ref = f'cima-0.cima-node:{core_index}'
                addr_value = 0
                mapping_info.append(CIMADeviceMappingInfo(index = [0, 0, 0], device=device_ref, address = addr_value))
                CIMA_mapping_info = CIMAMappingInfo(in_line_buffer_addr=[[hex(op_layer.in_buffer_addr[0]), hex(op_layer.in_buffer_addr[1])]], 
                                                    credit_len=[op_layer.credit_len],
                                                    mappings = mapping_info)
                self.ir.add_layer(op_layer.layer_name, current_layer, CIMA_mapping_info=CIMA_mapping_info,)
            
            else:
                raise ValueError(f'暂不支持 层类型 {type(l)}!!!')
                 
        # 输出层
        out_layer = self.layer_info['graph_output']
        inputs = []
        for l in out_layer.ref_name:
            ref_layer_info = self.layer_info[l] 
            outputs_dict = dict(ref=ref_layer_info.layer_name, channel=ref_layer_info.out_channel, height=ref_layer_info.output_image_size[0][1],
                                width=ref_layer_info.output_image_size[0][1])
            inputs.append(outputs_dict)
        self.ir.add_layer('graph_output',type='output',inputs=inputs)
        
        # 添加device
        self.ir.add_device(
            name = 'cima-0',
            kind = 'cima-node',
            num = 36,
            height = 6,
            width = 6,
            task_num = 128
        )

        return self.ir

    def to_systemc_config(self, config_path, log_level = 0, trace_enable = True, 
                          perf_single_trace = False, run_time=1000000, batch=1):
        '''
        runtime: unit: ns
        '''
        # load ir
        ir = self.to_ir()
        # gen simulation code
        from e100_irbackend.systemC.CIMA.v2.gen_code import CodeGen
        code = CodeGen(ir)
        code.run(output_file=config_path, log_level = log_level, trace_enable = trace_enable, perf_single_trace = perf_single_trace, run_time=run_time, batch=batch)


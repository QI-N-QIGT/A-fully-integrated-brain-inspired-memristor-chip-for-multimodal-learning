from ..helper import *
from ..placement import *
from ..esti_model import *
from ..device.c200 import C200DeviceMappingInfo
from ..device.a111 import A111DeviceMappingInfo
import warnings
from scipy.spatial.distance import cdist
from e100_irtool.core import make_op
from ..parser import IrParser

class Base(object):

    def __init__(self, node_info, node_weight, hardware_config, weight_format='CHW', average_copy=None, specify_para_num=None, specify_split_num=None, place_strategy=OneOnOne, window_copy=False, ir=None, adaptive_split_ir=False, dmac_layer=None):
        
        self.node_info = node_info
        self.node_weight = node_weight
        self.weight_format = weight_format
        self.average_copy = average_copy
        self.specify_para_num = specify_para_num
        self.specify_split_num = specify_split_num
        self.hardware_config = hardware_config
        self.place_strategy = place_strategy
        self.window_copy = window_copy
        self.ir = ir
        self.adaptive_split_ir = adaptive_split_ir
        self.dmac_layer = dmac_layer

    def get_hardware_info(self):
        
        self.XB_num = self.hardware_config['xb_number']
        self.XB_size = self.hardware_config['xb_shape']
        self.hd_name = self.hardware_config['name']
        self.dac_num = self.hardware_config['dac_num']
        self.adc_num = self.hardware_config['adc_num']
        self.dac_precision = self.hardware_config['dac_precision']
        if 'a111-tile' in self.hd_name[0] and self.window_copy != False:
            raise ValueError(f'a111-tile 不支持 window copy的复制方法!!!')
        self.device_field = self.hd_name[0]

    def split_average(self, CIMA_datawidth=8):
        
        if self.average_copy != None:
            for i in self.average_copy.keys():
                if i in self.node_weight.keys():
                    (w, h) = self.node_weight[i]
                    w_ = w * self.average_copy[i][1]
                    h_ = h * self.average_copy[i][0]
                    self.node_weight[i] = [w_, h_]
                else:
                    warnings.warn(f'需要mapping到device的层不包括: {i} !!!')
        if self.weight_format == 'HWC':
            XB_size = self.XB_size
            DMAC_size = None
            if 'cima' in self.hd_name[0]:
                if CIMA_datawidth == 8:
                    XB_size = [self.XB_size[0] * 2, self.XB_size[1]]
                elif CIMA_datawidth == 4:
                    XB_size = [self.XB_size[0] * 4, self.XB_size[1]]
                DMAC_size = self.hardware_config['dmac_shape']
            (self.split_node_weight, self.split_num) = split_node_HWC(self.node_weight, self.node_info, self.specify_para_num, XB_size, DMAC_size, self.dmac_layer, device=self.device_field)
        elif self.weight_format == 'CHW':
            self.split_num = {}
            for i in self.node_weight.keys():
                if self.specify_para_num != None and i in self.specify_para_num.keys():
                    (p_diff_array, p_same_array) = self.specify_para_num[i]
                else:
                    (p_diff_array, p_same_array) = (1, 1)
                if self.window_copy and self.node_info[i]['op_type'] in ['conv2d', 'conv_transpose2d']:
                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0]
                        _w = self.specify_split_num[i][1]
                        self.split_num[i] = [p_diff_array, p_same_array, _w, _h]
                    else:
                        self.split_num[i] = [p_diff_array, p_same_array, 1, 1]
                else:
                    self.node_weight[i][1] = self.node_weight[i][1] * p_same_array
                    self.node_weight[i][0] = self.node_weight[i][0] * p_same_array
                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0]
                        _w = self.specify_split_num[i][1]
                    else:
                        _h = math.ceil(self.node_weight[i][1] / self.XB_size[1])
                        _w = math.ceil(self.node_weight[i][0] / self.XB_size[0])
                    self.split_num[i] = [p_diff_array, p_same_array, _w, _h]
            if self.window_copy:
                (self.split_node_weight, self.split_num) = split_node_window_duplicate(self.node_info, self.XB_size, self.split_num)
            else:
                self.split_node_weight = split_node(self.node_weight, self.split_num)
        else:
            raise ValueError(f'暂不支持权重格式{self.weight_format}')

    def run(self, CIMA_alpha=0, CIMA_method='random_search', CIMA_datawidth=8):
        
        self.get_hardware_info()
        self.split_average(CIMA_datawidth=CIMA_datawidth)
        if self.adaptive_split_ir:
            layers_info = self.ir.layers
            next_layer_dict = get_next_layer(self.ir.layers)
            split_layer_name = []
            new_split_num = {}
            for (k, v) in self.split_num.items():
                if v[2] * v[3] != 1:
                    current_layer = layers_info[k]
                    split_layer_name.append(k)
                    if v[3] > 1:
                        insert_split_node_name = f'{k}_Split'
                        assert current_layer.op.in_channel % v[3] == 0
                        axis = 1
                        split = []
                        split_output = []
                        for i in range(v[3]):
                            split.append(current_layer.op.in_channel // v[3])
                            split_output.append({'channel': current_layer.op.in_channel // v[3], 'width': current_layer.inputs[0].width, 'height': current_layer.inputs[0].height})
                        op_ = make_op('split', axis=axis, split=split)
                        split_input = current_layer.inputs
                        self.ir.add_layer(insert_split_node_name, op=op_, inputs=split_input, outputs=split_output)
                    split_in_channel = current_layer.op.in_channel // v[3]
                    if current_layer.op.out_channel % v[2] != 0:
                        warnings.warn(f'当前层 {k} 输出通道为 {current_layer.op.out_channel}, 拆分次数为 {v[2]}, 拆分为: {math.ceil(current_layer.op.out_channel // v[2])} !!!')
                        current_layer.op.out_channel += 1
                    split_out_channel = math.ceil(current_layer.op.out_channel // v[2])
                    in_width = current_layer.inputs[0].width
                    in_height = current_layer.inputs[0].height
                    out_width = current_layer.outputs[0].width
                    out_height = current_layer.outputs[0].height
                    for h_ in range(v[3]):
                        for w_ in range(v[2]):
                            new_insert_layer = current_layer.clone()
                            if v[3] > 1:
                                new_insert_layer.inputs[0].ref = insert_split_node_name + f':{h_}'
                            new_node_name = k + f'_{h_}_{w_}'
                            new_insert_layer.inputs[0].channel = split_in_channel
                            new_insert_layer.outputs[0].channel = split_out_channel
                            original_weight_shape = current_layer.weights['weight'].shape
                            if len(original_weight_shape) == 4:
                                new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel, original_weight_shape[2], original_weight_shape[3])
                            elif len(original_weight_shape) == 2:
                                new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel)
                            else:
                                raise ValueError(f'暂不支持 权重维度 : {original_weight_shape} !!!')
                            new_insert_layer.op.in_channel = split_in_channel
                            new_insert_layer.op.out_channel = split_out_channel
                            if 'bias' in new_insert_layer.weights.keys():
                                new_insert_layer.weights['bias'].shape = split_out_channel
                            self.ir.layers[new_node_name] = new_insert_layer
                            new_split_num[new_node_name] = [self.split_num[k][0], self.split_num[k][1], 1, 1]
                        if v[2] > 1:
                            insert_concat_node_name = f'{k}_Concat_{h_}'
                            op_ = make_op('concat', axis=1)
                            concat_input = []
                            for w_ in range(v[2]):
                                concat_input.append(dict(ref=k + f'_{h_}_{w_}', channel=split_out_channel, width=out_width, height=out_height))
                            concat_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                            self.ir.add_layer(insert_concat_node_name, op=op_, inputs=concat_input, outputs=concat_output)
                    if v[3] > 1:
                        insert_add_node_name = f'{k}_Add'
                        op_ = make_op('add')
                        add_input = []
                        for h_ in range(v[3]):
                            ref_name = k + f'_{h_}_0'
                            if v[2] > 1:
                                ref_name = f'{k}_Concat_{h_}'
                            add_input.append(dict(ref=ref_name, channel=current_layer.op.out_channel, width=out_width, height=out_height))
                        add_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                        self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                    for nl in next_layer_dict[k]:
                        c = 0
                        if v[3] > 1:
                            for i in self.ir.layers[nl].inputs:
                                if i.ref == k:
                                    self.ir.layers[nl].inputs[c].ref = insert_add_node_name
                                c += 1
                        elif v[2] > 1:
                            for i in self.ir.layers[nl].inputs:
                                if i.ref == k:
                                    self.ir.layers[nl].inputs[c].ref = insert_concat_node_name
                                c += 1
                    self.ir.layers.pop(k)
                else:
                    new_split_num[k] = v
            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
            self.ir.dump_json(file=f'Hardware_adaptive_ir.yaml')
            if 'cima' in self.hd_name[0]:
                self.ir = fuse_op(self.ir, split_fuse=True)
            new_split_node_weight = {}
            for (k, v) in self.split_node_weight.items():
                k_ = k.split('.')
                if k_[0] in split_layer_name:
                    new_split_node_weight[f'{k_[0]}_{k_[2]}_{k_[3]}.0.0.0'] = v
                else:
                    new_split_node_weight[k] = v
            self.split_node_weight = new_split_node_weight
            ir_parser = IrParser(ir=self.ir)
            self.node_info = ir_parser.node_info
            self.split_num = new_split_num
        self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()
        if 'rram-144k' in self.hd_name[0]:
            sum_ = len(self.placed_nodes)
        else:
            assert isinstance(self.placed_nodes, dict)
            sum_ = 0
            for i in self.placed_nodes:
                v = self.placed_nodes[i]
                sum_ += len(v)
        rest_xb = self.XB_num - sum_
        print()
        if rest_xb < 0:
            raise ValueError(f'按照当前策略 {self.place_strategy.__name__} 无法放下！至少需要 {sum_} 个XB !!! 当前拥有 {self.XB_num} 个XB !!!')
        self.ref_to_device(CIMA_alpha=CIMA_alpha, CIMA_method=CIMA_method, CIMA_datawidth=CIMA_datawidth)

    def ref_to_device(self, CIMA_alpha=0, CIMA_method='random', CIMA_datawidth=8):
        
        self.node_mapping_info = {}
        assert len(self.placed_nodes) <= len(self.hd_name)
        if 'rram-144k' in self.hd_name[0]:
            for index in range(len(self.placed_nodes)):
                device_ref = self.hd_name[index]
                for node_addr in self.placed_nodes[index]:
                    key = list(node_addr.keys())[0]
                    value = list(node_addr.values())[0]
                    name_ = key.split('.')
                    node_name = name_[0]
                    if self.window_copy:
                        index_ = [int(name_[1]), int(name_[2]), int(name_[3].split('_')[0])]
                    else:
                        index_ = [int(name_[1]), int(name_[2]), int(name_[3])]
                    if node_name not in self.node_mapping_info.keys():
                        self.node_mapping_info[node_name] = []
                    mapping_info = C200DeviceMappingInfo(index=index_, device=device_ref, address=value)
                    self.node_mapping_info[node_name].append(mapping_info)
        elif 'a111-tile' in self.hd_name[0]:
            self.input_buffer_addr = {}
            self.output_buffer_addr = {}
            self.in_buf_type = {}
            self.out_buf_type = {}
            self.tile_all = []
            tile_op = []
            mapped_xb_id_count = 0
            mapped_tile_id_count = 0
            placed_nodes = copy.deepcopy(self.placed_nodes)
            count = 0
            self.layer_occupied_xb = {}
            while True:
                if placed_nodes == {}:
                    break
                node_name = list(placed_nodes.keys())[0]
                if len(placed_nodes[node_name]) > 4:
                    raise ValueError(f'{node_name} 超过4个xb')
                if len(tile_op) < 2 and mapped_xb_id_count + len(placed_nodes[node_name]) <= 4:
                    tile_op.append(node_name)
                    count += 1
                else:
                    self.tile_all.append(tile_op)
                    tile_op = []
                    mapped_tile_id_count += 1
                    mapped_xb_id_count = 0
                    count = 0
                    continue
                self.layer_occupied_xb[node_name] = (len(placed_nodes[node_name]) // 2 + 1) * 2
                count = 0
                for node_addr in placed_nodes[node_name]:
                    if mapped_xb_id_count % 2 == 1 and count == 0:
                        mapped_xb_id_count += 1
                    index = 4 * mapped_tile_id_count + mapped_xb_id_count
                    if index > 48:
                        print()
                        print()
                        raise ValueError(f'需要的xb数量 超过 总和(48个XB)！！！')
                    device_ref = self.hd_name[index]
                    key = list(node_addr.keys())[0]
                    value = list(node_addr.values())[0]
                    name_ = key.split('.')
                    index_ = [int(name_[1]), int(name_[2]), int(name_[3])]
                    if node_name not in self.node_mapping_info.keys():
                        self.node_mapping_info[node_name] = []
                    mapping_info = A111DeviceMappingInfo(index=index_, device=device_ref, address=value)
                    self.node_mapping_info[node_name].append(mapping_info)
                    mapped_xb_id_count += 1
                    count += 1
                if len(list(placed_nodes.keys())) == 1:
                    self.tile_all.append(tile_op)
                placed_nodes.pop(node_name)
            esram_start_addr = 0
            self.tile_occupied_xb = {}
            buf_size_type = [2048, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
            for tile in self.tile_all:
                first_node_name = tile[0]
                device_name = self.node_mapping_info[first_node_name][0].device
                tile_name = '.'.join(device_name.split('.')[0:3])
                self.tile_occupied_xb[tile_name] = []
                tile_start_addr = 0
                for node_name in tile:
                    self.tile_occupied_xb[tile_name].append(self.layer_occupied_xb[node_name])
                    node_info = self.node_info[node_name]
                    in_len_ = node_info['in_data_len']
                    out_len_ = node_info['out_data_len']
                    tile_in_len = (in_len_ // 2048 + 1) * 2048
                    if tile_in_len > 32 * 1024:
                        warnings.warn(f'{tile_in_len} 超过tile buffer 内存上限 32KB, 该层 {node_name} 需要拆分进行多次计算')
                        self.in_buf_type[node_name] = 6
                    if tile_in_len // 2048 <= 6:
                        self.in_buf_type[node_name] = tile_in_len // 2048 - 1
                    elif tile_in_len // 2048 <= 8:
                        self.in_buf_type[node_name] = 6
                    else:
                        self.in_buf_type[node_name] = 6
                    index = tile.index(node_name)
                    if index == 0:
                        esram_len = (in_len_ // 256 + 1) * 256
                        esram_start_addr += esram_len
                        self.input_buffer_addr[node_name] = {'base': hex(2013265920), 'start': hex(tile_start_addr), 'end': hex(tile_start_addr + buf_size_type[self.in_buf_type[node_name]])}
                    else:
                        self.input_buffer_addr[node_name] = {'base': hex(2013265920), 'start': hex(tile_start_addr), 'end': hex(tile_start_addr + buf_size_type[self.in_buf_type[node_name]])}
                    tile_start_addr += buf_size_type[self.in_buf_type[node_name]]
                    tile_out_len = (out_len_ // 2048 + 1) * 2048
                    if tile_out_len > 32 * 1024:
                        warnings.warn(f'{tile_out_len} 超过tile buffer 内存上限 32KB, 该层{node_name} 需要拆分进行多次计算')
                        self.in_buf_type[node_name] = 6
                    if tile_out_len // 2048 <= 6:
                        self.out_buf_type[node_name] = tile_in_len // 2048 - 1
                    elif tile_out_len // 2048 <= 8:
                        self.out_buf_type[node_name] = 6
                    else:
                        self.out_buf_type[node_name] = 6
                    if index != len(tile) - 1:
                        self.output_buffer_addr[node_name] = {'base': hex(2013265920), 'start': hex(tile_start_addr), 'end': hex(tile_start_addr + buf_size_type[self.out_buf_type[node_name]])}
                    else:
                        esram_len = (out_len_ // 256 + 1) * 256
                        self.output_buffer_addr[node_name] = {'base': hex(1744830464), 'start': hex(esram_start_addr), 'end': hex(esram_start_addr + esram_len)}
                        esram_start_addr += esram_len
                    if tile_start_addr >= 32 * 1024:
                        warnings.warn(f'{self.input_buffer_addr[node_name]} 超过tile buffer 内存上限 32KB, 该层{node_name}需要拆分进行多次计算')
        elif 'cima' in self.hd_name[0]:
            layer_ref = {}
            for (k, v) in self.node_info.items():
                layer_ref[k] = v['ref']
            available_nodes_xb = copy.deepcopy(self.hd_name)
            device_name = list(self.ir.devices.keys())[0]
            mesh_height = self.ir.devices[device_name].height
            mesh_width = self.ir.devices[device_name].width
            alpha = CIMA_alpha
            if CIMA_method.lower() == 'workload_balance':
                (self.node_mapping_info_list, self.record_io_workload) = Workload_balance_search(layer_ref, self.placed_nodes, available_nodes_xb, self.node_info, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=True)
            elif CIMA_method.lower() == 'a_search':
                (self.node_mapping_info_list, self.record_io_workload) = A_search(layer_ref, self.placed_nodes, available_nodes_xb, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=True)
            elif CIMA_method.lower() == 'random_search':
                (self.node_mapping_info_list, self.record_io_workload) = packaged_random_search(layer_ref, self.placed_nodes, available_nodes_xb, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=True, dmac_layer=self.dmac_layer)
            elif CIMA_method.lower() == 'onebyone_search':
                (self.node_mapping_info_list, self.record_io_workload) = onebyone_search(layer_ref, self.placed_nodes, available_nodes_xb, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=True)
            else:
                raise ValueError(f'暂不支持 {CIMA_method} !!!')
            self.in_line_buffer_addr = {}
            self.credit_len = {}
            linebuf_assigned = {}
            assert self.ir != None
            next_layer_dict = get_next_layer(self.ir.layers)
            pre_layer_dict = get_pre_layer(self.ir.layers)
            layers = self.ir.layers
            layers_name = list(layers.keys())
            layers_name.reverse()
            mapping_mesh_node = {}
            hosti_core = [(3, 0)]
            ddr_core = [(3, 5)]
            none_core = [(1, 5), (2, 5), (4, 5)]
            cant_mapped_core = hosti_core + ddr_core + none_core
            all_points = []
            for i in range(mesh_height):
                for j in range(mesh_width):
                    if (i, j) not in cant_mapped_core:
                        all_points.append((i, j))
            self.Max_Memory_Size = 1048576 // 2
            for name in layers_name:
                if layers[name].type != 'op':
                    continue
                if layers[name].op.op_id in ['flatten', 'reshape']:
                    continue
                if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', 'silu', 'resize', 'split', 'add', 'fused_add', 'fused_concat', 'concat'] and name not in self.node_mapping_info_list.keys():
                    in_channel = layers[name].inputs[0].channel
                    if in_channel == 255:
                        warnings.warn(f'当前层 {name} 插入fake通道, 通道数从{in_channel} 变为 256!!!')
                        in_channel += 1
                    height = layers[name].inputs[0].height
                    width = layers[name].inputs[0].width
                    if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d']:
                        kernel_size = layers[name].op.kernel
                        len_ = in_channel * width * kernel_size
                    elif layers[name].op.op_id in ['concat', 'fused_concat']:
                        len_ = in_channel * width * 4
                        if name in ['Concat_208'] and CIMA_datawidth == 8:
                            len_ *= 4
                    else:
                        len_ = in_channel * width
                        if layers[name].op.op_id in ['add', 'fused_add'] and CIMA_datawidth == 8:
                            len_ *= 4
                    nl = []
                    if name in next_layer_dict.keys():
                        nl = next_layer_dict[name]
                    pl = []
                    if name in pre_layer_dict.keys():
                        pl = pre_layer_dict[name]
                    relative_name = nl + pl
                    occupied_core = []
                    for n in relative_name:
                        addr_ = None
                        if n in self.node_mapping_info_list.keys():
                            addr_ = self.node_mapping_info_list[n]
                        elif n + '.0.0.0' in self.node_mapping_info_list.keys():
                            n_ = n + '.0.0.0'
                            if n_ in self.node_mapping_info_list.keys():
                                addr_ = self.node_mapping_info_list[n_]
                        elif n in mapping_mesh_node.keys():
                            addr_ = mapping_mesh_node[n]
                        if addr_ != None:
                            core_id = int(addr_.split('.')[1].split(':')[1])
                            core = (core_id // mesh_width, core_id % mesh_width)
                            if core not in occupied_core:
                                occupied_core.append(core)
                    rest_possible_nodes = []
                    for x in all_points:
                        if x not in occupied_core:
                            rest_possible_nodes.append(x)
                    if rest_possible_nodes == []:
                        raise ValueError(f'满足 mapping 要求的节点 {name} 所需内存空间不足 !!! ')
                    try:
                        if occupied_core != []:
                            closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=occupied_core)
                        else:
                            closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=[(3, 0)])
                    except np.AxisError:
                        print()
                        print()
                        print()
                        print()
                        print()
                        print()
                        exit(1)
                    index_ = closest_point[0] * mesh_width + closest_point[1]
                    device_ref = f'{device_name}.cima-node:{index_}'
                    current_node = device_ref
                    mapping_info = CIMADeviceMappingInfo(index=[0, 0, 0], device=device_ref, address=0)
                    if name not in self.node_mapping_info.keys():
                        self.node_mapping_info[name] = []
                    self.node_mapping_info[name].append(mapping_info)
                    if name not in self.credit_len.keys():
                        self.credit_len[name] = []
                    self.credit_len[name].append(width)
                    if current_node not in linebuf_assigned.keys():
                        linebuf_assigned[current_node] = [0, 0]
                    if name not in self.in_line_buffer_addr.keys():
                        self.in_line_buffer_addr[name] = []
                    self.in_line_buffer_addr[name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
                    linebuf_assigned[current_node][1] += len_
                    while True:
                        if linebuf_assigned[current_node][1] % 32 == 0:
                            break
                        linebuf_assigned[current_node][1] += 1
                    mapping_mesh_node[name] = current_node
                elif layers[name].op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc', 'split', 'add', 'fused_add', 'fused_concat', 'concat']:
                    if layers[name].op.op_id in ['split', 'add', 'fused_add', 'fused_concat', 'concat']:
                        nl_ = name
                    else:
                        nl_ = name + f'.0.0.0'
                    assert nl_ in self.node_mapping_info_list.keys()
                    addr = self.node_mapping_info_list[nl_]
                    index_ = [0, 0, 0]
                    if name not in self.node_mapping_info.keys():
                        self.node_mapping_info[name] = []
                    if layers[name].op.op_id in ['split', 'add', 'fused_add', 'fused_concat', 'concat']:
                        value = 0
                        device_ref = '.'.join(addr.split('.')[:2])
                    elif 'cima-dmac' in addr:
                        value = 0
                        device_ref = addr
                    else:
                        device_ref = '.'.join(addr.split('.')[:-1])
                        value_ = addr.split('.')[-1].split(' ')
                        value = []
                        for v in range(len(value_)):
                            if v == 0:
                                value.append(int(value_[v].split('[')[1].split(',')[0]))
                            elif v == 3:
                                value.append(int(value_[v].split(']')[0]))
                            else:
                                value.append(int(value_[v].split(',')[0]))
                    mapping_info = CIMADeviceMappingInfo(index=index_, device=device_ref, address=value)
                    self.node_mapping_info[name].append(mapping_info)
                    current_node = '.'.join(addr.split('.')[:2])
                    if current_node not in linebuf_assigned.keys():
                        linebuf_assigned[current_node] = [0, 0]
                    if name not in self.credit_len.keys():
                        self.credit_len[name] = []
                    node_info = self.node_info[name]
                    if node_info['in_channel'] == 255:
                        warnings.warn(f'当前层 {name} 插入fake通道, 通道数从{in_channel} 变为 256!!!')
                        node_info['in_channel'] += 1
                    if node_info['op_type'] in ['matmul', 'fused_fc']:
                        len_ = node_info['in_channel']
                        self.credit_len[name].append(1)
                    elif node_info['op_type'] in ['conv2d', 'fused_conv2d']:
                        len_ = node_info['input_shape'][1] * node_info['in_channel'] * max(node_info['kernel_size'], node_info['stride'])
                        self.credit_len[name].append(node_info['input_shape'][1])
                    elif node_info['op_type'] in ['fused_concat', 'concat']:
                        len_ = node_info['in_channel'][0] * node_info['input_shape'][0][1] * 4
                        if name in ['Concat_208', 'Concat_281']:
                            len_ *= 2
                        self.credit_len[name].append(node_info['input_shape'][0][1])
                    elif node_info['op_type'] in ['split', 'add', 'fused_add']:
                        len_ = node_info['in_channel'][0] * node_info['input_shape'][0][1]
                        if CIMA_datawidth == 8:
                            if node_info['op_type'] in ['add', 'fused_add']:
                                len_ *= 4
                                if name in ['Add_56', 'Add_49', 'Conv_199_Add']:
                                    len_ *= 2
                        self.credit_len[name].append(node_info['input_shape'][0][1])
                    else:
                        raise ValueError(f"暂不支持 op_type: {node_info['op_type']}")
                    if name not in self.in_line_buffer_addr.keys():
                        self.in_line_buffer_addr[name] = []
                    self.in_line_buffer_addr[name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
                    linebuf_assigned[current_node][1] += len_
                    while True:
                        if linebuf_assigned[current_node][1] % 32 == 0:
                            break
                        linebuf_assigned[current_node][1] += 1
                    mapping_mesh_node[name] = current_node
                else:
                    raise ValueError(f'暂不支持的op: {layers[name].op.op_id}')
            count = 0
            for layer_name in next_layer_dict.keys():
                if 'graph_input' in layer_name:
                    continue
                current_layer = self.ir.layers[layer_name]
                if current_layer.type == 'op' and current_layer.op.op_id in ['fused_concat', 'concat', 'fused_add', 'add']:
                    distance_thr = 50
                    current_index_num = int(layer_name.split('_')[1])
                    for pl_ in pre_layer_dict[layer_name]:
                        pre_index_num = int(pl_.split('_')[1])
                        IsInsertDram = False
                        if pl_ in ['Conv_161_Concat_0', 'Conv_85', 'Conv_245_Concat_0', 'Conv_202_Add', 'Conv_161', 'Conv_245']:
                            IsInsertDram = True
                        if current_index_num - pre_index_num > distance_thr or IsInsertDram:
                            pl = pre_layer_dict[pl_]
                            nl = [layer_name]
                            pre_layer = self.ir.layers[pl_]
                            pre_node = int(mapping_mesh_node[pl_].split('.')[1].split(':')[1])
                            pre_node_coor = (pre_node // mesh_width, pre_node % mesh_width)
                            all_possible_nodes = ['DDR']
                            (count, linebuf_assigned) = self.make_CIMA_transfer_thread(count, pre_layer, pl_, mapping_mesh_node, pre_node_coor, [pre_node_coor], mesh_width, pl, nl, linebuf_assigned, all_possible_nodes)
            next_layer_dict = get_next_layer(self.ir.layers)
            for layer_name in next_layer_dict.keys():
                if 'graph_input' in layer_name:
                    continue
                current_layer = self.ir.layers[layer_name]
                if current_layer.type == 'op' and current_layer.op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc', 'maxpool2d', 'avgpool2d', 'global_avg_pool2d']:
                    mapped_name = layer_name + '.0.0.0'
                    if current_layer.op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d']:
                        current_node = int(mapping_mesh_node[layer_name].split('.')[1].split(':')[1])
                    else:
                        current_node = int(self.node_mapping_info_list[mapped_name].split('.')[1].split(':')[1])
                    current_node_coor = [current_node // mesh_width, current_node % mesh_width]
                    nl = next_layer_dict[layer_name]
                    pl = []
                    if layer_name in pre_layer_dict.keys():
                        pl = pre_layer_dict[layer_name]
                    if current_layer.op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc'] and len(next_layer_dict[layer_name]) == 1:
                        pe_relative = self.node_mapping_info_list[mapped_name].split('.')[2]
                        pe_number = int(pe_relative.split(':')[-1])
                        if current_node_coor[0] == 0 and pe_number == 0:
                            continue
                        if current_node_coor[0] == mesh_height - 1 and pe_number == 2:
                            continue
                        if current_node_coor[1] == mesh_width - 1 and pe_number == 1:
                            continue
                        if current_node_coor[1] == 0 and pe_number == 3:
                            continue
                        if 'graph_output' in nl:
                            next_node = 18
                        else:
                            next_node = int(mapping_mesh_node[nl[0]].split('.')[1].split(':')[1])
                        next_node_coor = (next_node // mesh_width, next_node % mesh_width)
                        if pe_number == 0:
                            if not (next_node_coor[1] == current_node_coor[1] and next_node_coor[0] < current_node_coor[0]):
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[0]):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                (count, linebuf_assigned) = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, current_node_coor, [next_node_coor], mesh_width, pl, nl, linebuf_assigned, all_possible_nodes)
                        elif pe_number == 2:
                            if not (next_node_coor[1] == current_node_coor[1] and next_node_coor[0] > current_node_coor[0]):
                                all_possible_nodes = []
                                for i in range(current_node_coor[0] + 1, mesh_height):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                (count, linebuf_assigned) = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, current_node_coor, [next_node_coor], mesh_width, pl, nl, linebuf_assigned, all_possible_nodes)
                        elif pe_number == 1:
                            if next_node_coor[1] <= current_node_coor[1]:
                                all_possible_nodes = []
                                for i in range(current_node_coor[1] + 1, mesh_width):
                                    if (current_node_coor[0], i) not in none_core:
                                        all_possible_nodes.append((current_node_coor[0], i))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                (count, linebuf_assigned) = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, current_node_coor, [next_node_coor], mesh_width, pl, nl, linebuf_assigned, all_possible_nodes)
                        elif pe_number == 3:
                            if next_node_coor[1] >= current_node_coor[1]:
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[1]):
                                    if (current_node_coor[0], i) not in none_core:
                                        all_possible_nodes.append((current_node_coor[0], i))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                (count, linebuf_assigned) = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, current_node_coor, [next_node_coor], mesh_width, pl, nl, linebuf_assigned, all_possible_nodes)
                        else:
                            raise ValueError(f'pe 相对位置错误!!! 不应该出现 {pe_number}!!!')
                    else:
                        next_node_coor = []
                        for nl_ in nl:
                            if 'graph_output' in nl_:
                                next_node_coor.append((3, 0))
                            else:
                                next_node = int(mapping_mesh_node[nl_].split('.')[1].split(':')[1])
                                next_node_coor.append((next_node // mesh_width, next_node % mesh_width))
                        all_possible_nodes = all_points + hosti_core + ddr_core
                        if current_layer.op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc']:
                            pe_relative = self.node_mapping_info_list[mapped_name].split('.')[2]
                            pe_number = int(pe_relative.split(':')[-1])
                            if pe_number == 0 and current_node_coor[0] != 0:
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[0]):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                            elif pe_number == 2 and current_node_coor[0] != mesh_height - 1:
                                all_possible_nodes = []
                                for i in range(current_node_coor[0], mesh_height):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                            elif pe_number == 1:
                                all_possible_nodes_new = []
                                for node in all_possible_nodes:
                                    if node[1] > current_node_coor[1] and node not in none_core:
                                        all_possible_nodes_new.append(node)
                                if all_possible_nodes_new == []:
                                    all_possible_nodes_new += ddr_core
                                all_possible_nodes = all_possible_nodes_new
                            elif pe_number == 3:
                                all_possible_nodes_new = []
                                for node in all_possible_nodes:
                                    if node[1] < current_node_coor[1] and node not in none_core:
                                        all_possible_nodes_new.append(node)
                                if all_possible_nodes_new == []:
                                    all_possible_nodes_new += ddr_core
                                all_possible_nodes = all_possible_nodes_new
                        (count, linebuf_assigned) = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, current_node_coor, next_node_coor, mesh_width, pl, nl, linebuf_assigned, all_possible_nodes)
                elif current_layer.type == 'op' and current_layer.op.op_id in ['fused_concat', 'concat', 'fused_add', 'add']:
                    current_node = int(mapping_mesh_node[layer_name].split('.')[1].split(':')[1])
                    current_node_coor = [current_node // mesh_width, current_node % mesh_width]
                    for il in current_layer.inputs:
                        all_possible_nodes = all_points + [(3, 0), (3, 5)]
                        ref_name = il.ref
                        last_layer_name = ref_name
                        if ':' in ref_name:
                            last_layer_name = ref_name.split(':')[0]
                        last_node = int(mapping_mesh_node[last_layer_name].split('.')[1].split(':')[1])
                        last_node_coor = (last_node // mesh_width, last_node % mesh_width)
                        if tuple(current_node_coor) == last_node_coor:
                            last_layer = self.ir.layers[last_layer_name]
                            nl = [layer_name]
                            pl = []
                            for pln in last_layer.inputs:
                                pl.append(pln.ref)
                            if last_node_coor in all_possible_nodes:
                                all_possible_nodes.remove(last_node_coor)
                            (count, linebuf_assigned) = self.make_CIMA_transfer_thread(count, last_layer, ref_name, mapping_mesh_node, last_node_coor, [current_node_coor], mesh_width, pl, nl, linebuf_assigned, all_possible_nodes)
            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
        else:
            raise ValueError(f'暂不支持设备 {self.hd_name[0]}的mapping!!!')

    def make_CIMA_transfer_thread(self, count, current_layer, layer_name, mapping_mesh_node, current_node_coor, next_node_coor, mesh_width, pl, nl, linebuf_assigned, all_possible_nodes):
        identity_name = f'identity_{count}'
        count += 1
        input_shape = current_layer.outputs[0]
        in_channel = input_shape.channel
        if in_channel == 255:
            warnings.warn(f'当前层 {identity_name} 插入fake通道, 通道数从{in_channel} 变为 256!!!')
            in_channel += 1
        height = input_shape.height
        width = input_shape.width
        len_ = math.ceil(in_channel * width)
        self.make_identity_op(input_shape, layer_name, identity_name)
        device_name = mapping_mesh_node[layer_name].split('.')[0]
        if all_possible_nodes == ['DDR']:
            closest_point = (3, 5)
            device_ref = f'{device_name}.cima-dram'
            index_ = closest_point[0] * mesh_width + closest_point[1]
            current_node = f'{device_name}.cima-node:{index_}'
            len_ *= 2
        else:
            rest_possible_nodes = []
            for x in all_possible_nodes:
                index_rpn = x[0] * mesh_width + x[1]
                device_ref_rpn = f'{device_name}.cima-node:{index_rpn}'
                if device_ref_rpn in linebuf_assigned.keys():
                    mem_occupied_size = linebuf_assigned[device_ref_rpn][1]
                else:
                    mem_occupied_size = 0
                if list(x) != current_node_coor and x not in next_node_coor:
                    rest_possible_nodes.append(x)
                if mem_occupied_size > self.Max_Memory_Size:
                    warnings.warn(f' 若算子 {identity_name} mapping 在 Core{x}, 则内存空间将超出限制, 内存空间大小为 {hex(mem_occupied_size)} !!!')
            occupied_core = [tuple(current_node_coor)] + next_node_coor
            try:
                if occupied_core != []:
                    closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=occupied_core)
                else:
                    closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=[(3, 0)])
            except np.AxisError:
                print()
                print()
                print()
                print()
                print()
                print()
                exit(1)
            index_ = closest_point[0] * mesh_width + closest_point[1]
            device_ref = f'{device_name}.cima-node:{index_}'
            current_node = device_ref
        if current_node not in linebuf_assigned.keys():
            linebuf_assigned[current_node] = [0, 0]
        mapping_info = CIMADeviceMappingInfo(index=[0, 0, 0], device=device_ref, address=0)
        if identity_name not in self.node_mapping_info.keys():
            self.node_mapping_info[identity_name] = []
        self.node_mapping_info[identity_name].append(mapping_info)
        mapping_mesh_node[identity_name] = current_node
        if identity_name not in self.credit_len.keys():
            self.credit_len[identity_name] = []
        if current_layer.op.op_id in ['matmul', 'fused_fc']:
            self.credit_len[identity_name].append(1)
        else:
            self.credit_len[identity_name].append(width)
        if identity_name not in self.in_line_buffer_addr.keys():
            self.in_line_buffer_addr[identity_name] = []
        self.in_line_buffer_addr[identity_name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
        linebuf_assigned[current_node][1] += len_
        while True:
            if linebuf_assigned[current_node][1] % 32 == 0:
                break
            linebuf_assigned[current_node][1] += 1
        for nl_ in nl:
            layer_inputs = self.ir.layers[nl_].inputs
            for li in layer_inputs:
                if li.ref == layer_name:
                    li.ref = identity_name
        return (count, linebuf_assigned)

    def find_closest_point(self, points, linebuf_assigned, data_len, mesh_width=6, exclude_points=[]):
        points = np.array(points)
        exclude_points = np.array(exclude_points)
        distances = cdist(points, exclude_points, metric='cityblock')
        distances = np.sum(distances, axis=1)
        sorted_index_list = sorted(range(len(list(distances))), key=lambda k: distances[k])
        if len(linebuf_assigned.keys()) != 0:
            device_name = list(linebuf_assigned.keys())[0].split('.')[0]
            CanMap = False
            for p_id in sorted_index_list:
                core_id = points[p_id]
                index_ = core_id[0] * mesh_width + core_id[1]
                device_ref = f'{device_name}.cima-node:{index_}'
                if device_ref in linebuf_assigned.keys():
                    if linebuf_assigned[device_ref][1] + data_len <= self.Max_Memory_Size:
                        min_mem_size_id = p_id
                        CanMap = True
                        break
                else:
                    min_mem_size_id = p_id
                    CanMap = True
                    break
            if not CanMap:
                raise ValueError(f'所有可能的Core都无法部署当前算子, 算子所需要的数据空间为：{data_len}!!!')
        else:
            min_mem_size_id = sorted_index_list[0]
        closest_point = tuple(points[min_mem_size_id]) if min_mem_size_id is not None else None
        return closest_point

    def Is_CIMA_mapped_layer(self, layer):
        if layer.type == 'op' and layer.op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc']:
            return True
        return False

    def make_identity_op(self, input_shape, ref_layer_name, identity_name):
        from e100_irtool import make_op
        op_ = make_op('identity')
        inputs_ = [dict(ref=ref_layer_name, channel=input_shape.channel, height=input_shape.height, width=input_shape.width)]
        outputs_ = [dict(channel=input_shape.channel, height=input_shape.height, width=input_shape.width)]
        self.ir.add_layer(identity_name, op=op_, inputs=inputs_, outputs=outputs_)

    def update_info(self):
        
        node_info = {}
        out_loop = 0
        for i in self.node_info.keys():
            if self.window_copy:
                [p, r, w, h] = self.split_num[i]
                calc_num = math.ceil(self.node_info[i]['calc_num'] / (p * r))
                out_loop = p
            else:
                [r, w, h] = self.split_num[i]
                calc_num = math.ceil(self.node_info[i]['calc_num'] / r)
                out_loop = r
            for j in range(out_loop):
                for k in range(h):
                    for l in range(w):
                        if self.window_copy:
                            new_name = i + '.' + str(j) + '.' + str(k) + '.' + str(l) + '_wd'
                        else:
                            new_name = i + '.' + str(j) + '.' + str(k) + '.' + str(l)
                        shape = self.split_node_weight[new_name]
                        in_pre = self.node_info[i]['in_precision']
                        out_pre = self.node_info[i]['out_precision']
                        node_info[new_name] = dict(shape=shape, calc_num=calc_num, in_precision=in_pre, out_precision=out_pre)
        return node_info
from ..device.a111 import *
from e100_irtool.core import make_ir, make_op, make_layer
from ..fused_op.op import *

class A111ComputeBaseLayer:

    def __init__(self, layer_name, tile_id=0, xb_id_list=[0], in_channel=0, out_channel=0, adc_range=1, relu=False, shift_num=1, weight_scale=1, relu_threshold=0):
        self.layer_name = layer_name
        self.tile_id = tile_id
        self.xb_id_list = xb_id_list
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.adc_range = adc_range
        self.relu = relu
        self.shift_num = shift_num
        self.weight_scale = weight_scale
        self.relu_threshold = relu_threshold

class A111ConvLayer(A111ComputeBaseLayer):

    def __init__(self, layer_name, *, tile_id=0, xb_id_list=[0], in_channel=0, out_channel=0, adc_range=1, relu=False, avgpool=False, shift_num=1, weight_scale=1, kernel_size=1, stride=1, output_pad=[0, 0, 0, 0], relu_threshold=0):
        super().__init__(layer_name, tile_id, xb_id_list, in_channel, out_channel, adc_range, relu, shift_num, weight_scale, relu_threshold)
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_pad = output_pad
        self.avgpool = avgpool

    def set_ref_name(self, ref_name):
        self.ref_name = ref_name

    def set_input_image_size(self, image_size):
        self.input_image_size = image_size

    def set_output_image_size(self, image_size):
        self.output_image_size = image_size

    def set_in_buffer_addr(self, buffer_addr):
        self.in_buffer_addr = buffer_addr

    def set_out_buffer_addr(self, buffer_addr):
        self.out_buffer_addr = buffer_addr

    def set_weight_shape(self, weight_shape):
        self.weight_shape = weight_shape

class A111FCLayer(A111ComputeBaseLayer):

    def __init__(self, layer_name, *, tile_id=0, xb_id_list=[0], in_channel=0, out_channel=0, adc_range=1, relu=False, shift_num=1, weight_scale=1, relu_threshold=0):
        super().__init__(layer_name, tile_id=tile_id, xb_id_list=xb_id_list, in_channel=in_channel, out_channel=out_channel, adc_range=adc_range, relu=relu, shift_num=shift_num, weight_scale=weight_scale, relu_threshold=relu_threshold)

    def set_ref_name(self, ref_name):
        self.ref_name = ref_name

    def set_input_image_size(self, image_size):
        self.input_image_size = image_size

    def set_output_image_size(self, image_size):
        self.output_image_size = image_size

    def set_in_buffer_addr(self, buffer_addr):
        self.in_buffer_addr = buffer_addr

    def set_out_buffer_addr(self, buffer_addr):
        self.out_buffer_addr = buffer_addr

    def set_weight_shape(self, weight_shape):
        self.weight_shape = weight_shape

class A111InputLayer:

    def __init__(self, layer_name, input_image_size):
        self.layer_name = layer_name
        self.input_image_size = input_image_size

class A111OutputLayer:

    def __init__(self, layer_name):
        self.layer_name = layer_name

    def set_output_image_size(self, output_image_size):
        self.output_image_size = output_image_size

class A111PipeGraph:

    def __init__(self):
        self.layers = []
        self.layer_graph = {}
        self.devices = {}

    def add_layer_list(self, layer_list):
        for l in layer_list:
            self.add_layer(l)

    def add_layer(self, layer):
        ISSameTileWithLastLayer = False
        if self.layers != []:
            last_layer = self.layers[-1]
        if isinstance(layer, A111InputLayer):
            self.layer_graph['graph_input'] = layer.layer_name
        elif isinstance(layer, A111OutputLayer):
            self.layer_graph[layer.layer_name] = 'graph_output'
            base = 1744830464
            start = self.layers[1].input_image_size[0] * self.layers[1].input_image_size[1] * self.layers[1].in_channel
            end = start + last_layer.output_image_size[0] * last_layer.output_image_size[1] * last_layer.out_channel
            last_layer.set_out_buffer_addr([base, start, end])
        elif isinstance(layer, A111ConvLayer):
            if len(self.layers) == 1:
                layer.set_ref_name('graph_input:0')
                in_image_size = last_layer.input_image_size
                in_addr = self.set_layer_in_buffer_addr(layer.tile_id, layer.xb_id_list[0], ISInputLayer=True)
                layer.set_in_buffer_addr(in_addr)
            else:
                layer.set_ref_name(last_layer.layer_name)
                in_image_size = last_layer.output_image_size
                self.layer_graph[last_layer.layer_name] = layer.layer_name
                if layer.tile_id == last_layer.tile_id:
                    ISSameTileWithLastLayer = True
                in_addr = self.set_layer_in_buffer_addr(layer.tile_id, layer.xb_id_list[0], ISSameTileWithLastLayer=ISSameTileWithLastLayer)
                layer.set_in_buffer_addr(in_addr)
                last_layer.set_out_buffer_addr(in_addr)
            layer.set_input_image_size(in_image_size)
            layer.set_weight_shape([layer.out_channel, layer.in_channel, layer.kernel_size, layer.kernel_size])
            assert len(in_image_size) == 2, f'图像大小格式 {in_image_size} 错误!!! 应为两个整数[H,W]组成的列表.'
            kernel = layer.kernel_size
            stride = layer.stride
            out_h = (in_image_size[0] - kernel) // stride + 1
            out_w = (in_image_size[1] - kernel) // stride + 1
            if layer.avgpool:
                out_h = out_h // 2
                out_w = out_w // 2
            out_h = out_h + layer.output_pad[0] + layer.output_pad[1]
            out_w = out_w + layer.output_pad[2] + layer.output_pad[3]
            out_image_size = [out_h, out_w]
            layer.set_output_image_size(out_image_size)
        elif isinstance(layer, A111FCLayer):
            if len(self.layers) == 1:
                layer.set_ref_name('graph_input')
                in_addr = self.set_layer_in_buffer_addr(layer.tile_id, layer.xb_id_list[0], ISInputLayer=True)
                layer.set_in_buffer_addr(in_addr)
            else:
                layer.set_ref_name(last_layer.layer_name)
                in_image_size = last_layer.output_image_size
                self.layer_graph[last_layer.layer_name] = layer.layer_name
                if layer.tile_id == last_layer.tile_id:
                    ISSameTileWithLastLayer = True
                in_addr = self.set_layer_in_buffer_addr(layer.tile_id, layer.xb_id_list[0], ISSameTileWithLastLayer=ISSameTileWithLastLayer)
                layer.set_in_buffer_addr(in_addr)
                last_layer.set_out_buffer_addr(in_addr)
            layer.set_weight_shape([layer.out_channel, layer.in_channel])
            layer.set_input_image_size([1, 1])
            layer.set_output_image_size([1, 1])
        else:
            raise ValueError(f'暂不支持格式{type(layer)}!!!')
        if isinstance(layer, A111ConvLayer) or isinstance(layer, A111FCLayer):
            tile_id = layer.tile_id
            if tile_id not in self.devices.keys():
                self.devices[tile_id] = {}
                self.devices[tile_id]['pool_en'] = [0, 0, 0, 0]
                self.devices[tile_id]['op_list'] = []
            self.devices[tile_id]['op_list'].append(layer.layer_name)
            if isinstance(layer, A111ConvLayer) and layer.avgpool:
                first_xb_id = layer.xb_id_list[0]
                tile_index = tile_id // 6
                assert tile_index <= 1
                self.devices[tile_id]['pool_en'][first_xb_id // 2 + tile_index * 2] = 1
        self.layers.append(layer)

    def set_layer_in_buffer_addr(self, tile_id, first_xb, ISInputLayer=False, ISSameTileWithLastLayer=False):
        
        if ISInputLayer:
            return (2013265920, 0, 16384)
        else:
            if ISSameTileWithLastLayer:
                base_addr = 2013265920
            else:
                assert tile_id >= 0 and tile_id <= 11
                base_addr = 1879048192 + 16777216 * (tile_id % 6)
            assert first_xb in [0, 2]
            start_addr = 0 + 16384 * (first_xb // 2)
            end_addr = 16384 + start_addr
            return (base_addr, start_addr, end_addr)

    def to_ir(self):
        self.ir = make_ir()
        layer_info = {}
        for l in self.layers:
            if isinstance(l, A111ConvLayer) or isinstance(l, A111FCLayer):
                layer_info[l.layer_name] = l
        in_layer = layer_info[self.layer_graph['graph_input']]
        inputs_dict = dict(channel=in_layer.in_channel, height=in_layer.input_image_size[0], width=in_layer.input_image_size[1])
        self.ir.add_layer('graph_input', type='input', inputs=[inputs_dict])
        for l in self.layers:
            if isinstance(l, A111ConvLayer):
                op_layer = l
                op_id = 'conv2d'
                if op_layer.relu or op_layer.avgpool:
                    op_id = 'fused_conv2d'
                    relu = None
                    pool = None
                    if op_layer.relu:
                        relu = dict(op_id='relu')
                    if op_layer.avgpool:
                        pool = dict(op_id='avgpool2d', kernel=2, stride=2, padding=0)
                    op_ = make_op(op_id, in_channel=op_layer.in_channel, out_channel=op_layer.out_channel, kernel=op_layer.kernel_size, stride=op_layer.stride, padding=0, relu=relu, pool=pool)
                else:
                    op_ = make_op(op_id, in_channel=op_layer.in_channel, out_channel=op_layer.out_channel, kernel=op_layer.kernel_size, stride=op_layer.stride, padding=0)
                in_info = [dict(ref=op_layer.ref_name, channel=op_layer.in_channel, height=op_layer.input_image_size[0], width=op_layer.input_image_size[1])]
                weight_info = dict(weight=dict(shape=op_layer.weight_shape))
                out_info = [dict(channel=op_layer.out_channel, height=op_layer.output_image_size[0], width=op_layer.output_image_size[1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info, weights=weight_info)
                in_buffer_addr = dict(base=hex(op_layer.in_buffer_addr[0]), start=hex(op_layer.in_buffer_addr[1]), end=hex(op_layer.in_buffer_addr[2]))
                out_buffer_addr = dict(base=hex(op_layer.out_buffer_addr[0]), start=hex(op_layer.out_buffer_addr[1]), end=hex(op_layer.out_buffer_addr[2]))
                mapping_info = []
                c = 0
                assert op_layer.in_channel * op_layer.kernel_size ** 2 % len(op_layer.xb_id_list) == 0
                addr_h = op_layer.in_channel * op_layer.kernel_size ** 2 // len(op_layer.xb_id_list)
                addr_w = op_layer.out_channel
                for xb in op_layer.xb_id_list:
                    device_ref = f'a111-0.a111-npu:0.a111-tile:{op_layer.tile_id}.a111-xb:{xb}'
                    addr_value = [0, 0, addr_h, addr_w]
                    mapping_info.append(A111DeviceMappingInfo(index=[0, c, 0], device=device_ref, address=addr_value))
                    c += 1
                a111_mapping_info = A111MappingInfo(row_split_num=len(op_layer.xb_id_list), input_buffer_addr=in_buffer_addr, output_buffer_addr=out_buffer_addr, in_buf_type=6, out_buf_type=6, mappings=mapping_info)
                out_pad = {'top': op_layer.output_pad[0], 'bottom': op_layer.output_pad[1], 'left': op_layer.output_pad[2], 'right': op_layer.output_pad[3]}
                buffer_wrap = 1
                if self.layer_graph[op_layer.layer_name] == 'graph_output':
                    buffer_wrap = 0
                a111_calc_info = A111CalcInfo(weight_scale=op_layer.weight_scale, adc_range=op_layer.adc_range, relu_threshold=op_layer.relu_threshold, shift_num=op_layer.shift_num, buffer_wrap_en=buffer_wrap, output_padding=out_pad)
                self.ir.add_layer(op_layer.layer_name, current_layer, a111_mapping_info=a111_mapping_info, a111_calc_info=a111_calc_info)
            elif isinstance(l, A111FCLayer):
                op_layer = l
                op_id = 'matmul'
                if op_layer.relu:
                    op_id = 'fused_fc'
                    relu = dict(op_id='relu')
                    op_ = make_op(op_id, in_channel=op_layer.in_channel, out_channel=op_layer.out_channel, relu=relu)
                else:
                    op_ = make_op(op_id, in_channel=op_layer.in_channel, out_channel=op_layer.out_channel)
                in_info = [dict(ref=op_layer.ref_name, channel=op_layer.in_channel, height=op_layer.input_image_size[0], width=op_layer.input_image_size[1])]
                weight_info = dict(weight=dict(shape=op_layer.weight_shape))
                out_info = [dict(channel=op_layer.out_channel, height=op_layer.output_image_size[0], width=op_layer.output_image_size[1])]
                current_layer = make_layer(op=op_, inputs=in_info, outputs=out_info, weights=weight_info)
                in_buffer_addr = dict(base=op_layer.in_buffer_addr[0], start=op_layer.in_buffer_addr[1], end=op_layer.in_buffer_addr[2])
                out_buffer_addr = dict(base=op_layer.out_buffer_addr[0], start=op_layer.out_buffer_addr[1], end=op_layer.out_buffer_addr[2])
                mapping_info = []
                c = 0
                for xb in op_layer.xb_id_list:
                    device_ref = f'a111-0.a111-npu:0.a111-tile:{op_layer.tile_id}.a111-xb:{xb}'
                    addr_h = op_layer.in_channl * op_layer.kernel_size ** 2
                    addr_w = op_layer.out_channel
                    addr_value = [0, 0, addr_h, addr_w]
                    mapping_info.append(A111DeviceMappingInfo(index=[0, c, 0], device=device_ref, address=addr_value))
                    c += 1
                a111_mapping_info = A111MappingInfo(row_split_num=len(op_layer.xb_id_list), input_buffer_addr=in_buffer_addr, output_buffer_addr=out_buffer_addr, in_buf_type=6, out_buf_type=6, mappings=mapping_info)
                buffer_wrap = 1
                if self.layer_graph[op_layer.layer_name] == 'graph_output':
                    buffer_wrap = 0
                a111_calc_info = A111CalcInfo(weight_scale=op_layer.weight_scale, adc_range=op_layer.adc_range, relu_threshold=op_layer.relu_threshold, shift_num=op_layer.shift_num, buffer_wrap_en=buffer_wrap)
                self.ir.add_layer(op_layer.layer_name, current_layer, a111_mapping_info=a111_mapping_info, a111_calc_info=a111_calc_info)
        out_layer = layer_info[self.layers[-1].layer_name]
        outputs_dict = dict(ref=out_layer.layer_name, channel=out_layer.out_channel, height=out_layer.output_image_size[0], width=out_layer.output_image_size[1])
        self.ir.add_layer('graph_output', type='output', inputs=[outputs_dict])
        self.ir.add_device('a111-0', 'a111-npu', number=1)
        for (k, v) in self.devices.items():
            self.ir.devices['a111-0'].devices[f'a111-tile:{k}'].info.op_list = v['op_list']
            self.ir.devices['a111-0'].devices[f'a111-tile:{k}'].info.pool0_en = v['pool_en'][0]
            self.ir.devices['a111-0'].devices[f'a111-tile:{k}'].info.pool1_en = v['pool_en'][1]
            self.ir.devices['a111-0'].devices[f'a111-tile:{k}'].info.pool2_en = v['pool_en'][2]
            self.ir.devices['a111-0'].devices[f'a111-tile:{k}'].info.pool3_en = v['pool_en'][3]
        return self.ir
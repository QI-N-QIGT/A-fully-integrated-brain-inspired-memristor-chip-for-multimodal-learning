from .fused_numpy import FusedOpNumpyRuntime, numpy
# from .utils import *
from e100_irmapper.device.a111 import MappedLayer # noqa
from typing import Callable
from e100_irtool.core import BaseIR
from e100_irtool.tools import flatten_layers  # noqa
from .tileop import TileOpObj
import math
from ...cimtensor import CIMNumpyTensor as CNT
import time
time_record = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))

class A111NumpyRT(FusedOpNumpyRuntime):
    
    name = "A111_chip"
    
    def __init__(self, *, mapped_array_data=None,):
        super().__init__()
        self.mapped_array_data = mapped_array_data
    
    def init_rpc(self, ir, simulation = False):
        # init rpc api
        self.rpc_api = {}
        if not simulation: 
            for name, device in ir.devices.items():
                if device.ip != None:
                    from rpc_api.callable_api import init_api
                    self.rpc_api[name] = init_api(device.ip) 
    
    def run_ir(self, ir, inputs, weights=None, *, outputs=None, callback=None,
               logfile={'dump_reg':f'reg_file/dumped_reg_file_{time_record}.txt', 
                        'dump_serial_script':False, 'verbose':True,
                        'log_file':f'log_file/log_file_{time_record}.txt'}):
        assert isinstance(ir, BaseIR), f'invalid IR type={type(ir)}'
        layers = ir.flatten_layers()
        devices = ir.devices
        inp, oup = ir.get_io_layers(layers)
        inl, oul = layers[inp], layers[oup]
        if isinstance(inputs, dict):
            data = {k: tuple(v) if isinstance(v, (tuple, list)) else (v,)
                    for k, v in inputs.items() if v is not None}
        elif isinstance(inputs, (tuple, list)):
            assert len(inputs) == len(inl.inputs)
            data = {inp: tuple(inputs)}
        else:
            data = {inp: (inputs,)}

        for k, v in data.items():
            assert k in layers, f'invalid input name {k!r}'
            assert isinstance(v, (tuple, list)), \
                f'invalid inputs type {type(v)}'
                
        ons = None
        if outputs is not None:
            if isinstance(outputs, str):
                ons = set(outputs)
            elif outputs is True:
                ons = set(layers.keys()) - {inp, oup}
            elif isinstance(outputs, (tuple, list, set)):
                ons = set(outputs)
            for k in ons:
                assert k in layers, f'invalid output name {k!r}'

        # assert isinstance(weights, dict), f'invalid weights {type(weights)}'

        if callback is not None:
            assert isinstance(callback, Callable), \
                f'invalid callback {type(callback)}'
        
        # 收集所有在A111上运行的tile，并配置tileop
        a111_layer = {}
        config_tileop_dict = {}
        device_info = {}
        last_esram_addr = []
        if isinstance(inputs, CNT):
            batch_num =  inputs.data.shape[0]
        
        if devices != None:
            # self.init_rpc(ir)
            for device_name, device in devices.items():
                for tile_name, tile in device.devices.items():
                    if tile.kind == 'a111-tile':
                        if tile.info.op_list != None:
                            for layer_name in tile.info.op_list:
                                assert layer_name not in a111_layer
                                a111_layer.update({layer_name:f"{device_name}_{tile_name}"})
                            # config tileop
                            tileop_config, last_esram_addr = self.config_tileop(tile_name, tile, layers, batch_num=batch_num, 
                                                                                last_esram_addr=last_esram_addr)
                            config_tileop_dict.update({f"{device_name}_{tile_name}":tileop_config})
                            device_info.update({f"{device_name}_{tile_name}":tile})
                            
        
        # run
        configured_tile_list = []
        # last_esram_addr = []
        for name, layer in layers.items():
            
            if name in data:
                continue    # layer is done
           
            if name == oup:
                break       # output layer
            
            if any(dd.parse_ref()[0] not in data for dd in layer.inputs):
                continue    # layer can't be run
            x = []
            for dd in layer.inputs:
                nm, idx = dd.parse_ref()
                x.append(data[nm][0 if idx is None else idx])
            
           
            tile_wts = {}
            tile_ats = {}

            tile_ats.update({f"{name}_ats":layer.op.get_attrs()})
            if "auto_pad" in layer.op.get_attrs():
                tile_ats[f"{name}_ats"].pop("auto_pad")
            y = None
            if layer.op.op_id in ['conv2d','matmul','fc','linear','fused_conv2d','fused_fc']:
                
                if name in a111_layer.keys():
                    if a111_layer[name] not in configured_tile_list:
                        
                        # config
                        tile_device_info = [device_info[a111_layer[name]]]
                        tileop_info = [config_tileop_dict[a111_layer[name]]]
                        tile_layer_info = [layers]
                        device_name = a111_layer[name].split('_')[0]
                        rpc = [self.rpc_api[device_name]]
                        # runtile
                        configured_tile_list.append(a111_layer[name])
                        y = self.run_tile(a111_layer[name], *x, *tile_layer_info,
                              *tile_device_info, *tileop_info, *rpc, **tile_ats, **logfile)
                        # last op 
                        last_layer_name = device_info[a111_layer[name]].info.op_list[-1]
                        if not isinstance(y, (tuple, list)):
                            y = (y,)
                        data[last_layer_name] =  tuple(y)
                    else:
                        # 已经运行
                        continue
                else:
                    tile_layer_info = {}     
                    tile_layer_info.update({f"{name}_info":layer})
                    for k in layer.op.weights:
                        wn = f'{name}.{k}'
                        if k not in layer.op.optional_weights:
                            assert wn in weights, f'missing weight {wn}'
                        tile_wts[f"{name}.{k}"] = weights.get(wn)
        
                    y = self.run_layer( name, *x, **tile_wts, **tile_ats, **tile_layer_info,)
            assert y != None
            if not isinstance(y, (tuple, list)):
                y = (y,)
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=tile_wts,
                        attrs=tile_ats, outputs=y)
            data[name] = tuple(y)
            if ons is not None and all(k in data for k in ons):
                break       # all outputs are ready
            
        if ons is not None:
            res = {}
            for k in ons:
                v = data[k]
                res[k] = v[0] if len(v) == 1 else v
            if isinstance(outputs, str):
                res = iter(res.values()).next()
        else:
            res = []
            for dd in oul.inputs:
                nm, idx = dd.parse_ref()
                res.append(data[nm][0 if idx is None else idx])
            if len(res) == 1:
                res = res[0]

        return res

    def run_op(self, op_id, *args, **kwargs):
        fn = getattr(self, f'fn_{op_id}', None)
        assert isinstance(fn, Callable), f'fn_{op_id} is not a function'
        return fn(*args, **kwargs)
        
    def run_layer(self,layer_name, *args, **kwargs):
        # input data
        x = args[0]
        
        assert f"{layer_name}_info" in kwargs.keys()
        layer = kwargs[f"{layer_name}_info"]
        layer_info = {"layer_info":layer}
        layer_weight = {}
        layer_attr  = {}
        if f"{layer_name}.weight" in kwargs.keys():
            layer_weight = {"layer_weight":{"weight":kwargs[f"{layer_name}.weight"]}}
        else:
            layer_weight = {"layer_weight":{"weight":None}}
        if f"{layer_name}.bias" in kwargs.keys():
            layer_weight["layer_weight"].update({"bias":kwargs[f"{layer_name}.bias"]})
        else:
            layer_weight["layer_weight"].update({"bias":None})
            
        if f"{layer_name}_ats" in kwargs.keys():
            layer_attr = {"layer_ats":kwargs[f"{layer_name}_ats"]}
        if layer.op.op_id in ['fused_conv2d', 'fused_fc']:
            x = self.run_op(layer.op.op_id, x, **layer_info, **layer_weight, **layer_attr)
        else:
            x = self.run_op(layer.op.op_id, x, **layer_weight["layer_weight"], **layer_attr["layer_ats"])
        return x 
    
    def run_tile(self, tile_name, *args, **kwargs):
        x = args[0]
        tile_layer_info = args[1]
        tile_device_info = args[2]
        tileopobj_info = args[3]
        rpc_api = args[4]
        # config
        rpc_api.call('a111_config', tileopobj_info)
        # runtile
        
        in_data = x.data # in data
        batch_num = in_data.shape[0]
        in_scale = x.scale
        # 配置输入参数
        tile_id = tileopobj_info.tile_id
        xb_id_list = tileopobj_info.xb_id_list
        tile_mode = tileopobj_info.tile_mode
        first_layer_name = tile_device_info.info.op_list[0]
        first_layer =  tile_layer_info[first_layer_name]
        last_layer_name = tile_device_info.info.op_list[-1]
        last_layer =  tile_layer_info[last_layer_name]
        first_op_type = 'FC'
        last_op_type = 'FC'
        if first_layer.op.op_id in ['conv2d','fused_conv2d']:
            first_op_type = 'CONV'
        if last_layer.op.op_id in ['conv2d', 'fused_conv2d']:
            last_op_type = 'CONV'
            
        output_feature_size = None
        if last_op_type == 'FC':
            output_feature_size = [batch_num, last_layer.outputs[0].channel]
        else:
            output_feature_size = [batch_num, last_layer.outputs[0].channel,
                                   last_layer.outputs[0].height, last_layer.outputs[0].width]
        # in esram addr
        in_addr_esram = tile_device_info.info.in_esram_addr[0]
        # out esram addr
        out_addr_esram = tile_device_info.info.out_esram_addr[0]

        last_relu = False
        if  last_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
            last_relu = True
        # numpy 数组 转为 list
        in_data = in_data.tolist()
        output = rpc_api.call('a111_run', tile_id, xb_id_list, tile_mode, in_data, output_feature_size,
                             in_addr_esram=in_addr_esram, out_addr_esram = out_addr_esram, first_op_type = first_op_type, 
                             last_op_type = last_op_type, last_relu = last_relu, dump_reg = kwargs['dump_reg'], 
                             dump_serial_script = kwargs['dump_serial_script'], verbose = kwargs['verbose'], 
                             log_file = kwargs['log_file'])
        output = numpy.array(output)

        out_, out_scale = CNT.to_cimtensor(data=output).items
        total_scale = in_scale * out_scale
        
        return CNT(data=out_, scale=total_scale) 

    def config_tileop(self, tile_name, tile_info, layer_info, batch_num=1, last_esram_addr = []):
        
        # 区分是xb0,1,2,3 还是 xb4,5,6,7
        res_num = int(tile_name.split(':')[-1]) % 2
        # tile_id
        tile_id = int(tile_name.split(':')[-1]) // 2
        # xb_id_list
        xb_id_list =  []
        # xb_column_list & xb_column_num_list
        xb_start_column_list = []
        xb_column_num_list = []
        # adc_range list
        adc_range_list = []
        # 
        layer_name_xb_id_dict = {}

        # 根据输入的batch修改tile device info 的 out esram 地址
        if batch_num > 1:
            origin_in_len = tile_info.info.in_esram_addr[1]
            origin_out_len = tile_info.info.out_esram_addr[1]
            if last_esram_addr == []:
                out_addr_esram = origin_in_len * batch_num  + tile_info.info.in_esram_addr[0]
                while True:
                    if out_addr_esram % 512 == 0:
                        break
                    out_addr_esram += 1
                    assert out_addr_esram <= 0x300000  # 输出地址小于3M
                tile_info.info.out_esram_addr = [out_addr_esram, batch_num * origin_out_len]
            else:
                tile_info.info.in_esram_addr =  last_esram_addr
                out_addr_esram = origin_in_len * batch_num + last_esram_addr[0]
                while True:
                    if out_addr_esram % 512 == 0:
                        break
                    out_addr_esram += 1
                    assert out_addr_esram <= 0x300000  # 输出地址小于3M
                tile_info.info.out_esram_addr = [out_addr_esram, batch_num * origin_out_len]
            last_esram_addr = [out_addr_esram, origin_out_len]

        
        for layer_name in tile_info.info.op_list:
            layer_ = layer_info[layer_name]
            mapped_device = layer_.a111_mapping_info.mappings
            xb_id_ = []
            for k, v in mapped_device.items():
                xb_id = int(v.device.split(':')[-1])
                xb_id_list.append(xb_id)
                xb_id_.append(xb_id)
                start_column = v.address[1]
                column_num = v.address[3]
                assert start_column in [0, 32, 64, 128]
                xb_start_column_list.append(start_column // 32)
                if layer_.op.op_id in ["fused_fc", "matmul", "fc", "linear"]:
                    if column_num <= 32:  
                        column_num = 32
                    elif column_num <= 64:
                        column_num = 64
                    else:
                        column_num = 128
                if column_num == 32:
                    xb_column_num_list.append(0)
                elif column_num == 64:
                    xb_column_num_list.append(1)
                elif column_num == 128:
                    xb_column_num_list.append(3)
                else:
                    raise ValueError(f" 硬件不支持column num: {column_num} !!!")
            layer_name_xb_id_dict[layer_name] = xb_id_
            # adc range 
            adc_range_list.append(layer_.a111_calc_info.adc_range)
        # tile_mode
        tile_mode = tile_info.info.tile_mode
        # pool mode
        pool_en_list = [tile_info.info.pool0_en, tile_info.info.pool1_en,
                        tile_info.info.pool2_en, tile_info.info.pool3_en,]
        # xb_arr_sel 
        xb_arr_sel = 0
        if res_num == 1:
            xb_arr_sel = 3 
        # xbg mode
        xbg_mode_list = [0,0,0,0]
        xbg_in_pix_type_list=[3,3,3,3]
        xbg_out_pix_type_list=[3,3,3,3]
        xbg_kernel_type_list = [0,0,0,0]
        xbg_toggle_en0_list = [0,0,0,0]
        xbg_toggle_bit0_list = [0,0,0,0]
        xbg_calc_mode_list = [0,0,0,0]
        xbg_tile_buf_en0_list = [0,0,0,0]
        xbg_tile_cal_en0_list = [0,0,0,0]
        xbg_relu_en_list = [0,0,0,0]
        xbg_fcn_en0_list=[0,0,0,0]
        # input addr
        input_addr_list = [0,0,0,0]
        input_len_list = [0,0,0,0]
        in_img_size_list = [[1,1],[1,1],[1,1],[1,1]]
        # output addr
        output_addr_list = [0x0, 0x0, 0x0, 0x0]
        out_img_size_list = [[1,1],[1,1],[1,1],[1,1]]
        xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
        # buf type
        in_buf_type_list = [0x0, 0x0, 0x0, 0x0]
        out_buf_type_list = [0x0, 0x0, 0x0, 0x0]
        # linebuf addr
        linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
        linebuf_width_list = [0x0, 0x0 , 0x0, 0x0]
        # sfu and shift
        relu_th_list = [0x0, 0x0, 0x0, 0x0]
        act_mode_list = [0x0, 0x0, 0x0, 0x0]
        shift_list = [0x0, 0x0, 0x0, 0x0]
        
        for layer_name in tile_info.info.op_list:
            first_layer_id = layer_name_xb_id_dict[layer_name][0]
            if first_layer_id == 0:
                index = res_num * 2
            elif first_layer_id == 2:   
                index = res_num * 2 +  first_layer_id - 1
            layer_ = layer_info[layer_name]
            # print(index)
            # input()
            if len(layer_name_xb_id_dict[layer_name]) == 1:
                xbg_mode_list[index] = 0
            elif len(layer_name_xb_id_dict[layer_name]) == 2:
                xbg_mode_list[index] = 1
            elif len(layer_name_xb_id_dict[layer_name]) == 4:
                xbg_mode_list[index] = 2
            else:
                raise ValueError(f' 硬件不支持 {layer_name_xb_id_dict[layer_name]} !!!')
            # para type
            # 暂不支持并行计算 TODO
            # xb op mode
            # 默认输出小于128通道，TODO
            # xb calc mode
            # 默认计算分为高低4bit，TODO
            # xb in pix type
            in_pix_channel = layer_.op.in_channel
            assert in_pix_channel >= 4
            xbg_in_pix_type_list[index] = int(math.log(in_pix_channel,2)- 2)
            # xb out pix type
            out_pix_channel = layer_.op.out_channel
            assert out_pix_channel >= 8
            xbg_out_pix_type_list[index] = int(math.log(out_pix_channel,2)- 3)
            # xb kernel type
            if layer_.op.op_id in ['conv2d', 'fused_conv2d']:
                kernel_size = layer_.kernel
                if kernel_size == 1:
                    xbg_kernel_type_list[index] = 0
                elif kernel_size == 3:
                    xbg_kernel_type_list[index] = 1
                elif kernel_size == 7:
                    xbg_kernel_type_list[index] = 2
            # xbg pool mode list
            # 默认为max，TODO
            # xbg toggle bit
            # xbg_toggle_en0_list[index] = 1
            xbg_toggle_bit0_list[index] = 1
            xbg_calc_mode_list[index] = 3
            xbg_tile_buf_en0_list[index] = 1
            xbg_tile_cal_en0_list[index] = 1
            # relu en
            if layer_.op.op_id not in ['fused_fc', 'fused_conv2d']:
                xbg_relu_en_list[index] = 1
            elif layer_.op.relu == None:
                xbg_relu_en_list[index] = 1
            # xbg fc en
            if layer_.op.op_id in ['matmul', 'fc', 'linear', 'fused_fc']:
                xbg_fcn_en0_list[index] = 1
            # input addr
            input_addr_list[index] = layer_.a111_mapping_info.input_tile_buffer_addr[0]
            input_len_list[index] = layer_.a111_mapping_info.input_tile_buffer_addr[1]
            # in buf type
            if input_len_list[index] <= 0x3000:
                in_buf_type_list[index] = input_len_list[index] // (0x800 + 1)
            elif input_len_list[index] <= 0x4000:
                in_buf_type_list[index] = 6
            elif input_len_list[index] <= 0x8000:
                in_buf_type_list[index] = 7
            else:
                raise ValueError(f'数据容量 {input_len_list[index]} 长度超过 buffer size 0x8000 !!!')
            
            in_img_size_list[index] = [layer_.inputs[0].height, layer_.inputs[0].width]
            # output addr
            
            if layer_.a111_mapping_info.output_tile_buffer_addr != None:
                output_addr_list[index] = 0x78000000 + layer_.a111_mapping_info.output_tile_buffer_addr[0]
            else:
                output_addr_list[index] = 0x68000000 + tile_info.info.out_esram_addr[0]
            out_img_size_list[index] = [layer_.outputs[0].height, layer_.outputs[0].width]
            
            out_channel = layer_.outputs[0].channel
            if layer_.op.op_id in ['conv2d', 'fused_conv2d']:
                xbg_axi_cnt_list[index] = layer_.outputs[0].height * layer_.outputs[0].width * out_channel
            else:
                assert out_channel <= 128
                if out_channel <= 32:
                    xbg_axi_cnt_list[index] = 32
                elif out_channel <= 64:
                    xbg_axi_cnt_list[index] = 64
                else:
                    xbg_axi_cnt_list[index] = 128
            if 'fused' not in layer_.op.op_id or layer_.op.relu == None:
                xbg_axi_cnt_list[index] = 2 * xbg_axi_cnt_list[index]
            # out buf type
            if xbg_axi_cnt_list[index] <= 0x3000:
                out_buf_type_list[index] = xbg_axi_cnt_list[index] // (0x800 + 1)
            elif xbg_axi_cnt_list[index] <= 0x4000:
                out_buf_type_list[index] = 6
            elif xbg_axi_cnt_list[index] <= 0x8000:
                out_buf_type_list[index] = 7
            else:
                raise ValueError(f'数据容量 {xbg_axi_cnt_list[index]} 长度超过 buffer size 0x8000 !!!')

            linebuf_addr_offset_list[index] = layer_.a111_mapping_info.input_tile_buffer_addr[0]
            padding = 0
            if layer_.op.op_id in ['conv2d', 'fused_conv2d']:
                padding = layer_.op.padding
            linebuf_width_list[index] = (layer_.inputs[0].width + 2 * padding) * layer_.op.in_channel  
            # sfu and shift
            relu_th_list[index] = layer_.a111_calc_info.relu_threshold
            # act mode 默认
            shift_list[index] = layer_.a111_calc_info.shift_num
        # res 
        res_in_sel = 0
        res_out_sel = 0
        res_9bit_en = 0
        last_layer = layer_info[tile_info.info.op_list[-1]]
        # Not relu
        if 'fused' not in last_layer.op.op_id or last_layer.op.relu == None:
            res_in_sel = (xb_id_list[-1] + 4 * res_num) // 2
            res_out_sel = (xb_id_list[-1] + 4 * res_num) // 2
            res_9bit_en = 1
        
        # 判断各层输出是否需要pad
        pad_en_list = [0,0,0,0]
        if last_layer.op.op_id in ['conv2d','fused_conv2d'] and last_layer.op.padding == 1:
            index_ = res_num * 2 +  xb_id_list[0]
            pad_en_list[index_] = 1
        
        return TileOpObj(tile_id=tile_id, xb_id_list=xb_id_list,
                        tile_mode = tile_mode, pool_en_list = pool_en_list,
                        xb_arr_sel = xb_arr_sel,# tile mode 
                        
                        xbg_mode_list = xbg_mode_list, xbg_calc_mode_list=xbg_calc_mode_list, xbg_in_pix_type_list=xbg_in_pix_type_list,
                        xbg_out_pix_type_list = xbg_out_pix_type_list,  xbg_kernel_type_list=xbg_kernel_type_list, 
                        xbg_toggle_en0_list=xbg_toggle_en0_list, xbg_toggle_bit0_list=xbg_toggle_bit0_list,
                        xbg_tile_buf_en0_list=xbg_tile_buf_en0_list, xbg_tile_cal_en0_list=xbg_tile_cal_en0_list, xbg_fcn_en0_list=xbg_fcn_en0_list,
                        xbg_relu_en_list=xbg_relu_en_list, # xbg mode

                        xb_start_column_list = xb_start_column_list, xb_column_num_list = xb_column_num_list, # xb column
                        input_addr_list = input_addr_list, input_len_list = input_len_list, in_img_size_list = in_img_size_list , # input 
                        output_addr_list = output_addr_list , out_img_size_list = out_img_size_list, xbg_axi_cnt_list = xbg_axi_cnt_list, # output 
                        in_buf_type_list=in_buf_type_list, out_buf_type_list=out_buf_type_list, # buf type
                        linebuf_addr_offset_list = linebuf_addr_offset_list, linebuf_width_list = linebuf_width_list, # linebuf
                        relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list, # sfu
                        adc_range_list = adc_range_list, # xb adc range
                        res_in_sel = res_in_sel, res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, # res_9bit , when relu == false, res_9bit_en = 1
                        pad_en_list = pad_en_list # padding
                        ), last_esram_addr  
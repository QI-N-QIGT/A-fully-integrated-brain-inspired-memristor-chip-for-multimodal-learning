from ..numpy import NumpyRuntime
from ..cimtensor import CIMNumpyTensor as CNT
from .utils import *
from e100_irmapper.device.c200 import MappedLayer
from typing import Callable
from e100_irtool.core import BaseIR
from e100_irtool.tools import flatten_layers
from e100_irtool.runtime.utils import conv_t_shapes
import copy
import warnings

class C200NumpyRT(NumpyRuntime):
    name = 'C200'
    tensor_type = CNT
    quant_all = False
    chip_quant_method = 'Uniform'
    half_level = 127
    weight_format = 'CHW'
    thr = None

    def __init__(self, *, mapped_array_data=None, device_adc_scale_data=None, quant_all=False, chip_quant_method='Uniform', half_level=127, weight_format='CHW', thr=None, macro_method=False, multi_batch=False, clamp_cpu=None):
        
        super().__init__()
        self.mapped_array_data = mapped_array_data
        self.device_adc_scale_data = device_adc_scale_data
        self.half_level = half_level
        self.quant_all = quant_all
        self.weight_format = weight_format
        self.chip_quant_method = chip_quant_method
        self.thr = thr
        self.macro_method = macro_method
        self.multi_batch = multi_batch
        self.clamp_cpu = clamp_cpu
        self.rpc_api = {}

    def init_rpc(self, ir, simulation=False):
        if not simulation:
            for (name, device) in ir.devices.items():
                if device.ip != None:
                    from rpc_api.callable_api import init_api
                    self.rpc_api[name] = init_api(device.ip)

    def get_device_and_addr(self, ir, layer_name=None):
        device_name = []
        addr = []
        if layer_name == None:
            name = []
            for (n_, layer_) in ir.layers.items():
                if layer_.type != 'op':
                    continue
                if layer_.op.op_id in ['conv2d', 'fc', 'matmul', 'linear', 'conv_transpose2d']:
                    assert layer_.c200_mapping_info != None
                    if layer_.c200_mapping_info.runtime == 'c200':
                        name.append(n_)
        else:
            name = layer_name
        for n in name:
            device_ = []
            addr_ = []
            assert n in list(ir.layers.keys())
            mappings = ir.layers[n].c200_mapping_info.mappings
            for (key, value) in mappings.items():
                device_.append(value.device)
                addr_.append(value.address)
            device_name.append(device_)
            addr.append(addr_)
        return (device_name, addr, name)

    def program(self, ir, array_data, layer_name=None, program_times=3):
        (device, addr, program_name) = self.get_device_and_addr(ir, layer_name=layer_name)
        assert len(device) == len(program_name)
        assert len(addr) == len(device)
        for index in range(len(program_name)):
            print()
            programming_array_data = {}
            assert len(device[index]) == len(addr[index])
            for d in range(len(device[index])):
                device_ = device[index][d]
                assert device_ in list(array_data.keys())
                programming_array_data[device_] = array_data[device_]
                addr_ = addr[index][d]
                for (k, v) in programming_array_data.items():
                    assert addr_[0] <= 576
                    assert addr_[0] + addr_[2] <= 576
                    assert addr_[1] <= 128
                    assert addr_[1] + addr_[3] <= 128
                    device_name = k.split('.')[0]
                    assert device_name in self.rpc_api.keys()
                    if 'rram-144k' in k:
                        array_idx = int(k.split(':')[-1])
                        api = self.rpc_api[device_name]
                        v1 = v[addr_[0]:addr_[0] + addr_[2], addr_[1]:addr_[1] + addr_[3]]
                        assert v1.max() <= 7
                        assert v1.min() >= -8
                        v_ = (v1 + 8).tolist()
                        api.call('c200_program', array_idx, v_, addr_, program_times)
                    else:
                        raise ValueError(f'暂不支持设备: {device_name} !!!')

    def read(self, ir, layer_name=None):
        (device, addr, program_name) = self.get_device_and_addr(ir, layer_name=layer_name)
        assert len(device) == len(program_name)
        assert len(addr) == len(device)
        chip_data = {}
        for index in range(len(program_name)):
            chip_data[program_name[index]] = []
            print()
            assert len(device[index]) == len(addr[index])
            for d in range(len(device[index])):
                device_ = device[index][d]
                addr_ = addr[index][d]
                device_name = device_.split('.')[0]
                assert device_name in self.rpc_api.keys()
                if 'rram-144k' in device_:
                    array_idx = int(device_.split(':')[-1])
                    api = self.rpc_api[device_name]
                    data_ = api.call('c200_read', array_idx, addr_)
                    chip_data[program_name[index]].append(np.array(data_))
                else:
                    raise ValueError(f'暂不支持设备: {device_name} !!!')
        return chip_data

    def run_ir(self, ir_, inputs, weights=None, log_info=False, *, outputs=None, callback=None, **kwargs):
        ir = copy.deepcopy(ir_)
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))
        assert isinstance(ir, BaseIR), f'invalid IR type={type(ir)}'
        layers = ir.flatten_layers()
        (inp, oup) = ir.get_io_layers(layers)
        (inl, oul) = (layers[inp], layers[oup])
        if isinstance(inputs, dict):
            data = {k: tuple(v) if isinstance(v, (tuple, list)) else (v,) for (k, v) in inputs.items() if v is not None}
        elif isinstance(inputs, (tuple, list)):
            assert len(inputs) == len(inl.inputs)
            data = {inp: tuple(inputs)}
        else:
            data = {inp: (inputs,)}
        for (k, v) in data.items():
            assert k in layers, f'invalid input name {k!r}'
            assert isinstance(v, (tuple, list)), f'invalid inputs type {type(v)}'
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
        if callback is not None:
            assert isinstance(callback, Callable), f'invalid callback {type(callback)}'
        self.layer_calc_time = {}
        for (name, layer) in layers.items():
            if log_info:
                print()
            if name in data:
                continue
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                data[name] = [np.array(layer.op.value)]
                continue
            if any((dd.parse_ref()[0] not in data for dd in layer.inputs)):
                continue
            if name == oup:
                data[name] = []
                for o in layer.inputs:
                    name_ = o.parse_ref()[0]
                    assert name_ in data.keys()
                    data[name].append(data[name_])
                break
            IsReuseLayer = False
            if layer.type == 'reuse':
                reuse_layer_name = name
                name = layer.layer
                reuse_layer = ir.layers[name]
                reuse_layer.inputs = layer.inputs
                layer = reuse_layer
                IsReuseLayer = True
            x = []
            for dd in layer.inputs:
                (nm, idx) = dd.parse_ref()
                x.append(data[nm][0 if idx is None else idx])
            ats = layer.op.get_attrs()
            for ats_n in ['with_batch', 'channel_pos']:
                if ats_n in ats.keys():
                    ats.pop(ats_n)
            wts = dict()
            device_info = dict()
            layer_info = dict()
            if layer.c200_mapping_info != None:
                if layer.op.bias:
                    bias_name = f'{name}.bias'
                    if bias_name not in weights.keys():
                        warnings.warn(f'{bias_name} 不在 weights 中, {name} 没有加载bias !!!')
                    wts['bias'] = weights.get(bias_name)
                device_info.update(dict(device_info=ir.devices))
                layer_info.update(dict(layer_info=ir.layers[name]))
            elif layer.op.op_id in ['conv2d', 'matmul', 'fc', 'linear', 'conv_transpose2d']:
                for k in layer.op.weights:
                    wn = f'{name}.{k}'
                    if k not in layer.op.optional_weights:
                        assert wn in weights, f'missing weight {wn}'
                    wts[k] = weights.get(wn)
            elif layer.op.op_id in ['batch_norm2d']:
                for k in layer.op.weights:
                    ats[k] = np.array(ats[k])
            if layer.c200_mapping_info == None and layer.c200_calc_info != None:
                layer_info.update(dict(layer_info=ir.layers[name]))
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts, attrs=ats, outputs=None, **kwargs)
            time1 = time.time()
            if layer.op.op_id in ['conv2d', 'matmul', 'fc', 'linear', 'conv_transpose2d'] and layer.c200_mapping_info != None:
                y = self.run_layer(layer, *x, **wts, **ats, **device_info, **layer_info, **self.rpc_api)
            else:
                y = self.run_layer(layer, *x, **wts, **ats, **device_info, **layer_info)
            time2 = time.time()
            self.layer_calc_time[name] = round(time2 - time1, 4)
            if not isinstance(y, (tuple, list)):
                y = (y,)
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts, attrs=ats, outputs=y, **kwargs)
            if IsReuseLayer:
                data[reuse_layer_name] = tuple(y)
            else:
                data[name] = tuple(y)
            if ons is not None and all((k in data for k in ons)):
                break
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
                (nm, idx) = dd.parse_ref()
                res.append(data[nm][0 if idx is None else idx])
            if len(res) == 1:
                res = res[0]
        return res

    def run_op(self, op_id, *args, **kwargs):
        fn = getattr(self, f'fn_{op_id}', None)
        assert isinstance(fn, Callable), f'fn_{op_id} is not a function'
        return fn(*args, **kwargs)

    def run_layer(self, layer, *args, **kwargs):
        return self.run_op(layer.op.op_id, *args, **kwargs)

    def fn_matmul(self, x, **kwargs):
        
        if 'device_info' not in kwargs.keys():
            if not self.quant_all:
                in_ = x.data
                in_scale = x.scale
            else:
                in_d = CNT.scale_recover(x)
                (in_, in_scale) = CNT.to_cimtensor_quant(data=in_d.data, half_level=self.half_level, method=self.chip_quant_method, thr=self.thr, multi_batch=self.multi_batch).items
                in_ = in_ / in_scale
                in_scale = np.ones_like(in_scale)
            weight_scale = 1
            if 'layer_info' in kwargs.keys():
                weight_scale = kwargs['layer_info'].c200_calc_info.weight_scale
                kwargs.pop('layer_info')
            kwargs['weight'] = kwargs['weight'] / weight_scale
            output = super().fn_fc(in_, **kwargs)
            if self.clamp_cpu != None:
                assert self.clamp_cpu > 0
                output = np.clip(output, -(self.clamp_cpu + 1), self.clamp_cpu)
            return CNT(data=output, scale=in_scale)
        else:
            if not self.quant_all:
                in_ = x.data
                in_scale = x.scale
            else:
                in_d = CNT.scale_recover(x)
                (in_, in_scale) = CNT.to_cimtensor_quant(data=in_d.data, half_level=self.half_level, method=self.chip_quant_method, thr=self.thr, multi_batch=self.multi_batch).items
            layer_info = kwargs['layer_info']
            it_time = layer_info.c200_calc_info.it_time
            weight_scale = layer_info.c200_calc_info.weight_scale
            assigned_output_quant_scale = layer_info.c200_calc_info.assigned_output_quant_scale
            output_quant_mode = layer_info.c200_calc_info.output_quant_mode
            reg_shift_mode = layer_info.c200_calc_info.reg_shift_mode
            output_half_level = layer_info.c200_calc_info.output_half_level
            shift_expansion_mode = layer_info.c200_calc_info.shift_expansion_mode
            out_channel = layer_info.weights['weight'].shape[0]
            n_scale = layer_info.c200_calc_info.noise_scale
            adc_clamp = layer_info.c200_calc_info.adc_clamp
            ADC_LUT = layer_info.c200_calc_info.ADC_LUT
            adc_quant = layer_info.c200_calc_info.adc_quant
            fit_k = layer_info.c200_calc_info.fit_k
            fit_bias = layer_info.c200_calc_info.fit_b
            runtime = layer_info.c200_mapping_info.runtime
            device_info = kwargs['device_info']
            basic_device_info = device_info[list(device_info.keys())[0]]
            dac_bits = basic_device_info.profile.in_bits
            adc_bits = basic_device_info.profile.out_bits
            signed = basic_device_info.profile.signed
            if signed:
                dac_bits = dac_bits - 1
                adc_bits = adc_bits - 1
            weight_addr_list = []
            row_start_addr_record = {}
            col_start_addr_record = {}
            for (k, v) in layer_info.c200_mapping_info.mappings.items():
                split_config = {}
                (r_index, h_index, w_index) = v.index
                if h_index in row_start_addr_record.keys():
                    assert v.address[2] == row_start_addr_record[h_index]
                else:
                    row_start_addr_record[h_index] = v.address[2]
                if w_index in col_start_addr_record.keys():
                    assert v.address[3] == col_start_addr_record[w_index]
                else:
                    col_start_addr_record[w_index] = v.address[3]
                split_config['index'] = v.index
                split_config['array_idx'] = v.device
                split_config['weight_addr'] = v.address
                device_name = v.device.split('.')[0]
                if device_name in kwargs.keys():
                    split_config[v.device] = kwargs[device_name]
                weight_addr_list.append(split_config)
            mapped_array_data_ = {}
            for sc in weight_addr_list:
                sum_row = 0
                for i in range(sc['index'][1] + 1):
                    sum_col = 0
                    for j in range(sc['index'][2] + 1):
                        if j == 0:
                            sum_col = 0
                        else:
                            sum_col += col_start_addr_record[j]
                    if i == 0:
                        sum_row = 0
                    else:
                        sum_row += row_start_addr_record[i - 1]
                sc['array_output_col_start'] = sum_col
                sc['array_input_row_start'] = sum_row
            col_repeat_num = layer_info.c200_mapping_info.col_repeat_num
            row_repeat_num = layer_info.c200_mapping_info.row_repeat_num
            bias_digital = False
            bias = None
            if 'bias' in kwargs.keys():
                if isinstance(kwargs['bias'], list) and (kwargs['bias'] != None).all():
                    bias_digital = True
                    bias = kwargs['bias']
            if len(in_.shape) == 1:
                in_ = np.expand_dims(in_, axis=0)
            if self.multi_batch:
                in_ = np.expand_dims(in_, axis=2)
                in_scale = np.expand_dims(in_scale, axis=1)
                in_scale = np.expand_dims(in_scale, axis=2)
            else:
                in_ = in_.transpose(1, 0)
            (output, output_scale, maxp, minp, shift_scale) = calc_mvm(weight_addr_list, in_, in_scale, [row_repeat_num, col_repeat_num], out_channel, output_half_level, weight_scale, dac_bits, adc_bits, self.device_adc_scale_data, reg_shift_mode, shift_expansion_mode, self.mapped_array_data, assigned_output_quant_scale, output_quant_mode, it_time, bias_digital=bias_digital, bias=bias, n_scale=n_scale, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=self.macro_method, runtime=runtime, multi_batch=self.multi_batch, fit_k=fit_k, fit_bias=fit_bias)
            if self.multi_batch:
                output = np.squeeze(output, axis=1)
                output_scale = np.squeeze(output_scale, axis=1)
                if output_quant_mode == 1:
                    shift_scale = np.squeeze(shift_scale, axis=1)
                batch = in_.shape[1]
                maxp = np.array(maxp).sum()
                minp = np.array(minp).sum()
                maxp = maxp / batch
                minp = minp / batch
            return CNT.to_cimtensor(data=output, scale=output_scale, max_percent=maxp, min_percent=minp, shift_scale=shift_scale)
    fn_linear = fn_matmul
    fn_fc = fn_matmul

    def fn_conv2d(self, x, **kwargs):
        
        if 'device_info' not in kwargs.keys():
            if not self.quant_all:
                in_ = x.data
                in_scale = x.scale
            else:
                in_d = CNT.scale_recover(x)
                (in_, in_scale) = CNT.to_cimtensor_quant(data=in_d.data, half_level=self.half_level, method=self.chip_quant_method, thr=self.thr, multi_batch=self.multi_batch).items
                in_ = in_ / in_scale
                in_scale = np.ones_like(in_scale)
            weight_scale = 1
            if 'layer_info' in kwargs.keys():
                weight_scale = kwargs['layer_info'].c200_calc_info.weight_scale
                kwargs.pop('layer_info')
            kwargs['weight'] = kwargs['weight'] / weight_scale
            output = super().fn_conv2d(in_, **kwargs)
            if self.clamp_cpu != None:
                assert self.clamp_cpu > 0
                output = np.clip(output, -(self.clamp_cpu + 1), self.clamp_cpu)
            return CNT(data=output, scale=in_scale)
        else:
            if not self.quant_all:
                in_ = x.data
                in_scale = x.scale
            else:
                in_d = CNT.scale_recover(x)
                (in_, in_scale) = CNT.to_cimtensor_quant(data=in_d.data, half_level=self.half_level, method=self.chip_quant_method, thr=self.thr, multi_batch=self.multi_batch).items
            layer_info = kwargs['layer_info']
            it_time = layer_info.c200_calc_info.it_time
            weight_scale = layer_info.c200_calc_info.weight_scale
            assigned_output_quant_scale = layer_info.c200_calc_info.assigned_output_quant_scale
            output_quant_mode = layer_info.c200_calc_info.output_quant_mode
            reg_shift_mode = layer_info.c200_calc_info.reg_shift_mode
            output_half_level = layer_info.c200_calc_info.output_half_level
            shift_expansion_mode = layer_info.c200_calc_info.shift_expansion_mode
            out_channel = layer_info.op.out_channel
            n_scale = layer_info.c200_calc_info.noise_scale
            adc_clamp = layer_info.c200_calc_info.adc_clamp
            ADC_LUT = layer_info.c200_calc_info.ADC_LUT
            adc_quant = layer_info.c200_calc_info.adc_quant
            fit_k = layer_info.c200_calc_info.fit_k
            fit_bias = layer_info.c200_calc_info.fit_b
            runtime = layer_info.c200_mapping_info.runtime
            device_info = kwargs['device_info']
            basic_device_info = device_info[list(device_info.keys())[0]]
            dac_bits = basic_device_info.profile.in_bits
            adc_bits = basic_device_info.profile.out_bits
            signed = basic_device_info.profile.signed
            if signed:
                dac_bits = dac_bits - 1
                adc_bits = adc_bits - 1
            weight_addr_list = []
            row_start_addr_record = {}
            col_start_addr_record = {}
            for (k, v) in layer_info.c200_mapping_info.mappings.items():
                split_config = {}
                (r_index, h_index, w_index) = v.index
                if h_index in row_start_addr_record.keys():
                    assert v.address[2] == row_start_addr_record[h_index]
                else:
                    row_start_addr_record[h_index] = v.address[2]
                if w_index in col_start_addr_record.keys():
                    assert v.address[3] == col_start_addr_record[w_index]
                else:
                    col_start_addr_record[w_index] = v.address[3]
                split_config['index'] = v.index
                split_config['array_idx'] = v.device
                split_config['weight_addr'] = v.address
                device_name = v.device.split('.')[0]
                if device_name in kwargs.keys():
                    split_config[v.device] = kwargs[device_name]
                weight_addr_list.append(split_config)
            for sc in weight_addr_list:
                sum_row = 0
                for i in range(sc['index'][1] + 1):
                    sum_col = 0
                    for j in range(sc['index'][2] + 1):
                        if j == 0:
                            sum_col = 0
                        else:
                            sum_col += col_start_addr_record[j]
                    if i == 0:
                        sum_row = 0
                    else:
                        sum_row += row_start_addr_record[i - 1]
                sc['array_output_col_start'] = sum_col
                sc['array_input_row_start'] = sum_row
            col_repeat_num = layer_info.c200_mapping_info.col_repeat_num
            row_repeat_num = layer_info.c200_mapping_info.row_repeat_num
            if len(in_.shape) == 3:
                if self.weight_format == 'CHW':
                    (channel, input_rows, input_cols) = in_.shape
                elif self.weight_format == 'HWC':
                    (input_rows, input_cols, channel) = in_.shape
                else:
                    raise ValueError("暂不支持除['CHW','HWC']以外的weight format！！！")
            elif len(in_.shape) == 4:
                if self.weight_format == 'CHW':
                    (batch, channel, input_rows, input_cols) = in_.shape
                elif self.weight_format == 'HWC':
                    (batch, input_rows, input_cols, channel) = in_.shape
                else:
                    raise ValueError("暂不支持除['CHW','HWC']以外的weight format！！！")
            else:
                raise ValueError(f'{in_.shape}维度不支持!!!')
            padding = layer_info.op.padding
            kernel_size = layer_info.op.kernel
            if 'stride' in kwargs.keys():
                stride = kwargs['stride']
            else:
                stride = layer_info.op.stride
            out_feature_size_rows = int((input_rows + padding + padding - kernel_size) / stride + 1)
            out_feature_size_cols = int((input_cols + padding + padding - kernel_size) / stride + 1)
            if self.weight_format == 'CHW':
                array_input = feature_map_to_input(in_, stride=stride, kernel_size=kernel_size, padding=padding, multi_batch=self.multi_batch)
            elif self.weight_format == 'HWC':
                array_input = feature_map_to_input_np_HWC(in_, stride=stride, kernel_size=kernel_size, padding=padding, multi_batch=self.multi_batch)
            else:
                raise ValueError(f'暂不支持权重格式{self.weight_format}!!!')
            bias_digital = False
            bias = None
            if 'bias' in kwargs.keys() and (kwargs['bias'] != None).all():
                bias_digital = True
                bias = kwargs['bias']
            (output_, output_scale, maxp, minp, shift_scale) = calc_mvm(weight_addr_list, array_input, in_scale, [row_repeat_num, col_repeat_num], out_channel, output_half_level, weight_scale, dac_bits, adc_bits, self.device_adc_scale_data, reg_shift_mode, shift_expansion_mode, self.mapped_array_data, assigned_output_quant_scale, output_quant_mode, it_time, bias_digital=bias_digital, bias=bias, n_scale=n_scale, adc_clamp=adc_clamp, ADC_LUT=ADC_LUT, adc_quant=adc_quant, macro_method=self.macro_method, runtime=runtime, multi_batch=self.multi_batch, fit_k=fit_k, fit_bias=fit_bias)
            output = output_to_feature_map(output_, out_feature_size_rows, out_feature_size_cols, multi_batch=self.multi_batch)
            if self.multi_batch:
                output_scale = np.expand_dims(output_scale, axis=1)
                if output_quant_mode == 1:
                    shift_scale = np.expand_dims(shift_scale, axis=1)
                maxp = np.array(maxp).sum()
                minp = np.array(minp).sum()
                maxp = maxp / batch
                minp = minp / batch
            if self.weight_format == 'HWC':
                if len(output.shape) == 3:
                    output = output.transpose(1, 2, 0)
                elif len(output.shape) == 4:
                    output = output.transpose(0, 2, 3, 1)
                else:
                    raise ValueError(f'输出维度{output.shape}错误!!!')
            return CNT.to_cimtensor(data=output, scale=output_scale, max_percent=maxp, min_percent=minp, shift_scale=shift_scale)

    def fn_conv_transpose2d(self, x, **kwargs):
        
        if 'device_info' not in kwargs.keys():
            if not self.quant_all:
                in_ = x.data
                in_scale = x.scale
            else:
                in_d = CNT.scale_recover(x)
                (in_, in_scale) = CNT.to_cimtensor_quant(data=in_d.data, half_level=self.half_level, method=self.chip_quant_method, thr=self.thr, multi_batch=self.multi_batch).items
                in_ = in_ / in_scale
                in_scale = np.ones_like(in_scale)
            output = super().fn_conv_transpose2d(in_, **kwargs)
            if self.clamp_cpu != None:
                assert self.clamp_cpu > 0
                output = np.clip(output, -(self.clamp_cpu + 1), self.clamp_cpu)
            return CNT(data=output, scale=in_scale)
        else:
            in_ = x.data
            in_scale = x.scale
            group = kwargs['group']
            stride = kwargs['stride']
            padding = kwargs['padding']
            dilation = kwargs['dilation']
            output_padding = kwargs['output_padding']
            auto_pad = kwargs['auto_pad']
            weight_shape = kwargs['layer_info'].weights['weight'].shape
            (kci, co, *kernel) = weight_shape
            ndim = 2
            if self.channel_last:
                (ba, *xd, ci) = in_.shape
            else:
                (ba, ci, *xd) = in_.shape
            assert ci == kci * group, f'invalid input shape {in_.shape} with kernel {weight_shape}'
            (os, (k, s, p, d, dk, dp, di)) = conv_t_shapes(xd, kernel, stride, padding, dilation, output_padding, auto_pad)
            if di != xd:
                xp = np.zeros(self.to_axes(ba, ci, di), dtype=in_.dtype)
                oi = tuple((slice(dp[i], xd[i] * s[i] + dp[i], s[i]) for i in range(ndim)))
                xp[self.to_slices(oi)] = in_
                in_ = xp
            x = CNT.to_cimtensor(data=in_, scale=in_scale)
            if stride != 1:
                kwargs['stride'] = 1
            return self.fn_conv2d(x, **kwargs)

    def rand(self, shape):
        data = self._be.random.randint(low=0, high=self.half_level, size=shape)
        return CNT.to_cimtensor(data=data, multi_batch=self.multi_batch)

    def fn_abs(self, x):
        data = CNT.scale_recover(x).data
        re = super().fn_abs(data)
        if self.qaunt_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_acos(self, x):
        data = CNT.scale_recover(x).data
        re = super().fn_acos(data)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_concat(self, *x, axis):
        data = []
        for cnt in x:
            if isinstance(cnt, CNT):
                data.append(CNT.scale_recover(cnt).data)
                batch = data[0].shape[0]
            elif isinstance(cnt, np.ndarray):
                if len(cnt.shape) == 1:
                    cnt_list = [list(cnt)] * batch
                    cnt = np.array(cnt_list)
                data.append(cnt)
            else:
                raise ValueError(f'暂不支持 {type(x)} !!!')
        re = self._be.concatenate(data, axis=axis)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_reshape(self, x, *, shape):
        if self.multi_batch and shape[0] == 1:
            shape = list(shape)
            shape[0] = shape[0] * x.data.shape[0]
            shape = tuple(shape)
            scale_shape = x.scale.shape
            while len(scale_shape) < len(shape):
                x.scale = np.expand_dims(x.scale, axis=len(scale_shape))
                scale_shape = x.scale.shape
            scale_shape = x.scale.shape
        return CNT(data=self._be.reshape(x.data, shape), scale=self._be.reshape(x.scale, scale_shape))

    def fn_flatten(self, x, *, start_dim):
        if self.multi_batch:
            data = self._be.reshape(x.data, (*x.data.shape[:start_dim], -1))
            scale = self._be.reshape(x.scale, (*x.scale.shape[:start_dim], -1))
            return CNT(data=data, scale=scale)
        data = self._be.reshape(x.data, (*x.data.shape[:start_dim - 1], -1))
        scale = x.scale
        return CNT(data=data, scale=scale)

    def fn_transpose(self, x, *, perm):
        if self.multi_batch:
            return CNT(data=self._be.transpose(x.data, perm), scale=self._be.transpose(x.scale, perm))
        return CNT(data=self._be.transpose(x.data, perm), scale=x.scale)

    def fn_relu(self, x):
        if isinstance(x, CNT):
            re = self._be.clip(CNT.scale_recover(x).data, 0, None)
        elif isinstance(x, np.ndarray):
            re = self._be.clip(x, 0, None)
        else:
            raise ValueError(f'暂不支持数据类型 {type(x)} !!!')
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_leaky_relu(self, x, *, alpha):
        data = CNT.scale_recover(x).data
        re = self._be.where(data < 0, data * alpha, data)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_prelu(self, x, *, slope):
        data = CNT.scale_recover(x).data
        re = self._be.where(data < 0, data * slope, data)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_selu(self, x, *, alpha, gamma):
        data = CNT.scale_recover(x).data
        re = self._be.where(data < 0, (self._be.exp(data) - 1) * alpha * gamma, data * gamma)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_celu(self, x, *, alpha):
        data = CNT.scale_recover(x).data
        re = self._be.where(data < 0, (self._be.exp(data / alpha) - 1) * alpha, data)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_elu(self, x, *, alpha):
        data = CNT.scale_recover(x).data
        re = self._be.where(data < 0, (self._be.exp(data) - 1) * alpha, data)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_softmax(self, x, *, axis):
        data = CNT.scale_recover(x).data
        y = self._be.exp(data)
        s = self._be.sum(y, axis=axis, keepdims=True)
        re = y / s
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_log_softmax(self, x, *, axis):
        data = CNT.scale_recover(x).data
        y = self._be.exp(data)
        s = self._be.sum(y, axis=axis, keepdims=True)
        re = self._be.log(y / s)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_sigmoid(self, x):
        data = CNT.scale_recover(x).data
        re = 1 / (1 + self._be.exp(-data))
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_hard_sigmoid(self, x, *, alpha, beta):
        data = CNT.scale_recover(x).data
        re = self._be.clip(data * alpha + beta, 0, 1)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_softplus(self, x):
        data = CNT.scale_recover(x).data
        re = self._be.log(self._be.exp(data) + 1)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_softsign(self, x):
        data = CNT.scale_recover(x).data
        re = x / (1 + self._be.abs(data))
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_avg_pool1d(self, x, **kwargs):
        data = CNT.scale_recover(x).data
        re = self._pool(self._be.mean, 1, data, **kwargs)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_avg_pool2d(self, x, **kwargs):
        data = CNT.scale_recover(x).data
        re = self._pool(self._be.mean, 2, data, **kwargs)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_avg_pool3d(self, x, **kwargs):
        data = CNT.scale_recover(x).data
        re = self._pool(self._be.mean, 3, data, **kwargs)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)
    fn_avgpool1d = fn_avg_pool1d
    fn_avgpool2d = fn_avg_pool2d
    fn_avgpool3d = fn_avg_pool3d

    def fn_max_pool1d(self, x, **kwargs):
        data = CNT.scale_recover(x).data
        re = self._pool(self._be.amax, 1, data, **kwargs)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_max_pool2d(self, x, **kwargs):
        data = CNT.scale_recover(x).data
        re = self._pool(self._be.amax, 2, data, **kwargs)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_max_pool3d(self, x, **kwargs):
        data = CNT.scale_recover(x).data
        re = self._pool(self._be.amax, 3, data, **kwargs)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)
    fn_maxpool1d = fn_max_pool1d
    fn_maxpool2d = fn_max_pool2d
    fn_maxpool3d = fn_max_pool3d

    def _global_pool(self, x, *, ndim, func):
        axis = tuple(range(len(x.shape) - ndim, len(x.shape)))
        return func(x, axis=axis, keepdims=True)

    def fn_global_avg_pool1d(self, x):
        data = CNT.scale_recover(x).data
        re = self._global_pool(data, ndim=1, func=self._be.mean)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_global_avg_pool2d(self, x):
        data = CNT.scale_recover(x).data
        re = self._global_pool(data, ndim=2, func=self._be.mean)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_global_avg_pool3d(self, x):
        data = CNT.scale_recover(x).data
        re = self._global_pool(data, ndim=3, func=self._be.mean)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_global_max_pool1d(self, x):
        data = CNT.scale_recover(x).data
        re = self._global_pool(data, ndim=1, func=self._be.amax)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_global_max_pool2d(self, x):
        data = CNT.scale_recover(x).data
        re = self._global_pool(data, ndim=2, func=self._be.amax)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_global_max_pool3d(self, x):
        data = CNT.scale_recover(x).data
        re = self._global_pool(data, ndim=3, func=self._be.amax)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_pow(self, x, y):
        data = CNT.scale_recover(x).data
        re = self._be.power(data, y)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_reducemean(self, x, **kwargs):
        data = CNT.scale_recover(x).data
        axes = kwargs['axes']
        keepdims = kwargs['keepdims']
        re = self._be.mean(data, axis=axes)
        if keepdims:
            re = self._be.expand_dims(re, axis=axes)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_sqrt(self, x):
        data = CNT.scale_recover(x).data
        re = self._be.sqrt(data)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_add(self, x, y):
        data = CNT.scale_recover(x).data
        if str(type(y)) == "<class 'numpy.ndarray'>":
            re = data + y
        else:
            data_y = CNT.scale_recover(y).data
            re = data + data_y
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_sub(self, x, y):
        data = CNT.scale_recover(x).data
        if str(type(y)) == "<class 'numpy.ndarray'>":
            re = data - y
        else:
            data_y = CNT.scale_recover(y).data
            re = data - data_y
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_mul(self, x, y):
        if str(type(x)) == "<class 'numpy.ndarray'>":
            data = x
        else:
            data = CNT.scale_recover(x).data
        if str(type(y)) == "<class 'numpy.ndarray'>":
            re = data * y
        else:
            data_y = CNT.scale_recover(y).data
            re = data * data_y
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_div(self, x, y):
        data = CNT.scale_recover(x).data
        if str(type(y)) == "<class 'numpy.ndarray'>":
            re = data / (y + 10 ** (-6))
        else:
            data_y = CNT.scale_recover(y).data
            re = data / (data_y + 10 ** (-6))
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_resize(self, x, *, size, scale, mode):
        if scale != None:
            if len(scale) == 4:
                scale = scale[-2:]
        data = CNT.scale_recover(x).data
        re = super().fn_resize(data, size=size, scale=scale, mode=mode)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_tanh(self, x):
        data = CNT.scale_recover(x).data
        re = np.tanh(data)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=re, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=re, multi_batch=self.multi_batch)

    def fn_split(self, x, *, axis, split):
        data = CNT.scale_recover(x).data
        re = super().fn_split(data, axis=axis, split=split)
        re_ = []
        for d in re:
            if len(d.shape) == 3:
                d = d.squeeze()
            if self.quant_all:
                re_.append(CNT.to_cimtensor_quant(data=d, half_level=self.half_level, multi_batch=self.multi_batch))
            else:
                re_.append(CNT.to_cimtensor(data=d, multi_batch=self.multi_batch))
        return re_

    def fn_batch_norm(self, x, *, scale, bias, input_mean, input_var, epsilon):
        x = CNT.scale_recover(x).data
        if self.multi_batch:
            rank = len(x.shape) - 1
        else:
            rank = len(x.shape)
        input_mean = self._broadcast(input_mean, rank)
        input_var = self._broadcast(input_var, rank)
        y = (x - input_mean) / self._be.sqrt(input_var + epsilon)
        if scale is not None:
            y *= self._broadcast(scale, rank)
        if bias is not None:
            y += self._broadcast(bias, rank)
        if self.quant_all:
            return CNT.to_cimtensor_quant(data=y, half_level=self.half_level, multi_batch=self.multi_batch)
        else:
            return CNT.to_cimtensor(data=y, multi_batch=self.multi_batch)
    fn_batch_norm1d = fn_batch_norm
    fn_batch_norm2d = fn_batch_norm
    fn_batch_norm3d = fn_batch_norm
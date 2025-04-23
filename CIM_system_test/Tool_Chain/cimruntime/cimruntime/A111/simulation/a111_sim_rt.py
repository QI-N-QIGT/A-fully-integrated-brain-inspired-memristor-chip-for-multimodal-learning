# from .utils import *
from e100_irmapper.device.a111 import MappedLayer # noqa
from typing import Callable
from e100_irtool.core import BaseIR
from e100_irtool.tools import flatten_layers  # noqa
from ...torch import TorchRuntime
import math
from e100_irtool.runtime.utils import auto_pad_pool, concat_axis
import time
import torch
from e100_irtool.core.type_util import to_int_tuple
from ...cimtensor import CIMNumpyTensor as CNT
from .quant_layer import *
from .quant_util import get_act_quantizer, get_weight_quantizer

time_record = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))

class A111TorchRT(TorchRuntime):

    name = 'a111_cim_torch'
    channel_last = False
    tensor_is_readonly = False

    def __init__(self, device='cpu'):
        
        self._be = torch
        self._fn = torch.nn.functional
        self.tensor_type = torch.Tensor
        self.device = device
        
    def run_ir(self, ir, inputs, weights=None, *, outputs=None, callback=None, **kwargs):
        assert isinstance(ir, BaseIR), f'invalid IR type={type(ir)}'
        layers = ir.flatten_layers()
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
        
        batch_size = 1
        for k, v in data.items():
            assert k in layers, f'invalid input name {k!r}'
            assert isinstance(v, (tuple, list)), \
                f'invalid inputs type {type(v)}'
            # 初始化batch
            batch_size = v[0][0].shape[0]
                
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
   
        for name, layer in layers.items():
            # print(f'当前层 : {name} ===>')
            # print(data.keys())
            if name in data:
                continue    # layer is done
            
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                constant_data = torch.Tensor(layer.op.value * batch_size)
                # 数据格式统一
                data[name] = ((constant_data.reshape(batch_size, -1), 1), )
                continue
            
            if any(dd.parse_ref()[0] not in data for dd in layer.inputs):
                continue    # layer can't be run
            if name == oup:
                break       # output layer
            
            IsReuseLayer = False
            # resue layer
            if layer.type == 'reuse':
                reuse_layer_name = name
                name = layer.layer
                # 替换reuse层的输入
                reuse_layer = ir.layers[name].clone()
                reuse_layer.inputs = layer.inputs
                layer = reuse_layer
                IsReuseLayer = True
                # print(reuse_layer_name)
                
            x = []
            for dd in layer.inputs:
                nm, idx = dd.parse_ref()
                # if len(data[nm]) <= 2:
                #     x.append(data[nm][0])
                # else:
                #     x.append(data[nm][0 if idx is None else idx])
                # if len(data[nm]) == 1:
                #     x.append(data[nm][0])
                # else:
                #     x.append(data[nm])
                x.append(data[nm][0 if idx is None else idx])
            
            ats = layer.op.get_attrs()
            for ats_n in ['with_batch', 'channel_pos']:
                if ats_n in ats.keys():
                    ats.pop(ats_n)
            
            wts = dict()

            if layer.op.op_id in ['conv2d','matmul','fc','linear','conv_transpose2d']:
                for k in layer.op.weights:
                    wn = f'{name}.{k}'
                    if k not in layer.op.optional_weights:
                        assert wn in weights, f'missing weight {wn}'
                    wts[k] = weights.get(wn)
                
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=None, **kwargs)            
            
            # 判断当前层是否 A111 仿真计算
            is_simulation = False
            
            calc_info_dict = {}
            kwargs_ = {}
            if layer.op.op_id in ['conv2d', 'conv_transpose2d']:
                if name in kwargs.keys():
                    # calc_dict = kwargs[name]
                    kwargs_ = kwargs[name]
                    # if name == 'Conv_19':
                    #     print(kwargs_)
                    #     input()
                    is_simulation = True
                    
                    # d['A111_process'] = self.A111_process
                    # d['input_method'] = self.input_method
                    # d['input_shift'] = self.input_shift
                    # d['input_clip_bit'] = self.input_clip_bit
                    # d['using_hard_scale'] = self.using_hard_scale
                    # d['hard_scale_method'] = self.hard_scale_method
                    # d['in_quant_low_bit'] = self.in_quant_low_bit
                    # d['input_v'] = self.input_val
                    # d['mini_voltage'] = self.mini_v
                    # d['G_max'] = self.G_max
                    # d['ADC_bit'] = self.ADC_bit
                    # d['ADC_scale'] = self.ADC_scale
                    # d['number_of_XB'] = self.number_of_XB
                    # d['XB_lines'] = self.XB_lines
                    # d['shift_num'] = self.shift_num.cpu().item()
                    # d['Gain_Error'] = self.Gain_Error.cpu().numpy()
                    # d['offset_vector'] = self.offset_vector.cpu().numpy()
                    # d['int_flag'] = self.int_flag
                    # d['output_clip'] = self.output_clip
                    # d['GE_wise'] = self.GE_wise
                    # d['GE_mean'] = self.GE_mean
                    # d['GE_std'] = self.GE_std
                    # d['OF_wise'] = self.OF_wise
                    # d['OF_mean'] = self.OF_mean
                    # d['OF_std'] = self.OF_std
                    # d['hard_out_noise_method'] = self.hard_out_noise_method
                    
                    calc_info_dict['in_channels'] = layer.op.in_channel
                    calc_info_dict['out_channels'] = layer.op.out_channel
                    calc_info_dict['kernel_size'] = layer.op.kernel
                    calc_info_dict['w_quantizer'] = get_weight_quantizer(kwargs_['w_args'])
                    calc_info_dict['a_quantizer'] = get_act_quantizer(kwargs_['act_args'])
                    calc_info_dict['a_out_quantizer'] = get_act_quantizer(kwargs_['act_out_args'])
                    
                    # hard parameters
                    calc_info_dict['A111_process'] = kwargs_['hard_params']['A111_process']
                    calc_info_dict['input_method'] = kwargs_['hard_params']['input_method']
                    calc_info_dict['input_shift'] = kwargs_['hard_params']['input_shift']
                    calc_info_dict['input_clip_bit'] = kwargs_['hard_params']['input_clip_bit']
                    calc_info_dict['using_hard_scale'] = kwargs_['hard_params']['using_hard_scale']
                    calc_info_dict['hard_scale_method'] = kwargs_['hard_params']['hard_scale_method']
                    calc_info_dict['in_quant_low_bit'] = kwargs_['hard_params']['in_quant_low_bit']
                    calc_info_dict['input_v'] = kwargs_['hard_params']['input_v']
                    calc_info_dict['mini_voltage'] = kwargs_['hard_params']['mini_voltage']
                    calc_info_dict['G_max'] = kwargs_['hard_params']['G_max']
                    calc_info_dict['ADC_bit'] = kwargs_['hard_params']['ADC_bit']
                    calc_info_dict['ADC_scale'] = kwargs_['hard_params']['ADC_scale']
                    calc_info_dict['number_of_XB'] = kwargs_['hard_params']['number_of_XB']
                    calc_info_dict['XB_lines'] = kwargs_['hard_params']['XB_lines']
                    calc_info_dict['shift_num'] = kwargs_['hard_params']['shift_num']
                    calc_info_dict['Gain_Error'] = torch.from_numpy(kwargs_['hard_params']['Gain_Error']).to(self.device)
                    calc_info_dict['offset_vector'] = torch.from_numpy(kwargs_['hard_params']['offset_vector']).to(self.device)
                    calc_info_dict['int_flag'] = kwargs_['hard_params']['int_flag']
                    calc_info_dict['output_clip'] = kwargs_['hard_params']['output_clip']
                    calc_info_dict['GE_wise'] = kwargs_['hard_params']['GE_wise']
                    calc_info_dict['GE_mean'] = kwargs_['hard_params']['GE_mean']
                    calc_info_dict['GE_std'] = kwargs_['hard_params']['GE_std']
                    calc_info_dict['OF_wise'] = kwargs_['hard_params']['OF_wise']
                    calc_info_dict['OF_mean'] = kwargs_['hard_params']['OF_mean']
                    calc_info_dict['OF_std'] = kwargs_['hard_params']['OF_std']
                    calc_info_dict['hard_out_noise_method'] = kwargs_['hard_params']['hard_out_noise_method']
                    if 'hard_out_noise' in kwargs_['hard_params'].keys():
                        calc_info_dict['hard_out_noise'] = kwargs_['hard_params']['hard_out_noise']
                    else:
                        calc_info_dict['hard_out_noise'] = 0
                
                # 还原权重 为浮点数
                if 'weight_scale' in kwargs_.keys() and kwargs_['weight_scale'] != None:
                    wts['weight'] = wts['weight'] * kwargs_['weight_scale']
                
            elif layer.op.op_id in ['matmul','fc','linear']:
                
                if name in kwargs.keys():
                    kwargs_ = kwargs[name]
                    is_simulation = True
                    calc_info_dict['in_features'] = layer.op.in_channel
                    calc_info_dict['out_features'] = layer.op.out_channel
                    calc_info_dict['w_quantizer'] = get_weight_quantizer(kwargs_['w_args'])
                    calc_info_dict['a_quantizer'] = get_act_quantizer(kwargs_['act_args'])
                    calc_info_dict['a_out_quantizer'] = get_act_quantizer(kwargs_['act_out_args'])
                    
                    # calc_info_dict['offset_vector'] = torch.from_numpy(kwargs_['hard_params']['offset_vector']).to(self.device)
                    # calc_info_dict['Gain_Error'] = torch.from_numpy(kwargs_['hard_params']['Gain_Error']).to(self.device)
                    # calc_info_dict['hard_scale_method'] = kwargs_['hard_params']['hard_scale_method']
                    # calc_info_dict['int_flag'] = kwargs_['hard_params']['int_flag']
                    # calc_info_dict['in_quant_low_bit'] = kwargs_['hard_params']['in_quant_low_bit']
                    # calc_info_dict['using_hard_scale'] = kwargs_['hard_params']['using_hard_scale']
                    # calc_info_dict['mini_voltage'] = kwargs_['hard_params']['mini_voltage']
                    # calc_info_dict['G_max'] = kwargs_['hard_params']['G_max']
                    # calc_info_dict['ADC_bit'] = kwargs_['hard_params']['ADC_bit']
                    # calc_info_dict['ADC_scale'] = kwargs_['hard_params']['ADC_scale']
                    # calc_info_dict['input_v'] = kwargs_['hard_params']['input_v']
                    # calc_info_dict['shift_num'] = kwargs_['hard_params']['shift_num']
                    # calc_info_dict['out_scale_test'] = kwargs_['hard_params']['out_scale_test']
                    
                    # hard parameters
                    calc_info_dict['A111_process'] = kwargs_['hard_params']['A111_process']
                    calc_info_dict['input_method'] = kwargs_['hard_params']['input_method']
                    calc_info_dict['input_shift'] = kwargs_['hard_params']['input_shift']
                    calc_info_dict['input_clip_bit'] = kwargs_['hard_params']['input_clip_bit']
                    calc_info_dict['using_hard_scale'] = kwargs_['hard_params']['using_hard_scale']
                    calc_info_dict['hard_scale_method'] = kwargs_['hard_params']['hard_scale_method']
                    calc_info_dict['in_quant_low_bit'] = kwargs_['hard_params']['in_quant_low_bit']
                    calc_info_dict['input_v'] = kwargs_['hard_params']['input_v']
                    calc_info_dict['mini_voltage'] = kwargs_['hard_params']['mini_voltage']
                    calc_info_dict['G_max'] = kwargs_['hard_params']['G_max']
                    calc_info_dict['ADC_bit'] = kwargs_['hard_params']['ADC_bit']
                    calc_info_dict['ADC_scale'] = kwargs_['hard_params']['ADC_scale']
                    calc_info_dict['number_of_XB'] = kwargs_['hard_params']['number_of_XB']
                    calc_info_dict['XB_lines'] = kwargs_['hard_params']['XB_lines']
                    calc_info_dict['shift_num'] = kwargs_['hard_params']['shift_num']
                    calc_info_dict['Gain_Error'] = torch.from_numpy(kwargs_['hard_params']['Gain_Error']).to(self.device)
                    calc_info_dict['offset_vector'] = torch.from_numpy(kwargs_['hard_params']['offset_vector']).to(self.device)
                    calc_info_dict['int_flag'] = kwargs_['hard_params']['int_flag']
                    calc_info_dict['output_clip'] = kwargs_['hard_params']['output_clip']
                    calc_info_dict['GE_wise'] = kwargs_['hard_params']['GE_wise']
                    calc_info_dict['GE_mean'] = kwargs_['hard_params']['GE_mean']
                    calc_info_dict['GE_std'] = kwargs_['hard_params']['GE_std']
                    calc_info_dict['OF_wise'] = kwargs_['hard_params']['OF_wise']
                    calc_info_dict['OF_mean'] = kwargs_['hard_params']['OF_mean']
                    calc_info_dict['OF_std'] = kwargs_['hard_params']['OF_std']
                    calc_info_dict['hard_out_noise_method'] = kwargs_['hard_params']['hard_out_noise_method']
                    # calc_info_dict['hard_out_noise'] = kwargs_['hard_params']['hard_out_noise']
                    if 'hard_out_noise' in kwargs_['hard_params'].keys():
                        calc_info_dict['hard_out_noise'] = kwargs_['hard_params']['hard_out_noise']
                    else:
                        calc_info_dict['hard_out_noise'] = 0
                
                # 还原权重为浮点数
                if 'weight_scale' in kwargs_.keys() and kwargs_['weight_scale'] != None:
                    wts['weight'] = wts['weight'] * kwargs_['weight_scale']

            elif layer.op.op_id in ['add']:
                # assert name in kwargs.keys()
                is_simulation = True
                if name in kwargs.keys():
                    kwargs_ = kwargs[name]['params']
                    calc_info_dict['bit'] = kwargs_['bit']
                    calc_info_dict['thd_neg'] = kwargs_['thd_neg']
                    calc_info_dict['thd_pos'] = kwargs_['thd_pos']
                    calc_info_dict['quant_method'] = kwargs_['quant_method']
                    calc_info_dict['shift'] = kwargs_['shift']
                    if 'clip_before' not in kwargs_.keys():
                        calc_info_dict['clip_before'] = False
                    else:
                        calc_info_dict['clip_before'] = kwargs_['clip_before']
                else:
                    calc_info_dict['quant_method'] = -1
                # calc_info_dict['int_flag'] = kwargs_['int_flag']
                
            elif layer.op.op_id in ['global_avg_pool2d']:
                name_ = 'AdaptiveAvgPool2d_' + name.split('_')[-1]
                if name_ in kwargs.keys():
                    is_simulation = True
                    kwargs_ = kwargs[name_]['params']
                    calc_info_dict['quant_flag'] = kwargs_['quant_flag']
                    calc_info_dict['bit'] = kwargs_['bit']    
                    calc_info_dict['thd_neg'] = kwargs_['thd_neg']
                    calc_info_dict['thd_pos'] = kwargs_['thd_pos']
                    calc_info_dict['set_shift_num'] = kwargs_['set_shift_num']
                    calc_info_dict['shift_num'] = kwargs_['shift_num']
            elif layer.op.op_id in ['avgpool2d']:
                name_ = 'AvgPool2d_' + name.split('_')[-1]
                if name_ in kwargs.keys():
                    is_simulation = True
                    kwargs_ = kwargs[name_]['params']
                    calc_info_dict['quant_flag'] = kwargs_['quant_flag']
                    calc_info_dict['bit'] = kwargs_['bit']
                    calc_info_dict['set_shift_num'] = kwargs_['set_shift_num']
                    calc_info_dict['shift_num'] = kwargs_['shift_num']
                    calc_info_dict['thd_neg'] = kwargs_['thd_neg']
                    calc_info_dict['thd_pos'] = kwargs_['thd_pos']
                    
                    
            else:
                pass
            
            if is_simulation and layer.op.op_id in ['conv2d','matmul','fc','linear','conv_transpose2d',
                                                    'add','max_pool2d','global_avg_pool2d', 'avgpool2d']: 
                y = self.run_layer(layer, *x, **wts, **ats, **calc_info_dict)
            else:
                y = self.run_layer(layer, *x, **wts, **ats)

            if not isinstance(y, (tuple, list)):
                y = (y, 1)
            
            # 判断输出是否是两级结构
            if not isinstance(y[0], (tuple, list)):
                y = (y, )
            
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=y, **kwargs)
            
            if IsReuseLayer:
                data[reuse_layer_name] = y
            else:
                data[name] = y
                
            # data[name] = tuple(y)
            
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

    def run_layer(self, layer, *args, **kwargs):
        return self.run_op(layer.op.op_id, *args, **kwargs)

    def fn_conv2d(self, x, **kwargs):
        # 模拟A111 片上 卷积计算流程
        if 'a_quantizer' not in kwargs.keys():
            if isinstance(x, tuple):
                in_ = x[0] * x[1]
            else:
                in_ = x
            output = super().fn_conv2d(in_, **kwargs)
            output = (output, 1)
        else:
            if 'auto_pad' in kwargs.keys():
                kwargs.pop('auto_pad')
            layer_sim = conv2d_quant_noise(**kwargs).to(self.device)
            output = layer_sim(x)
        return output 
    
    def fn_matmul(self, x, **kwargs):
        # 模拟A111 片上 全连接计算流程
        if 'a_quantizer' not in kwargs.keys():
            if isinstance(x, tuple):
                in_ = x[0] * x[1]
            else:
                in_ = x
            output = super().fn_matmul(in_, **kwargs)
            output = (output, 1)
        else:
            layer_sim = linear_quant_noise(**kwargs).to(self.device)
            output = layer_sim(x)
        return output 
        
    def fn_relu(self, x):
        if isinstance(x, tuple):
            layer_sim = ReLu_quant().to(self.device)
            output = layer_sim(x)
            
        else:
            if isinstance(x, tuple):
                in_ = x[0] * x[1]
            else:
                in_ = x
            output = F.relu(in_)
            output = (output, 1)
        return output
    
    def fn_avgpool2d(self, x, **kwargs):
        if 'dilation' in kwargs.keys():
            kwargs.pop('dilation')
        if 'auto_pad' in kwargs.keys():
            kwargs.pop('auto_pad')
        layer_sim = AvgPool2d_quant(**kwargs).to(self.device)
        output = layer_sim(x)
        return output

    def fn_maxpool2d(self, x, **kwargs):
        if 'auto_pad' in kwargs.keys():
            kwargs.pop('auto_pad')
        layer_sim = MaxPool2d_quant(**kwargs).to(self.device)
        output = layer_sim(x)
        return output
    
    def fn_global_avg_pool2d(self, x, **kwargs):
        
        layer_sim = AdaptiveAvgPool2d_quant(**kwargs).to(self.device)
        output = layer_sim(x)
        return output
    
    def fn_add(self, x, y, **kwargs):
        
        if isinstance(x, tuple):
            layer_sim = add_quant(**kwargs).to(self.device)
            output = layer_sim(x, y)
        else:
            output = x + y
        return output
    
    def fn_flatten(self, x, *, start_dim):
        if isinstance(x, tuple):
            return torch.flatten(x[0], start_dim=start_dim), x[1]
        else:
            return torch.flatten(x, start_dim=start_dim)

    def fn_pad(self, x, *, pads, value=0):
        assert len(pads) % 2 == 0
        # 将 onnx pad 转换为torch pad
        dim_len = len(pads) // 2
        torch_pad = []
        for i in range(dim_len-1,-1,-1):
            torch_pad.append(pads[i])
            torch_pad.append(pads[i+dim_len])
        torch_pad = tuple(torch_pad)
        if isinstance(x, tuple):
            return F.pad(x[0], torch_pad, value=value), x[1]
        else:
            return F.pad(x, torch_pad, value=value)
    
    def fn_split(self, x, **kwargs):
        if isinstance(x, tuple):
            assert len(x) == 2
            split_size = x[0].shape[1] // kwargs['split']
            y = torch.split(x[0], split_size, dim=kwargs['axis'])
            re = []
            for i in y:
                re.append((i, x[1]))
            return re
        else:
            split_size = x.shape[1] // kwargs['split']
            return torch.split(x, split_size, dim=kwargs['axis'])
        
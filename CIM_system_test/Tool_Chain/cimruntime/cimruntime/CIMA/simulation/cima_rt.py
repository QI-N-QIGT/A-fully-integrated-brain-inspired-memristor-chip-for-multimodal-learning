from ...numpy import NumpyRuntime
from ...cimtensor import CIMNumpyTensor as CNT
from .utils import *
from e100_irmapper.device.CIMA import MappedLayer # noqa
from typing import Callable
from e100_irtool.core import BaseIR
from e100_irtool.tools import flatten_layers  # noqa
from e100_irtool.runtime.utils import *
import copy
import warnings

class CIMANumpyRT(NumpyRuntime):
    
    name = "CIMA"
    
    def __init__(self, *, activation_lut = None, weight_noise = 0., output_noise=0.):
        '''
        ================== 全整型数据流 (模拟CIMA芯片计算流程) ==================
        activation_lut: numpy, 激活函数查找表，用于实现复杂激活函数 (silu, sigmoid, tanh)
        '''
        super().__init__()
        self.activation_lut = activation_lut
        self.weight_noise = weight_noise
        self.output_noise = output_noise
              
    def run_ir(self, ir_, inputs, weights=None, log_info=False, *, outputs=None, callback=None, **kwargs):
        # 将layer排序
        ir = copy.deepcopy(ir_)
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True)) 
        
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

        for k, v in data.items():
            assert k in layers, f'invalid input name {k!r}'
            assert isinstance(v, (tuple, list)), \
                f'invalid inputs type {type(v)}'
        # print(data.keys())        
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
        
        # input()
        self.layer_calc_time = {}
        # print(oup)
        # print(layers.keys())
        # print(data.keys())
        self.log_info = log_info
        
        for name, layer in layers.items():
            
            if self.log_info:
                print(f'当前计算层 : {name} ===>')
            
            if name in data:
                continue    # layer is done
            
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                data[name] = [np.array(layer.op.value)]
                continue
            
            if any(dd.parse_ref()[0] not in data for dd in layer.inputs):
                continue    # layer can't be run
            
            if name == oup:
                data[name] = []
                for o in layer.inputs:
                    name_ = o.parse_ref()[0]
                    assert name_ in data.keys()
                    data[name].append(data[name_])
                
                break       # output layer
            
            IsReuseLayer = False
            # resue layer
            if layer.type == 'reuse':
                reuse_layer_name = name
                name = layer.layer
                # 替换reuse层的输入
                reuse_layer = ir.layers[name]
                reuse_layer.inputs = layer.inputs
                layer = reuse_layer
                IsReuseLayer = True
                
            x = []
            for dd in layer.inputs:
                nm, idx = dd.parse_ref()
                x.append(data[nm][0 if idx is None else idx])
            
            ats = layer.op.get_attrs()
            for ats_n in ['with_batch', 'channel_pos']:
                if ats_n in ats.keys():
                    ats.pop(ats_n)
            
            wts = dict()
            device_info = dict()
            layer_info = dict()
            
            if layer.CIMA_mapping_info != None:
                device_info.update(dict(device_info=ir.devices))
                layer_info.update(dict(layer_info=ir.layers[name], layer_name = name))
                
            if layer.op.op_id in ['conv2d','matmul','fc', 'linear','conv_transpose2d', 'fused_conv2d', 'fused_fc']:
                
                if layer.op.bias:
                    bias_name = f"{name}.bias"
                    if bias_name not in weights.keys():
                        warnings.warn(f'{bias_name} 不在 weights 中, {name} 没有加载bias !!!')
                    wts['bias'] = weights.get(bias_name)
                    
                for k in layer.op.weights:
                    wn = f'{name}.{k}'
                    if k not in layer.op.optional_weights:
                        assert wn in weights, f'missing weight {wn}'
                    wts[k] = weights.get(wn)
                
                # 获取权重数据
                wn = f'{name}.weight'
                wts['weight'] = weights.get(wn)
                
            elif layer.op.op_id in ['batch_norm2d']:
                
                for k in layer.op.weights:
                    ats[k] = np.array(ats[k])
            
            # record weight scale
            if layer.CIMA_mapping_info == None and layer.CIMA_calc_info != None:
                layer_info.update(dict(layer_info=ir.layers[name]))
            
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=None, **kwargs)
            # record time
            time1 = time.time()
            
            # 计算当前层
            y = self.run_layer(layer, *x, **wts, **ats, **device_info, **layer_info)
            
            # input()
            time2 = time.time()
            self.layer_calc_time[name] = round(time2 - time1, 4)
            
            if not isinstance(y, (tuple, list)):
                y = (y,)
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=y, **kwargs)
            
            if IsReuseLayer:
                data[reuse_layer_name] = tuple(y)
            else:
                data[name] = tuple(y)
            
            if ons is not None and all(k in data for k in ons):
                break       # all outputs are ready
        # print(data.keys())
           
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
        # print(res)
        return res

    def run_op(self, op_id, *args, **kwargs):
        fn = getattr(self, f'fn_{op_id}', None)
        assert isinstance(fn, Callable), f'fn_{op_id} is not a function'
        return fn(*args, **kwargs)

    def run_layer(self, layer, *args, **kwargs):
        return self.run_op(layer.op.op_id, *args, **kwargs)
    
    # matmuls
    def fn_matmul(self, x, **kwargs):
        '''
        =================================================
                      CIMA 硬件仿真整型数据流(MatMul)
        =================================================
        input:
          x: numpy, 输入为整型数据 '4bit/8bit' 
        weight / bias:
          权重和bias数据也是整型数据, 外部量化完成(模型训练给定)
        output:
          结果为硬件量化之后的输出结果 '4bit/8bit'
        =================================================
        '''
        
        # 判断是否用CIM做运算
        if 'device_info' not in kwargs.keys():
            # 软件计算
            output = super().fn_fc(x,**kwargs)
            return output
        else:
            # step 1. 获取计算所必须的运行时参数
            layer_info = kwargs['layer_info']
            ADC_qunat_level = layer_info.CIMA_calc_info.ADC_qunat_level      
            scale_shift_num = layer_info.CIMA_calc_info.scale_shift_num
            scale = layer_info.CIMA_calc_info.scale
            offset = layer_info.CIMA_calc_info.offset
            accumulate_shift_num = layer_info.CIMA_calc_info.accumulate_shift_num
            data_type = layer_info.CIMA_calc_info.data_type
            
            # step 2. 获取权重
            weight_data = kwargs['weight']
            
            # step 3. CIMA模拟核计算
            output = CIMA_analog_MAC(x, weight_data, dtype=data_type, ADC_qunat_level=ADC_qunat_level, 
                                     scale=scale, offset=offset, scale_shift_num=scale_shift_num,
                                     accumulate_shift_num=accumulate_shift_num)
            
            return output
        

    fn_linear = fn_matmul
    fn_fc = fn_matmul
    fused_fc = fn_matmul
    
    # conv
    
    def fn_conv2d(self, x, **kwargs):
        '''
        =================================================
                      CIMA 硬件仿真数据流(Conv2d)
        =================================================
        input:
          x: numpy, 输入为整型数据 '4bit/8bit' 
        weight / bias:
          权重和bias数据也是整型数据, 外部量化完成(模型训练给定)
        output:
          结果为硬件量化之后的输出结果 '4bit/8bit'
        =================================================
        '''
        
        if 'device_info' not in kwargs.keys():
            # B H W C => B C H W
            x = x.transpose(0, 3, 1, 2) 
            output = super().fn_conv2d(x,**kwargs)
            return output
        
        else:
            
            # step 1. 获取计算所必须的运行时参数
            layer_info = kwargs['layer_info']
            ADC_quant_level = layer_info.CIMA_calc_info.ADC_quant_level      
            scale_shift_num = layer_info.CIMA_calc_info.scale_shift_num
            scale = np.array(layer_info.CIMA_calc_info.scale)
            offset = np.array(layer_info.CIMA_calc_info.offset)
            accumulate_shift_num = layer_info.CIMA_calc_info.accumulate_shift_num
            data_type = layer_info.CIMA_calc_info.data_type
            
            # step 2. 获取输入的维度，仿真中默认与硬件保持一致，输入与权重的排列方式为HWC
            assert len(x.shape) == 4, f'默认x为带有batch的输入'
            batch, input_rows, input_cols, channel = x.shape
            # step 3. 获取算子参数
            padding = layer_info.op.padding
            kernel_size = layer_info.op.kernel
            # stride = layer_info.op.stride
            if 'stride' in kwargs.keys():
                stride = kwargs['stride']
            else:
                stride = layer_info.op.stride
            out_feature_size_rows = int((input_rows + padding + padding - kernel_size) / stride + 1)
            out_feature_size_cols = int((input_cols + padding + padding - kernel_size) / stride + 1)
            # step 4. 输入重排 
            # 此时输入应该默认为HWC的格式
            # 默认此时带有batch计算
            array_input = feature_map_to_input_np_HWC(x, stride = stride, kernel_size = kernel_size,
                                        padding = padding, multi_batch = True)
            
            # weight
            weight_data = kwargs['weight']
            
            # 获取计算设备名称，根据名称选择在RRAM还是DMAC上计算
            for k,v in layer_info.CIMA_mapping_info.mappings.items():
                device_name = v.device
                break
            if self.log_info:
                print(f'Compute location: {device_name}')
            if 'dmac' in device_name:
                output = CIMA_digital_MAC(array_input, weight_data, scale=scale, offset=offset, scale_shift_num=scale_shift_num,
                                        accumulate_shift_num=accumulate_shift_num)
            elif 'cima-xb' in device_name:
                # step5. CIMA模拟核计算
                output = CIMA_analog_MAC(array_input, weight_data, dtype=data_type, ADC_quant_level=ADC_quant_level, 
                                        scale=scale, offset=offset, scale_shift_num=scale_shift_num,
                                        accumulate_shift_num=accumulate_shift_num,
                                        conductance_noise=self.weight_noise, ADC_noise=self.output_noise)
            else:
                raise ValueError(f' 暂不支持设备 {device_name} !!!')
            
            # step5. 输出维度还原
            output = output_to_feature_map(output, out_feature_size_rows, out_feature_size_cols, multi_batch=True)
            
            # 此时输出默认为CHW排列, 需要转换为 HWC排列
            output = output.transpose(0,2,3,1)
                
            return output
    
    # fn_fused_conv2d = fn_conv2d
    # convtranspose2d
    
    def fn_conv_transpose2d(self, x, **kwargs):
        '''
        =================================================
                      CIMA 硬件仿真数据流(ConvTranspose2d)
        =================================================
        input:
          x: numpy, 输入为整型数据 '4bit/8bit' 
        weight / bias:
          权重和bias数据也是整型数据, 外部量化完成(模型训练给定)
        output:
          结果为硬件量化之后的输出结果 '4bit/8bit'
        =================================================
        '''
        
        if 'device_info' not in kwargs.keys():
            output = super().fn_conv_transpose2d(x,**kwargs)
            return output
        else:    
            in_ = x  
            # 获取参数
            group = kwargs['group']
            stride = kwargs['stride']
            padding = kwargs['padding']
            dilation = kwargs['dilation']
            output_padding = kwargs['output_padding']
            auto_pad = kwargs['auto_pad']
            
            weight_shape = kwargs['layer_info'].weights['weight'].shape
            kci, co, *kernel = weight_shape
            
            # 变换输入
            ndim = 2
            
            if self.channel_last:
                ba, *xd, ci = in_.shape
            else:
                ba, ci, *xd = in_.shape
            assert ci == kci * group, \
                f'invalid input shape {in_.shape} with kernel {weight_shape}'
            os, (k, s, p, d, dk, dp, di) = \
                conv_t_shapes(xd, kernel, stride, padding, dilation,
                            output_padding, auto_pad)
            if di != xd:
                xp = np.zeros(self.to_axes(ba, ci, di), dtype=in_.dtype)
                oi = tuple(slice(dp[i], xd[i] * s[i] + dp[i], s[i])
                        for i in range(ndim))
                xp[self.to_slices(oi)] = in_
                in_ = xp
                
            # 调用卷积的函数计算
            if stride != 1:
                kwargs['stride'] = 1
            
            return self.fn_conv2d(in_, **kwargs)
            
            
    def rand(self, shape):
        data = self._be.random.randint(low=0, high=self.half_level, size=shape)
        return CNT.to_cimtensor(data=data,multi_batch=self.multi_batch)
    
    def fn_identity(self, x, **kwargs):
        return x
    # transes
    
    def fn_concat(self, *x, **kwargs):
        axis = kwargs['layer_info'].op.axis
        # CIMA 推理数据采用BHWC排布，与原始的pytorch的 BCHW不一样
        if axis == 1:
            axis = 3
        output = CIMA_concat(*x, axis=axis)
        
        layer_info = kwargs['layer_info']
        data_type = layer_info.CIMA_calc_info.data_type
        op_type = layer_info.op.op_id
        
        # 判断是否需要激活
        if op_type == 'fused_concat':
            layer_name = kwargs['layer_name']
            if layer_info.op.silu != None:
                
                assert layer_name in self.activation_lut.keys(), f'{layer_name} not in {self.activation_lut.keys()}!!!'
                lut = self.activation_lut[layer_name]
                if data_type == '4bit': 
                    output_query = output + 8
                elif data_type == '8bit':
                    output_query = output + 128
                else:
                    raise ValueError(f'暂不支持 {data_type} !!!')
                output_query = output_query.astype(np.int32)
                output = lut[output_query]
            elif layer_info.op.relu != None:
                output = self._be.clip(output, 0)

            if layer_info.op.split != None:
                # attr
                axis = kwargs['layer_info'].op.split.axis
                # CIMA 推理数据采用BHWC排布，与原始的pytorch的 BCHW不一样
                if axis == 1:
                    axis = 3
                split = kwargs['layer_info'].op.split.split
                output = super().fn_split(output, axis=axis, split=split)
                re_ = []
                for d in output:
                    if len(d.shape) == 3:
                        d = d.squeeze()
                    re_.append(d)
                output = re_
            
        return output

    fn_fused_concat = fn_concat
    
    def fn_add(self, *x, **kwargs):
        
        layer_info = kwargs['layer_info']
        data_type = layer_info.CIMA_calc_info.data_type
        op_type = layer_info.op.op_id
        
        output = CIMA_add(*x, dtype=data_type)
        
        # 判断是否需要激活
        if op_type == 'fused_add':
            layer_name = kwargs['layer_name']
            if layer_info.op.silu != None:
                assert layer_name in self.activation_lut.keys(), f'{layer_name} not in {self.activation_lut.keys()}!!!'
                lut = self.activation_lut[layer_name]
                if data_type == '4bit': 
                    output_query = output + 8
                elif data_type == '8bit':
                    output_query = output + 128
                else:
                    raise ValueError(f'暂不支持 {data_type} !!!')
                output_query = output_query.astype(np.int32)
                output = lut[output_query]
            elif layer_info.op.relu != None:
                output = self._be.clip(output, 0)
            else:
                raise ValueError(f'暂未实现融合层 {layer_name} !!!')
        # 类型对齐为整型
        output = output.astype(np.int32)    
        return output
    
    fn_fused_add = fn_add
    
    def fn_mul_add(self, x, **kwargs):
        # attributes
        scale = int(kwargs['layer_info'].CIMA_calc_info.scale)
        scale_shift_num = int(kwargs['layer_info'].CIMA_calc_info.scale_shift_num)
        offset = kwargs['layer_info'].CIMA_calc_info.offset
        dtype = kwargs['layer_info'].CIMA_calc_info.data_type
        # mul add
        x = x.astype(np.int32)
        re = CIMA_mul_add(x, scale=scale, scale_shift_num=scale_shift_num, offset=offset, dtype=dtype)
        return re
    
    # activate
    def fn_silu(self, x, **kwargs):
        layer_name = kwargs['layer_name']
        data_type = kwargs['layer_info'].CIMA_calc_info.data_type
        assert layer_name in self.activation_lut.keys(), f'{layer_name} not in {self.activation_lut.keys()}!!!'
        lut = self.activation_lut[layer_name]
        # 
        output =  CIMA_silu(x, lut, data_type=data_type)
        return output
    
    def fn_relu(self, x, **kwargs):
        output = self._be.clip(x, 0)
        return output
    
    # poolings
    
    def fn_avg_pool2d(self, x, **kwargs):
        # TODO
        pass

    fn_avgpool2d = fn_avg_pool2d

    
    def fn_max_pool2d(self, x, **kwargs):
        stride = kwargs['layer_info'].op.stride
        padding = kwargs['layer_info'].op.padding
        kernel = kwargs['layer_info'].op.kernel
        # 将 x layout BHWC 变换为 BCHW
        x = x.transpose(0, 3, 1, 2)
        output = self._pool(self._be.amax, 2, x, stride=stride, padding=padding, kernel=kernel, ceil_mode=0, auto_pad=0, dilation=1)
        # 将结果layout 再变为 BHWC
        output = output.transpose(0, 2, 3, 1)
        return output
    
    fn_maxpool2d = fn_max_pool2d

    
    # resize
    def fn_resize(self, x, **kwargs):
        scale = kwargs['layer_info'].op.scale[2:]
        # 将 x layout BHWC 变换为 BCHW
        x = x.transpose(0, 3, 1, 2)
        re = super().fn_resize(x, size=None, scale=scale, mode='nearest')
        # 将结果layout 再变为 BHWC
        re = re.transpose(0, 2, 3, 1)
        return re
    
    # split
    def fn_split(self, x, **kwargs):
        # attr
        axis = kwargs['layer_info'].op.axis
        # CIMA 推理数据采用BHWC排布，与原始的pytorch的 BCHW不一样
        if axis == 1:
            axis = 3
        split = kwargs['layer_info'].op.split
        data = x
        re = super().fn_split(data, axis=axis, split=split)
        re_ = []
        for d in re:
            if len(d.shape) == 3:
                d = d.squeeze()
            re_.append(d)
        return re_
    
from .math import MathRuntime
from e100_irtool.core.type_util import to_int_tuple
from e100_irtool.runtime.utils import auto_pad_pool, concat_axis
from typing import Callable
from e100_irtool.core import BaseIR
from e100_irtool.tools import flatten_layers  # noqa

class TorchRuntime(MathRuntime):

    name = 'sim_torch'
    channel_last = False
    tensor_is_readonly = False

    def init_backend(self):
        import torch
        self._be = torch
        self._fn = torch.nn.functional
        self.tensor_type = torch.Tensor

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
   
        for name, layer in layers.items():
            # print(f'当前层 : {name} ===>')
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                continue
            if name in data:
                continue    # layer is done
            if any(dd.parse_ref()[0] not in data for dd in layer.inputs):
                continue    # layer can't be run
            if name == oup:
                break       # output layer
            x = []
            for dd in layer.inputs:
                nm, idx = dd.parse_ref()
                x.append(data[nm][0 if idx is None else idx])
            ats = layer.op.get_attrs()
            # for ats_n in ['auto_pad', 'with_batch', 'channel_pos']:
            #     if ats_n in ats.keys():
            #         ats.pop(ats_n)
            
            wts = dict()
            # device_info = dict()
            # layer_info = dict()
            
            if layer.op.op_id in ['conv2d','matmul','fc','linear']:
                for k in layer.op.weights:
                    wn = f'{name}.{k}'
                    if k not in layer.op.optional_weights:
                        assert wn in weights, f'missing weight {wn}'
                    wts[k] = weights.get(wn)
            
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=None, **kwargs)            
            
            # y = self.run_layer(layer, *x, **wts, **ats, **device_info, **layer_info)
            y = self.run_layer(layer, *x, **wts, **ats)
            
            if not isinstance(y, (tuple, list)):
                y = (y,)
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=y, **kwargs)
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

    def run_layer(self, layer, *args, **kwargs):
        return self.run_op(layer.op.op_id, *args, **kwargs)

    def rand(self, shape):
        return self._be.rand(shape)

    def as_tensor(self, data, dtype=None):
        if isinstance(dtype, str):
            dtype = getattr(self._be, dtype)
        return self._be.as_tensor(data, dtype=dtype)

    # transes

    def fn_concat(self, *x, axis, with_batch, channel_pos):
        axis = concat_axis(len(x[0].shape) - 1, self.channel_last, axis,
                           with_batch, channel_pos)
        return self._be.cat(x, dim=axis+1)

    def fn_reshape(self, x, *, shape, with_batch):
        if with_batch:
            return self._be.reshape(x, shape)
        else:
            return self._be.reshape(x, (x.shape[0], *shape))

    def fn_flatten(self, x, *, start_dim, with_batch):
        if with_batch:
            return self._be.flatten(x, start_dim=start_dim)
        else:
            return self._be.flatten(x, start_dim=start_dim + 1)

    def fn_transpose(self, x, *, perm, with_batch):
        if with_batch:
            return x.permute(perm)
        else:
            return x.permute((0, *(i + 1 for i in perm)))

    def fn_pad(self, x, *, pads, value):
        ndim = len(pads) // 2
        p = []
        for i in range(ndim, 0, -1):
            p.append(pads[i - 1])
            p.append(pads[ndim + i - 1])
        return self._fn.pad(x, tuple(p), value=value)

    # activations

    def fn_relu(self, x):
        return self._fn.relu(x)

    def fn_leaky_relu(self, x, *, alpha):
        return self._fn.leaky_relu(x, negative_slope=alpha)

    def fn_prelu(self, x, *, slope):
        return self._fn.prelu(x, slope)

    def fn_selu(self, x, *, alpha, gamma):
        return self._fn.elu(x, alpha) * gamma

    def fn_celu(self, x, *, alpha):
        return self._fn.celu(x, alpha=alpha)

    def fn_elu(self, x, *, alpha):
        return self._fn.elu(x, alpha=alpha)

    def fn_softmax(self, x, *, axis):
        return self._fn.softmax(x, dim=axis)

    def fn_log_softmax(self, x, axis):
        return self._fn.log_softmax(x, dim=axis)

    def fn_sigmoid(self, x):
        return self._be.sigmoid(x)

    def fn_hard_sigmoid(self, x, *, alpha, beta):
        return self._be.clip(x * alpha + beta, 0, 1)

    def fn_softplus(self, x):
        return self._fn.softplus(x)

    def fn_softsign(self, x):
        return self._fn.softsign(x)

    def fn_silu(self, x):
        return self._fn.silu(x)

    # normalizations

    def fn_batch_norm(self, x, *, scale, bias, input_mean, input_var,
                      epsilon):
        return self._fn.batch_norm(x, input_mean, input_var,
                                   weight=scale, bias=bias, eps=epsilon)

    fn_batch_norm1d = fn_batch_norm
    fn_batch_norm2d = fn_batch_norm
    fn_batch_norm3d = fn_batch_norm

    def fn_instance_norm(self, x, *, scale, bias, epsilon):
        return self._fn.instance_norm(x, weight=scale, bias=bias, eps=epsilon)

    fn_instance_norm1d = fn_instance_norm
    fn_instance_norm2d = fn_instance_norm
    fn_instance_norm3d = fn_instance_norm

    # poolings

    def _avg_pool(self, ndim, x, *, kernel, stride, padding, dilation,
                  ceil_mode, auto_pad):
        assert set(to_int_tuple(dilation, ndim=ndim)) == {1}, \
            f'dilation {dilation} is not supported'
        if padding is None:
            dims = x.shape[1:ndim+1] if self.channel_last \
                else x.shape[2:ndim+2]
            padding = auto_pad_pool(auto_pad, dims, kernel, stride,
                                    dilation)
        fn = getattr(self._fn, f'avg_pool{ndim}d')
        return fn(x, kernel, stride=stride, padding=padding,
                  ceil_mode=ceil_mode, count_include_pad=False)

    def fn_avg_pool1d(self, x, **kwargs):
        return self._avg_pool(1, x, **kwargs)

    def fn_avg_pool2d(self, x, **kwargs):
        return self._avg_pool(2, x, **kwargs)

    def fn_avg_pool3d(self, x, **kwargs):
        return self._avg_pool(3, x, **kwargs)

    fn_avgpool1d = fn_avg_pool1d
    fn_avgpool2d = fn_avg_pool2d
    fn_avgpool3d = fn_avg_pool3d

    def _max_pool(self, ndim, x, *, kernel, stride, padding, dilation,
                  ceil_mode, auto_pad):
        if padding is None:
            dims = x.shape[1:ndim+1] if self.channel_last \
                else x.shape[2:ndim+2]
            padding = auto_pad_pool(auto_pad, dims, kernel, stride,
                                    dilation)
        fn = getattr(self._fn, f'max_pool{ndim}d')
        return fn(x, kernel, stride=stride, padding=padding,
                  dilation=dilation, ceil_mode=ceil_mode)

    def fn_max_pool1d(self, x, **kwargs):
        return self._max_pool(1, x, **kwargs)

    def fn_max_pool2d(self, x, **kwargs):
        return self._max_pool(2, x, **kwargs)

    def fn_max_pool3d(self, x, **kwargs):
        return self._max_pool(3, x, **kwargs)

    fn_maxpool1d = fn_max_pool1d
    fn_maxpool2d = fn_max_pool2d
    fn_maxpool3d = fn_max_pool3d

    def fn_global_avg_pool1d(self, x):
        return self._fn.adaptive_avg_pool1d(x, 1)

    def fn_global_avg_pool2d(self, x):
        return self._fn.adaptive_avg_pool2d(x, 1)

    def fn_global_avg_pool3d(self, x):
        return self._fn.adaptive_avg_pool3d(x, 1)

    def fn_global_max_pool1d(self, x):
        return self._fn.adaptive_max_pool1d(x, 1)

    def fn_global_max_pool2d(self, x):
        return self._fn.adaptive_max_pool2d(x, 1)

    def fn_global_max_pool3d(self, x):
        return self._fn.adaptive_max_pool3d(x, 1)

    # matmuls

    def fn_matmul(self, x, *, weight, bias):
        y = self._fn.linear(x, weight, bias=bias)
        return y

    fn_linear = fn_matmul
    fn_fc = fn_matmul

    # convs

    def _conv(self, ndim, x, *, weight, bias, stride, padding, dilation,
              group, auto_pad):
        fn = getattr(self._fn, f'conv{ndim}d')
        return fn(x, weight, bias=bias, stride=stride, padding=padding,
                  dilation=dilation, groups=group)

    def fn_conv1d(self, x, **kwargs):
        return self._conv(1, x, **kwargs)

    def fn_conv2d(self, x, **kwargs):
        return self._conv(2, x, **kwargs)

    def fn_conv3d(self, x, **kwargs):
        return self._conv(3, x, **kwargs)

    def _conv_t(self, ndim, x, *, weight, bias, stride, padding, dilation,
                output_padding, group, auto_pad):
        fn = getattr(self._fn, f'conv_transpose{ndim}d')
        return fn(x, weight, bias=bias, stride=stride, padding=padding,
                  dilation=dilation, output_padding=output_padding,
                  groups=group)

    def fn_conv_transpose1d(self, x, **kwargs):
        return self._conv_t(1, x, **kwargs)

    def fn_conv_transpose2d(self, x, **kwargs):
        return self._conv_t(2, x, **kwargs)

    def fn_conv_transpose3d(self, x, **kwargs):
        return self._conv_t(3, x, **kwargs)

    # resize

    def fn_resize(self, x, *, size, scale, mode):
        if size is not None:
            return self._fn.interpolate(x, size, mode=mode)
        else:
            return self._fn.interpolate(x, size, scale, mode=mode,
                                        recompute_scale_factor=True)

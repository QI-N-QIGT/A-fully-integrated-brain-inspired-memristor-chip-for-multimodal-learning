from .math import MathRuntime
from e100_irtool.core.type_util import to_int_tuple
from itertools import product
from e100_irtool.runtime.utils import *

class NumpyRuntime(MathRuntime):
    
    # 第一版的numpy runtime实现方法 (2023.02z之前)
    name = 'numpy_old'

    def __init__(self):
        import numpy
        self._be = numpy
        self.tensor_type = numpy.ndarray

    def rand(self, shape):
        return self._be.random.random(shape)

    def as_tensor(self, data, dtype):
        return self._be.asarray(data, dtype=dtype)

    # transes

    def fn_concat(self, *x, axis):
        return self._be.concatenate(x, axis=axis)

    def fn_reshape(self, x, *, shape):
        return self._be.reshape(x, shape)

    def fn_flatten(self, x, *, start_dim):
        return self._be.reshape(x, (*x.shape[:start_dim], -1))

    def fn_transpose(self, x, *, perm):
        return self._be.transpose(x, perm)

    # activate

    def fn_relu(self, x):
        return self._be.clip(x, 0, None)

    def fn_leaky_relu(self, x, *, alpha):
        return self._be.where(x < 0, x * alpha, x)

    def fn_prelu(self, x, *, slope):
        return self._be.where(x < 0, x * slope, x)

    def fn_selu(self, x, *, alpha, gamma):
        return self._be.where(x < 0, (self._be.exp(x) - 1) * alpha * gamma,
                              x * gamma)

    def fn_celu(self, x, *, alpha):
        return self._be.where(x < 0, (self._be.exp(x / alpha) - 1) * alpha, x)

    def fn_elu(self, x, *, alpha):
        return self._be.where(x < 0, (self._be.exp(x) - 1) * alpha, x)

    def fn_softmax(self, x, *, axis):
        y = self._be.exp(x)
        s = self._be.sum(y, axis=axis, keepdims=True)
        return y / s

    def fn_log_softmax(self, x, *, axis):
        y = self._be.exp(x)
        s = self._be.sum(y, axis=axis, keepdims=True)
        return self._be.log(y / s)

    def fn_sigmoid(self, x):
        return 1 / (1 + self._be.exp(-x))

    def fn_hard_sigmoid(self, x, *, alpha, beta):
        return self._be.clip(x * alpha + beta, 0, 1)

    def fn_softplus(self, x):
        return self._be.log(self._be.exp(x) + 1)

    def fn_softsign(self, x):
        return x / (1 + self._be.abs(x))

    # normalizations

    def _broadcast(self, x, rank, ndim=None):
        n = rank - len(x.shape)
        if ndim is None:
            ndim = n
        if n > 0:
            x =  x.reshape(x.shape + (1,) * ndim)

        return x

    # def fn_batch_norm(self, x, *, scale, bias, input_mean, input_var,
    #                   epsilon):
    #     rank = len(x.shape)
    #     input_mean = self._broadcast(input_mean, rank)
    #     input_var = self._broadcast(input_var, rank)
    #     y = (x - input_mean) / self._be.sqrt(input_var + epsilon)
    #     if scale is not None:
    #         y *= self._broadcast(scale, rank)
    #     if bias is not None:
    #         y += self._broadcast(bias, rank)
    #     return y

    # fn_batch_norm1d = fn_batch_norm
    # fn_batch_norm2d = fn_batch_norm
    # fn_batch_norm3d = fn_batch_norm

    def fn_instance_norm(self, x, *, scale, bias, epsilon, ndim):
        rank = len(x.shape)
        axis = tuple(range(-ndim, 0))
        mean = self._be.mean(x, axis=axis, keepdims=True)
        var = self._be.var(x, axis=axis, keepdims=True)
        y = (x - mean) / self._be.sqrt(var + epsilon)
        if scale is not None:
            y *= self._broadcast(scale, rank, ndim)
        if bias is not None:
            y += self._broadcast(bias, rank, ndim)
        return y

    def fn_instance_norm1d(self, x, **kwargs):
        return self.fn_instance_norm(x, ndim=1, **kwargs)

    def fn_instance_norm2d(self, x, **kwargs):
        return self.fn_instance_norm(x, ndim=2, **kwargs)

    def fn_instance_norm3d(self, x, **kwargs):
        return self.fn_instance_norm(x, ndim=3, **kwargs)

    # kerneled ops

    def _kerneled(self, x, *, ndim, func, kernel, stride, padding, dilation,
                  ceil_mode=False, dont_pad=False, o_channel=None, **kwargs):
        bc = len(x.shape) - ndim
        p = to_int_tuple(padding, ndim=ndim*2)
        xs = x.shape
        if not dont_pad and any(i > 0 for i in p):
            pads = [(0, 0)] * bc + [(p[i], p[ndim + i]) for i in range(ndim)]
            x = self._be.pad(x, pads)
        k = to_int_tuple(kernel, ndim=ndim)
        s = to_int_tuple(stride, ndim=ndim)
        d = to_int_tuple(dilation, ndim=ndim)
        c = tuple(s[i] - 1 if ceil_mode else 0 for i in range(ndim))
        dk = tuple(d[i] * (k[i] - 1) + 1 for i in range(ndim))
        os = [((xs[bc + i] + p[i] + p[ndim + i] - dk[i] + c[i]) // s[i] + 1)
              for i in range(ndim)]
        if ceil_mode:
            for i in range(ndim):
                if (os[i] - 1) * s[i] >= xs[bc + i] + p[i]:
                    os[i] -= 1
        if o_channel is None:
            o_channel = xs[bc - 1]
        y = self._be.empty((*xs[:bc-1], o_channel, *os), dtype=x.dtype)
        n = (Ellipsis,) if bc else ()
        pl = p[:ndim] if dont_pad else (0,) * ndim
        for oi in product(*map(range, os)):
            ii = tuple(slice(max(0, oi[i] * s[i] - pl[i]),
                             oi[i] * s[i] - pl[i] + dk[i], d[i])
                       for i in range(ndim))
            y[n + oi] = func(x[n + ii], **kwargs)
        return y

     # poolings

    def _pool(self, func, ndim, x, *, kernel, stride, padding, dilation,
              ceil_mode, auto_pad):
        numpy = self._be
        if self.channel_last:
            ba, *xd, ci = x.shape
            axis = tuple(range(1, ndim+1))
        else:
            ba, ci, *xd = x.shape
            axis = tuple(range(2, ndim+2))
        assert len(xd) == ndim, \
            f'invalid input shape {x.shape} for {ndim}d pooling'
        os, (k, s, p, d, dk) = \
            pool_shapes(xd, kernel, stride, padding, dilation, auto_pad,
                        ceil_mode)
        y = numpy.empty(self.to_axes(ba, ci, os), dtype=x.dtype)
        for oi in numpy.ndindex(os):
            ii = (slice(max(0, oi[i] * s[i] - p[i]),
                        oi[i] * s[i] - p[i] + dk[i], d[i])
                  for i in range(ndim))
            y[self.to_slices(oi)] = func(x[self.to_slices(ii)], axis=axis)
        return y

    def fn_avg_pool1d(self, x, **kwargs):
        return self._pool(self._be.mean, 1, x, **kwargs)

    def fn_avg_pool2d(self, x, **kwargs):
        return self._pool(self._be.mean, 2, x, **kwargs)

    def fn_avg_pool3d(self, x, **kwargs):
        return self._pool(self._be.mean, 3, x, **kwargs)

    fn_avgpool1d = fn_avg_pool1d
    fn_avgpool2d = fn_avg_pool2d
    fn_avgpool3d = fn_avg_pool3d

    def fn_max_pool1d(self, x, **kwargs):
        return self._pool(self._be.amax, 1, x, **kwargs)

    def fn_max_pool2d(self, x, **kwargs):
        return self._pool(self._be.amax, 2, x, **kwargs)

    def fn_max_pool3d(self, x, **kwargs):
        return self._pool(self._be.amax, 3, x, **kwargs)

    fn_maxpool1d = fn_max_pool1d
    fn_maxpool2d = fn_max_pool2d
    fn_maxpool3d = fn_max_pool3d

    def _global_pool(self, func, ndim, x):
        axis = tuple(range(1, ndim + 1) if self.channel_last
                     else range(2, ndim + 2))
        return func(x, axis=axis, keepdims=True)

    def fn_global_avg_pool1d(self, x):
        return self._global_pool(self._be.mean, 1, x)

    def fn_global_avg_pool2d(self, x):
        return self._global_pool(self._be.mean, 2, x)

    def fn_global_avg_pool3d(self, x):
        return self._global_pool(self._be.mean, 3, x)

    def fn_global_max_pool1d(self, x):
        return self._global_pool(self._be.amax, 1, x)

    def fn_global_max_pool2d(self, x):
        return self._global_pool(self._be.amax, 2, x)

    def fn_global_max_pool3d(self, x):
        return self._global_pool(self._be.amax, 3, x)


    # matmuls

    def fn_matmul(self, x, *, weight, bias):
        y = self._be.matmul(x, weight.T)
        if bias is not None:
            y += bias
        return y

    fn_linear = fn_matmul
    fn_fc = fn_matmul

    # convs

    # def _conv(self, x, *, weight, bias, stride, padding, dilation, group,
    #           ndim):
        
    #     assert group == 1, 'conv with group != 1 is not supported'
    #     bs = x.shape[:-ndim-1]
    #     ws = weight.shape[0]

    #     def func(xx):
    #         xx = xx.reshape(*bs, -1)
    #         ww = weight.reshape(ws, -1)
    #         return self._be.matmul(xx, ww.T)

    #     y = self._kerneled(x, kernel=weight.shape[-ndim:], stride=stride,
    #                        padding=padding, dilation=dilation, ndim=ndim,
    #                        o_channel=ws, func=func)
    #     if bias is not None:
    #         y += self._broadcast(bias, len(y.shape), ndim)
    #     return y

    def _conv(self, x, ndim, *, weight, bias, stride, padding, dilation,
              group, auto_pad):
        numpy = self._be
        assert group == 1, 'conv with group != 1 is not supported'
        if self.channel_last:
            ba, *xd, ci = x.shape
            *kernel, kci, co = weight.shape
        else:
            ba, ci, *xd = x.shape
            co, kci, *kernel = weight.shape
        assert len(xd) == ndim, \
            f'invalid input shape {x.shape} for {ndim}d conv'
        assert ci == kci * group, \
            f'invalid input shape {x.shape} with kernel {weight.shape}'
        os, (k, s, p, d, dk) = \
            conv_shapes(xd, kernel, stride, padding, dilation, auto_pad)
        if any(p):
            x = numpy.pad(x, self.to_pads(p))
        wt = weight.reshape(-1, co) if self.channel_last \
            else weight.reshape(co, -1).T
        y = numpy.empty(self.to_axes(ba, co, os), dtype=x.dtype)
        for oi in numpy.ndindex(os):
            ii = (slice(max(0, oi[i] * s[i]), oi[i] * s[i] + dk[i], d[i])
                  for i in range(ndim))
            y[self.to_slices(oi)] = numpy.matmul(
                x[self.to_slices(ii)].reshape(ba, -1), wt)
        if bias is not None:
            y += self.broadcast(bias, len(y.shape), ndim)
        return y
    
    def fn_conv1d(self, x, **kwargs):
        return self._conv(x, ndim=1, **kwargs)

    def fn_conv2d(self, x, **kwargs):
        return self._conv(x, ndim=2, **kwargs)

    def fn_conv3d(self, x, **kwargs):
        return self._conv(x, ndim=3, **kwargs)

    def _conv_t(self, ndim, x, *, weight, bias, stride, padding, dilation,
                output_padding, group, auto_pad):
        numpy = self._be
        assert group == 1, 'conv_transpose with group != 1 is not supported'
        if self.channel_last:
            ba, *xd, ci = x.shape
            *kernel, co, kci = weight.shape
            wt = weight.reshape(-1, co, kci).transpose(0, 2, 1).reshape(-1, co)
        else:
            ba, ci, *xd = x.shape
            kci, co, *kernel = weight.shape
            wt = weight.reshape(kci, co, -1).transpose(0, 2, 1).reshape(-1, co)
        assert ci == kci * group, \
            f'invalid input shape {x.shape} with kernel {weight.shape}'
        os, (k, s, p, d, dk, dp, di) = \
            conv_t_shapes(xd, kernel, stride, padding, dilation,
                          output_padding, auto_pad)
        if di != xd:
            xp = numpy.zeros(self.to_axes(ba, ci, di), dtype=x.dtype)
            oi = tuple(slice(dp[i], xd[i] * s[i] + dp[i], s[i])
                       for i in range(ndim))
            xp[self.to_slices(oi)] = x
            x = xp
        y = numpy.empty(self.to_axes(ba, co, os), dtype=x.dtype)
        for oi in numpy.ndindex(os):
            ii = tuple(slice(oi[i] - (di[i] + 1 - dk[i]),
                             oi[i] - (di[i] + 1), -d[i])
                       for i in range(ndim))
            y[self.to_slices(oi)] = numpy.matmul(
                x[self.to_slices(ii)].reshape(ba, -1), wt)
        if bias is not None:
            y += self.broadcast(bias, len(y.shape), ndim)
        return y

    def fn_conv_transpose1d(self, x, **kwargs):
        return self._conv_t(1, x, **kwargs)

    def fn_conv_transpose2d(self, x, **kwargs):
        return self._conv_t(2, x, **kwargs)

    def fn_conv_transpose3d(self, x, **kwargs):
        return self._conv_t(3, x, **kwargs)

    # def fn_conv_transpose1d(self, x, **kwargs):
    #     return self._conv_transpose(x, ndim=1, **kwargs)

    # def fn_conv_transpose2d(self, x, **kwargs):
    #     print(kwargs)
    #     return self._conv_transpose(x, ndim=2, **kwargs)

    # def fn_conv_transpose3d(self, x, **kwargs):
    #     return self._conv_transpose(x, ndim=3, **kwargs)

    # pow
    def fn_pow(self, x, y):
        try:
            return self._be.power(x, y)
        except AttributeError:
            return x ** (y)
    
    # resize
    def fn_resize(self, x, *, size, scale, mode):
        numpy = self._be
        if self.channel_last:
            ba, *xd, ci = x.shape
        else:
            ba, ci, *xd = x.shape
        ndim = len(xd)
        yd, _ = resize_size_scale(xd, size, scale)
        assert mode == 'nearest', f'mode {mode} is not supported'
        y = numpy.empty(self.to_axes(ba, ci, yd), dtype=x.dtype)
        for oi in numpy.ndindex(yd):
            ii = (int(oi[i] / yd[i] * xd[i]) for i in range(ndim))
            y[self.to_slices(oi)] = x[self.to_slices(ii)]
        return y

    # split

    # def fn_split(self, x, *, axis, split, with_batch):
    def fn_split(self, x, *, axis, split):
        # if not with_batch:
        #     axis += 1
        secs = split_to_secs(split, x.shape[axis])
        i, idxs = 0, []
        for s in secs[:-1]:
            i += s
            idxs.append(i)
        return self._be.split(x, idxs, axis=axis)
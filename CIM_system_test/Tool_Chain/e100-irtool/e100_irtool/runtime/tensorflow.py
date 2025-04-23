from .math import MathRuntime
from ..core.type_util import to_int_tuple
from .utils import to_auto_pad, conv_t_shapes, resize_size_scale, \
                   concat_axis, split_to_secs


class TensorflowRuntime(MathRuntime):

    name = 'tensorflow'
    channel_last = True
    tensor_is_readonly = True

    DATA_FORMATS = (
        ('NCW', 'NWC'),
        ('NCHW', 'NHWC'),
        ('NCDHW', 'NDHWC')
    )

    def init_backend(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import tensorflow
        self._be = tensorflow
        self.tensor_type = tensorflow.Tensor

    def rand(self, shape):
        return self._be.random.uniform(shape)

    def as_tensor(self, data, dtype=None):
        return self._be.convert_to_tensor(data, dtype=dtype)

    def data_format(self, ndim):
        assert 1 <= ndim <= 3
        return self.DATA_FORMATS[ndim - 1][self.channel_last]

    def channel_pos(self):
        return 'channels_last' if self.channel_last else 'channels_first'

    # bitwises

    def fn_bitwise_not(self, x):
        return self._be.bitwise.bitwise_not(x)

    def fn_bitwise_and(self, x, y):
        return self._be.bitwise.bitwise_and(x, y)

    def fn_bitwise_or(self, x, y):
        return self._be.bitwise.bitwise_or(x, y)

    def fn_bitwise_xor(self, x, y):
        return self._be.bitwise.bitwise_xor(x, y)

    # transes

    def fn_concat(self, *x, axis, with_batch, channel_pos):
        axis = concat_axis(len(x[0].shape) - 1, self.channel_last, axis,
                           with_batch, channel_pos)
        return self._be.concat(x, axis+1)

    def fn_reshape(self, x, *, shape, with_batch):
        if with_batch:
            return self._be.reshape(x, shape)
        else:
            return self._be.reshape(x, (x.shape[0], *shape))

    def fn_flatten(self, x, *, start_dim, with_batch):
        if with_batch:
            return self._be.reshape(x, (*x.shape[:start_dim], -1))
        else:
            return self._be.reshape(x, (*x.shape[:start_dim + 1], -1))

    def fn_transpose(self, x, *, perm, with_batch):
        if with_batch:
            return self._be.transpose(x, perm=perm)
        else:
            return self._be.transpose(x, perm=(0, *(i + 1 for i in perm)))

    def fn_pad(self, x, *, pads, value):
        if self.channel_last:
            ba, *xd, ci = x.shape
        else:
            ba, ci, *xd = x.shape
        ndim = len(xd)
        assert len(pads) == ndim * 2, f'invalid pads {pads} for {ndim}d'
        return self._be.pad(x, self.to_pads(pads), constant_values=value)

    # activations

    def fn_relu(self, x):
        return self._be.nn.relu(x)

    def fn_leaky_relu(self, x, *, alpha):
        return self._be.nn.leaky_relu(x, alpha=alpha)

    def fn_prelu(self, x, *, slope):
        n = self._be.minimum(x, 0)
        p = self._be.maximum(x, 0)
        return n * slope + p

    def fn_selu(self, x, *, alpha, gamma):
        n = self._be.minimum(x, 0)
        p = self._be.maximum(x, 0)
        return ((self._be.exp(n) - 1) * alpha + p) * gamma

    def fn_celu(self, x, *, alpha):
        n = self._be.minimum(x, 0)
        p = self._be.maximum(x, 0)
        return (self._be.exp(n / alpha) - 1) * alpha + p

    def fn_elu(self, x, *, alpha):
        n = self._be.minimum(x, 0)
        p = self._be.maximum(x, 0)
        return (self._be.exp(n) - 1) * alpha + p

    def fn_softmax(self, x, *, axis):
        return self._be.math.softmax(x, axis=axis)

    def fn_log_softmax(self, x, *, axis):
        return self._be.math.log_softmax(x, axis=axis)

    def fn_sigmoid(self, x):
        return self._be.math.sigmoid(x)

    def fn_hard_sigmoid(self, x, *, alpha, beta):
        return self._be.clip_by_value(x * alpha + beta, 0, 1)

    def fn_softplus(self, x):
        return self._be.math.softplus(x)

    def fn_softsign(self, x):
        return self._be.math.softsign(x)

    def fn_silu(self, x):
        return self._be.nn.silu(x)

    # normalizations

    def fn_batch_norm(self, x, *, scale, bias, input_mean, input_var,
                      epsilon):
        rank = len(x.shape)
        ndim = rank - 2
        scale = self.broadcast(scale, rank, ndim)
        bias = self.broadcast(bias, rank, ndim)
        input_mean = self.broadcast(input_mean, rank, ndim)
        input_var = self.broadcast(input_var, rank, ndim)
        return self._be.nn.batch_normalization(x, input_mean, input_var,
                                               bias, scale, epsilon)

    fn_batch_norm1d = fn_batch_norm
    fn_batch_norm2d = fn_batch_norm
    fn_batch_norm3d = fn_batch_norm

    def fn_instance_norm(self, x, *, scale, bias, epsilon):
        rank = len(x.shape)
        ndim = len(x.shape) - 2
        axis = tuple(range(1, ndim + 1) if self.channel_last
                     else range(2, ndim + 2))
        mean = self._be.reduce_mean(x, axis, keepdims=True)
        var = self._be.math.reduce_variance(x, axis, keepdims=True)
        y = (x - mean) / self._be.sqrt(var + epsilon)
        if scale is not None:
            y *= self.broadcast(scale, rank, ndim)
        if bias is not None:
            y += self.broadcast(bias, rank, ndim)
        return y

    fn_instance_norm1d = fn_instance_norm
    fn_instance_norm2d = fn_instance_norm
    fn_instance_norm3d = fn_instance_norm

    # poolings

    def _pool(self, func, ndim, x, *, kernel, stride, padding, dilation,
              ceil_mode, auto_pad):
        assert set(to_int_tuple(dilation, ndim=ndim)) == {1}, \
            f'dilation {dilation} is not supported'
        assert len(x.shape) == ndim + 2, \
            f'invalid input rank {len(x.shape)} for {func}_pool{ndim}d'
        if auto_pad is None:
            dims = x.shape[1:ndim+1] if self.channel_last \
                else x.shape[2:ndim+2]
            auto_pad = to_auto_pad(dims, kernel, stride, padding, dilation)
        fn = getattr(self._be.nn, f'{func}_pool{ndim}d')
        return fn(x, kernel, strides=stride, padding=auto_pad,
                  data_format=self.data_format(ndim))

    def fn_avg_pool1d(self, x, **kwargs):
        return self._pool('avg', 1, x, **kwargs)

    def fn_avg_pool2d(self, x, **kwargs):
        return self._pool('avg', 2, x, **kwargs)

    def fn_avg_pool3d(self, x, **kwargs):
        return self._pool('avg', 3, x, **kwargs)

    fn_avgpool1d = fn_avg_pool1d
    fn_avgpool2d = fn_avg_pool2d
    fn_avgpool3d = fn_avg_pool3d

    def fn_max_pool1d(self, x, **kwargs):
        return self._pool('max', 1, x, **kwargs)

    def fn_max_pool2d(self, x, **kwargs):
        return self._pool('max', 2, x, **kwargs)

    def fn_max_pool3d(self, x, **kwargs):
        return self._pool('max', 3, x, **kwargs)

    fn_maxpool1d = fn_max_pool1d
    fn_maxpool2d = fn_max_pool2d
    fn_maxpool3d = fn_max_pool3d

    def _gpool(self, func, ndim, x):
        dims = x.shape[1:ndim+1] if self.channel_last else x.shape[2:ndim+2]
        fn = getattr(self._be.nn, f'{func}_pool{ndim}d')
        return fn(x, dims, 1, 'VALID', data_format=self.data_format(ndim))

    def fn_global_avg_pool1d(self, x):
        return self._gpool('avg', 1, x)

    def fn_global_avg_pool2d(self, x):
        return self._gpool('avg', 2, x)

    def fn_global_avg_pool3d(self, x):
        return self._gpool('avg', 3, x)

    def fn_global_max_pool1d(self, x):
        return self._gpool('max', 1, x)

    def fn_global_max_pool2d(self, x):
        return self._gpool('max', 2, x)

    def fn_global_max_pool3d(self, x):
        return self._gpool('max', 3, x)

    # matmuls

    def fn_matmul(self, x, *, weight, bias):
        y = self._be.matmul(x, weight if self.channel_last
                            else self._be.transpose(weight))
        if bias is not None:
            y += bias
        return y

    fn_linear = fn_matmul
    fn_fc = fn_matmul

    # convs

    def _conv(self, ndim, x, *, weight, bias, stride, padding, dilation,
              group, auto_pad):
        assert group == 1, f'group {group} is not supported'
        assert len(x.shape) == ndim + 2, \
            f'invalid input rank {len(x.shape)} for conv{ndim}d'
        if auto_pad is not None:
            assert padding is None
            padding = auto_pad
        elif ndim != 2:
            if self.channel_last:
                dims = x.shape[1:ndim+1]
                kernel = weight.shape[:-2]
            else:
                dims = x.shape[2:ndim+2]
                kernel = weight.shape[2:]
            padding = to_auto_pad(dims, kernel, stride, padding, dilation)
        else:
            p = to_int_tuple(padding, ndim=ndim * 2)
            padding = self.to_pads(p)
        fn = getattr(self._be.nn, f'conv{ndim}d')
        y = fn(x, weight, stride, padding, dilations=dilation,
               data_format=self.data_format(ndim))
        if bias is not None:
            y += self.broadcast(bias, len(y.shape), ndim)
        return y

    def fn_conv1d(self, x, **kwargs):
        return self._conv(1, x, **kwargs)

    def fn_conv2d(self, x, **kwargs):
        return self._conv(2, x, **kwargs)

    def fn_conv3d(self, x, *, stride, dilation, **kwargs):
        stride = self.to_axes(1, 1, to_int_tuple(stride, ndim=3))
        dilation = self.to_axes(1, 1, to_int_tuple(dilation, ndim=3))
        return self._conv(3, x, stride=stride, dilation=dilation, **kwargs)

    def _conv_t(self, ndim, x, *, weight, bias, stride, padding, dilation,
                output_padding, group, auto_pad):
        assert group == 1, f'group {group} is not supported'
        if self.channel_last:
            ba, *xd, ci = x.shape
            *kernel, co, kci = weight.shape
        else:
            ba, ci, *xd = x.shape
            kci, co, *kernel = weight.shape
        assert ci == kci * group, \
            f'invalid input shape {x.shape} with kernel {weight.shape}'
        os, (_, _, p, *_) = \
            conv_t_shapes(xd, kernel, stride, padding, dilation,
                          output_padding, auto_pad)
        if auto_pad is not None:
            assert padding is None
            padding = auto_pad
        elif ndim != 2:
            if self.channel_last:
                dims = x.shape[1:ndim+1]
                kernel = weight.shape[:-2]
            else:
                dims = x.shape[2:ndim+2]
                kernel = weight.shape[2:]
            padding = to_auto_pad(dims, kernel, stride, padding, dilation)
        else:
            padding = self.to_pads(p)
        fn = getattr(self._be.nn, f'conv{ndim}d_transpose')
        y = fn(x, weight, self.to_axes(ba, co, os), strides=stride,
               padding=padding, dilations=dilation,
               data_format=self.data_format(ndim))
        if bias is not None:
            y += self.broadcast(bias, len(y.shape), ndim)
        return y

    def fn_conv_transpose1d(self, x, **kwargs):
        return self._conv_t(1, x, **kwargs)

    def fn_conv_transpose2d(self, x, **kwargs):
        return self._conv_t(2, x, **kwargs)

    def fn_conv_transpose3d(self, x, **kwargs):
        return self._conv_t(3, x, **kwargs)

    # resize

    def fn_resize(self, x, *, size, scale, mode):
        if self.channel_last:
            ba, *xd, ci = x.shape
        else:
            ba, ci, *xd = x.shape
        ndim = len(xd)
        _, sc = resize_size_scale(xd, size, scale)
        scale = tuple(map(int, sc))
        assert all(a == b and a >= 1 for a, b in zip(sc, scale)), \
            f'resize scale {sc} is not supported'
        if ndim == 1:
            scale = scale[0]
        mod = getattr(self._be.keras.layers, f'UpSampling{ndim}D')
        return mod(scale)(x)

    # reducemean

    def fn_reducemean(self, x, *, axes, keepdims):
        return self._be.math.reduce_mean(x, axis=axes, keepdims=keepdims)

    # split

    def fn_split(self, x, *, axis, split, with_batch):
        if not with_batch:
            axis += 1
        secs = split_to_secs(split, x.shape[axis])
        return self._be.split(x, secs, axis=axis)

from .math import MathRuntime
from ..core.type_util import to_int_tuple
from .utils import auto_pad_pool, concat_axis, split_to_secs

class TorchRuntime(MathRuntime):
    name = 'torch'
    channel_last = False
    tensor_is_readonly = False

    def init_backend(self):
        import torch
        self._be = torch
        self._fn = torch.nn.functional
        self.tensor_type = torch.Tensor

    def rand(self, shape):
        return self._be.rand(shape)

    def as_tensor(self, data, dtype=None):
        if isinstance(dtype, str):
            dtype = getattr(self._be, dtype)
        return self._be.as_tensor(data, dtype=dtype)

    def fn_concat(self, *x, axis, with_batch, channel_pos):
        axis = concat_axis(len(x[0].shape) - 1, self.channel_last, axis, with_batch, channel_pos)
        return self._be.cat(x, dim=axis + 1)

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

    def fn_batch_norm(self, x, *, scale, bias, input_mean, input_var, epsilon):
        return self._fn.batch_norm(x, input_mean, input_var, weight=scale, bias=bias, eps=epsilon)
    fn_batch_norm1d = fn_batch_norm
    fn_batch_norm2d = fn_batch_norm
    fn_batch_norm3d = fn_batch_norm

    def fn_instance_norm(self, x, *, scale, bias, epsilon):
        return self._fn.instance_norm(x, weight=scale, bias=bias, eps=epsilon)
    fn_instance_norm1d = fn_instance_norm
    fn_instance_norm2d = fn_instance_norm
    fn_instance_norm3d = fn_instance_norm

    def _avg_pool(self, ndim, x, *, kernel, stride, padding, dilation, ceil_mode, auto_pad):
        assert set(to_int_tuple(dilation, ndim=ndim)) == {1}, f'dilation {dilation} is not supported'
        if padding is None:
            dims = x.shape[1:ndim + 1] if self.channel_last else x.shape[2:ndim + 2]
            padding = auto_pad_pool(auto_pad, dims, kernel, stride, dilation)
        fn = getattr(self._fn, f'avg_pool{ndim}d')
        return fn(x, kernel, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=False)

    def fn_avg_pool1d(self, x, **kwargs):
        return self._avg_pool(1, x, **kwargs)

    def fn_avg_pool2d(self, x, **kwargs):
        return self._avg_pool(2, x, **kwargs)

    def fn_avg_pool3d(self, x, **kwargs):
        return self._avg_pool(3, x, **kwargs)
    fn_avgpool1d = fn_avg_pool1d
    fn_avgpool2d = fn_avg_pool2d
    fn_avgpool3d = fn_avg_pool3d

    def _max_pool(self, ndim, x, *, kernel, stride, padding, dilation, ceil_mode, auto_pad):
        if padding is None:
            dims = x.shape[1:ndim + 1] if self.channel_last else x.shape[2:ndim + 2]
            padding = auto_pad_pool(auto_pad, dims, kernel, stride, dilation)
        fn = getattr(self._fn, f'max_pool{ndim}d')
        return fn(x, kernel, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)

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

    def fn_matmul(self, x, *, weight, bias):
        y = self._fn.linear(x, weight, bias=bias)
        return y
    fn_linear = fn_matmul
    fn_fc = fn_matmul

    def _conv(self, ndim, x, *, weight, bias, stride, padding, dilation, group, auto_pad):
        fn = getattr(self._fn, f'conv{ndim}d')
        return fn(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=group)

    def fn_conv1d(self, x, **kwargs):
        return self._conv(1, x, **kwargs)

    def fn_conv2d(self, x, **kwargs):
        return self._conv(2, x, **kwargs)

    def fn_conv3d(self, x, **kwargs):
        return self._conv(3, x, **kwargs)

    def _conv_t(self, ndim, x, *, weight, bias, stride, padding, dilation, output_padding, group, auto_pad):
        fn = getattr(self._fn, f'conv_transpose{ndim}d')
        return fn(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding, groups=group)

    def fn_conv_transpose1d(self, x, **kwargs):
        return self._conv_t(1, x, **kwargs)

    def fn_conv_transpose2d(self, x, **kwargs):
        return self._conv_t(2, x, **kwargs)

    def fn_conv_transpose3d(self, x, **kwargs):
        return self._conv_t(3, x, **kwargs)

    def fn_resize(self, x, *, size, scale, mode):
        if size is not None:
            return self._fn.interpolate(x, size, mode=mode)
        else:
            return self._fn.interpolate(x, size, scale, mode=mode, recompute_scale_factor=True)

    def fn_reducemean(self, x, *, axes, keepdims):
        return self._be.mean(x, dim=axes, keepdim=keepdims)

    def fn_split(self, x, *, axis, split, with_batch):
        if not with_batch:
            axis += 1
        secs = split_to_secs(split, x.shape[axis])
        return self._be.split(x, secs, dim=axis)
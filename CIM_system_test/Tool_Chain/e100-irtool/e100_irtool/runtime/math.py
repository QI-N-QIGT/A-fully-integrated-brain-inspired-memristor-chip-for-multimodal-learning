from .base import BaseRuntime
import math


class MathRuntime(BaseRuntime):

    _be = math

    # constant, identity

    def fn_constant(self, *, value):
        return value

    def fn_identity(self, x):
        return x

    # sign, abs, neg, ceil, floor

    def fn_sign(self, x):
        try:
            return self._be.sign(x)
        except AttributeError:
            return -1 if x < 0 else 1 if x > 0 else 0

    def fn_abs(self, x):
        return abs(x)

    def fn_neg(self, x):
        try:
            return self._be.negative(x)
        except AttributeError:
            return -x

    def fn_ceil(self, x):
        return self._be.ceil(x)

    def fn_floor(self, x):
        return self._be.floor(x)

    # add, sub, mul, div, mod, pow

    def fn_add(self, x, y):
        try:
            return self._be.add(x, y)
        except AttributeError:
            return x + y

    def fn_sub(self, x, y):
        try:
            return self._be.subtract(x, y)
        except AttributeError:
            return x - y

    def fn_mul(self, x, y):
        try:
            return self._be.multiply(x, y)
        except AttributeError:
            return x * y

    def fn_div(self, x, y):
        try:
            return self._be.divide(x, y)
        except AttributeError:
            return x / y

    def fn_mod(self, x, y):
        try:
            return self._be.remainder(x, y)
        except AttributeError:
            return x % y

    def fn_pow(self, x, y):
        try:
            return self._be.pow(x, y)
        except AttributeError:
            return x ** y

    # exp, log, sqrt

    def fn_exp(self, x):
        return self._be.exp(x)

    def fn_log(self, x):
        return self._be.log(x)

    def fn_sqrt(self, x):
        return self._be.sqrt(x)

    # sin, cos, tan

    def fn_sin(self, x):
        return self._be.sin(x)

    def fn_cos(self, x):
        return self._be.cos(x)

    def fn_tan(self, x):
        return self._be.tan(x)

    # asin, acos, atan

    def fn_asin(self, x):
        try:
            return self._be.arcsin(x)
        except AttributeError:
            return self._be.asin(x)

    def fn_acos(self, x):
        try:
            return self._be.arccos(x)
        except AttributeError:
            return self._be.acos(x)

    def fn_atan(self, x):
        try:
            return self._be.arctan(x)
        except AttributeError:
            return self._be.atan(x)

    # sinh, cosh, tanh

    def fn_sinh(self, x):
        return self._be.sinh(x)

    def fn_cosh(self, x):
        return self._be.cosh(x)

    def fn_tanh(self, x):
        return self._be.tanh(x)

    # asinh, acosh, atanh

    def fn_asinh(self, x):
        try:
            return self._be.arcsinh(x)
        except AttributeError:
            return self._be.asinh(x)

    def fn_acosh(self, x):
        try:
            return self._be.arccosh(x)
        except AttributeError:
            return self._be.acosh(x)

    def fn_atanh(self, x):
        try:
            return self._be.arctanh(x)
        except AttributeError:
            return self._be.atanh(x)

    # logical not, and, or, xor

    def fn_logical_not(self, x):
        try:
            return self._be.logical_not(x)
        except AttributeError:
            return not bool(x)

    def fn_logical_and(self, x, y):
        try:
            return self._be.logical_and(x, y)
        except AttributeError:
            return bool(x) and bool(y)

    def fn_logical_or(self, x, y):
        try:
            return self._be.logical_or(x, y)
        except AttributeError:
            return bool(x) or bool(y)

    def fn_logical_xor(self, x, y):
        try:
            return self._be.logical_xor(x, y)
        except AttributeError:
            return bool(x) ^ bool(y)

    # bitwise not, and, or, xor

    def fn_bitwise_not(self, x):
        try:
            return self._be.bitwise_not(x)
        except AttributeError:
            return ~x

    def fn_bitwise_and(self, x, y):
        try:
            return self._be.bitwise_and(x, y)
        except AttributeError:
            return x & y

    def fn_bitwise_or(self, x, y):
        try:
            return self._be.bitwise_or(x, y)
        except AttributeError:
            return x | y

    def fn_bitwise_xor(self, x, y):
        try:
            return self._be.bitwise_xor(x, y)
        except AttributeError:
            return x ^ y

    # equal, less, greater

    def fn_equal(self, x, y):
        try:
            return self._be.equal(x, y)
        except AttributeError:
            return x == y

    def fn_less(self, x, y):
        try:
            return self._be.less(x, y)
        except AttributeError:
            return x < y

    def fn_less_or_equal(self, x, y):
        try:
            return self._be.less_or_equal(x, y)
        except AttributeError:
            return x <= y

    def fn_greater(self, x, y):
        try:
            return self._be.greater(x, y)
        except AttributeError:
            return x > y

    def fn_greater_or_equal(self, x, y):
        try:
            return self._be.greater_or_equal(x, y)
        except AttributeError:
            return x >= y

    # utils

    def broadcast(self, x, rank, ndim=None):
        if x is None or len(x.shape) >= rank:
            return x
        n = rank - len(x.shape)
        if ndim is None:
            ndim = n
        if n >= ndim:
            if self.channel_last:
                return x[(Ellipsis, *((None,) * ndim), slice(None))]
            else:
                return x[(Ellipsis, *((None,) * ndim))]
        assert False, f'can\'t broadcast {x.shape} to rank {rank}(ndim={ndim})'

    def to_axes(self, batch, channel, dims):
        if self.channel_last:
            return (batch, *dims, channel)
        else:
            return (batch, channel, *dims)

    def to_pads(self, paddings):
        p = paddings
        ndim = len(p) // 2
        return self.to_axes((0, 0), (0, 0), ((p[i], p[ndim + i])
                            for i in range(ndim)))

    def to_slices(self, dims):
        return self.to_axes(slice(None), slice(None), dims)

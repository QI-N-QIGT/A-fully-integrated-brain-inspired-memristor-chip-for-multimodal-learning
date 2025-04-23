from .numpy import NumpyRuntime
import numpy
from typing import Callable
from ..core.op import enum_op_ids


class RTTests:

    epsilon = 1e-7
    atol = rtol = 1e-4

    @classmethod
    def setup_other(cls):
        pass

    @classmethod
    def setUpClass(cls):
        cls.other = cls.setup_other()
        cls.numpy = NumpyRuntime(channel_last=cls.other.channel_last)

    def _rand(self, shape=(), signed=True):
        x = numpy.random.random(shape)
        if signed:
            x = (x - .5) * 10.0
        if shape:
            return numpy.array(x, dtype=numpy.float32)
        else:
            return float(x) + self.epsilon

    def t(self, op_id, *in_shapes, batch=9, channel=None, **kwargs):
        x1 = [self._rand((batch, *s) if channel is None
                         else self.numpy.to_axes(batch, channel, s))
              for s in in_shapes]
        x2 = [self.other.as_tensor(x) for x in x1]
        kwargs2 = dict(kwargs)
        for k, v in kwargs.items():
            if isinstance(v, numpy.ndarray):
                kwargs2[k] = self.other.as_tensor(v)
        y1 = self.numpy.run_op(op_id, *x1, **kwargs)
        y2 = self.other.run_op(op_id, *x2, **kwargs2)
        if op_id == 'split':
            y2 = [yy.numpy() for yy in y2]
            self.assertEqual(len(y1), len(y2))
            for yy1, yy2 in zip(y1, y2):
                self.assertEqual(yy1.shape, yy2.shape)
                self.assertTrue(numpy.all(yy1 == yy2))
        else:
            y2 = y2.numpy()
            self.assertEqual(y1.shape, y2.shape)
            self.assertTrue(numpy.allclose(y1, y2, atol=self.atol,
                                           rtol=self.rtol),
                            msg=f'\nx\n:{x1}\nnumpy:\n{y1}\n'
                                f'{self.other.name}\n:{y2}\n')

    def test_op_coverage(self):
        rts = [self.numpy, self.other]
        for op_id in enum_op_ids():
            fn = f'fn_{op_id}'
            for rt in rts:
                f = getattr(rt, fn, None)
                self.assertIsNotNone(
                    f, msg=f'{type(rt).__qualname__}.{fn} not exist')
                self.assertTrue(isinstance(f, Callable))

    def test_transes(self):
        self.t('reshape', [3, 4], shape=[2, 6], with_batch=False)
        self.t('reshape', [3, 4], shape=[-1, 2, 6], with_batch=True)
        self.t('flatten', [3, 4], start_dim=0, with_batch=False)
        self.t('flatten', [3, 4], start_dim=1, with_batch=True)
        self.t('transpose', [2, 3, 4], perm=[2, 0, 3, 1], with_batch=True)
        self.t('transpose', [2, 3, 4], perm=[2, 0, 1], with_batch=False)
        self.t('pad', [2, 3], pads=[1, 0], value=1)
        self.t('pad', [2, 3, 4], pads=[0, 1, 1, 0], value=2)
        self.t('pad', [2, 3, 4, 5], pads=[0, 1, 0, 1, 0, 0], value=3)

    def test_activations(self):
        s = [8, 11]
        self.t('relu', s)
        self.t('leaky_relu', s, alpha=self._rand())
        self.t('prelu', s, slope=self._rand((s[self.numpy.channel_last],)))
        self.t('selu', s, alpha=self._rand(), gamma=self._rand())
        self.t('celu', s, alpha=self._rand())
        self.t('elu', s, alpha=self._rand())
        self.t('softmax', s, axis=-1)
        self.t('log_softmax', s, axis=-1)
        self.t('sigmoid', s)
        self.t('hard_sigmoid', s, alpha=self._rand(), beta=self._rand())
        self.t('softplus', s)
        self.t('softsign', s)
        self.t('silu', s)

    def test_norms(self):
        c, d, h, w = 7, 5, 4, 3

        def run(name):
            self.t(f'{name}_norm1d', [w], **kwargs)
            self.t(f'{name}_norm2d', [h, w], **kwargs)
            self.t(f'{name}_norm3d', [d, h, w], **kwargs)

        kwargs = dict(scale=None, bias=None, epsilon=self.epsilon, channel=c)
        run('instance')
        kwargs.update(scale=self._rand([c]), bias=self._rand([c]))
        run('instance')
        kwargs.update(input_mean=self._rand([c]),
                      input_var=self._rand([c], signed=False))
        run('batch')

    def test_global_pool(self):
        c, d, h, w = 7, 5, 4, 3
        self.t('global_avg_pool1d', [w], channel=c)
        self.t('global_max_pool1d', [w], channel=c)
        self.t('global_avg_pool2d', [h, w], channel=c)
        self.t('global_max_pool2d', [h, w], channel=c)
        self.t('global_avg_pool3d', [d, h, w], channel=c)
        self.t('global_max_pool3d', [d, h, w], channel=c)

    def test_matmul(self):
        ci, co = 11, 7
        kwargs = dict(weight=self._rand([ci, co] if self.numpy.channel_last
                      else [co, ci]), bias=self._rand([co]))
        self.t('matmul', [ci], **kwargs)

    def test_split(self):
        self.t('split', [4, 6], axis=1, split=2, with_batch=True)
        self.t('split', [4, 6], axis=1, split=[1, 2, 1], with_batch=True)
        self.t('split', [4, 6], axis=0, split=2, with_batch=False)
        self.t('split', [4, 6], axis=0, split=[1, 2, 1], with_batch=False)

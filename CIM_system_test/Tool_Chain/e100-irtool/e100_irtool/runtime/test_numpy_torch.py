from unittest import TestCase, main
from .rt_tests import RTTests
from .torch import TorchRuntime


class TestNumpyTorchRuntime(RTTests, TestCase):

    @classmethod
    def setup_other(cls):
        return TorchRuntime()

    def test_concat(self):
        self.t('concat', [2], [3], axis=1, with_batch=True,
               channel_pos='first')
        self.t('concat', [2], [3], axis=0, with_batch=False,
               channel_pos='first')
        self.t('concat', [3, 2], [3, 3], [3, 1], axis=1, with_batch=False,
               channel_pos='first')
        self.t('concat', [3, 2], [3, 3], [3, 1], axis=2, with_batch=True,
               channel_pos='first')
        self.t('concat', [3, 2], [3, 3], [3, 1], axis=0, with_batch=False,
               channel_pos='last')
        self.t('concat', [3, 2], [3, 3], [3, 1], axis=1, with_batch=True,
               channel_pos='last')

    def test_pools(self):
        c, d, h, w = 7, 5, 4, 3

        def run():
            self.t('max_pool1d', [w], **kwargs)
            self.t('max_pool2d', [h, w], **kwargs)
            self.t('max_pool3d', [d, h, w], **kwargs)
            if kwargs['dilation'] != 1:
                return
            self.t('avg_pool1d', [w], **kwargs)
            self.t('avg_pool2d', [h, w], **kwargs)
            self.t('avg_pool3d', [d, h, w], **kwargs)

        kwargs = dict(kernel=2, stride=2, padding=0, dilation=1,
                      ceil_mode=False, auto_pad=None, channel=c)
        run()
        kwargs.update(padding=1)
        run()
        kwargs.update(dilation=2, padding=0)
        run()
        kwargs.update(dilation=1, padding=1, ceil_mode=True)
        run()
        kwargs.update(padding=None, auto_pad='VALID')
        run()
        kwargs.update(padding=None, auto_pad='SAME')
        run()
        kwargs.update(padding=None, auto_pad='SAME', ceil_mode=True)
        run()

    def test_conv(self):
        ci, co, d, h, w = 11, 9, 7, 6, 5
        k1, k2, k3 = 3, 2, 1

        def run():
            self.t('conv1d', [w],
                   weight=self._rand([co, ci, k1]),
                   bias=self._rand([co]), **kwargs)
            self.t('conv2d', [h, w],
                   weight=self._rand([co, ci, k1, k2]),
                   bias=self._rand([co]), **kwargs)
            self.t('conv3d', [d, h, w],
                   weight=self._rand([co, ci, k1, k2, k3]),
                   bias=self._rand([co]), **kwargs)

        kwargs = dict(stride=1, padding=1, dilation=1, group=1, channel=ci,
                      auto_pad=None)
        run()
        kwargs.update(padding=0)
        run()
        kwargs.update(dilation=2)
        run()
        kwargs.update(stride=2, padding=1, dilation=2)
        run()

    def test_conv_transpose(self):
        ci, co, d, h, w = 11, 9, 7, 6, 5
        k1, k2, k3 = 3, 2, 2

        def run():
            self.t('conv_transpose1d', [w],
                   weight=self._rand([ci, co, k1]),
                   bias=self._rand([co]), **kwargs)
            self.t('conv_transpose2d', [h, w],
                   weight=self._rand([ci, co, k1, k2]),
                   bias=self._rand([co]), **kwargs)
            self.t('conv_transpose3d', [d, h, w],
                   weight=self._rand([ci, co, k1, k2, k3]),
                   bias=self._rand([co]), **kwargs)

        kwargs = dict(stride=1, padding=1, dilation=1, group=1,
                      output_padding=0, channel=ci, auto_pad=None)
        run()
        kwargs.update(padding=0)
        run()
        kwargs.update(dilation=2, stride=2, padding=1, output_padding=1)
        run()

    def test_resize(self):
        c, d, h, w = 2, 5, 6, 7

        kwargs = dict(mode='nearest', size=None)
        self.t('resize', [c, w], scale=2, **kwargs)
        self.t('resize', [c, h, w], scale=2, **kwargs)
        self.t('resize', [c, d, h, w], scale=2, **kwargs)
        self.t('resize', [c, d, h, w], scale=[1.2, 2.1, 0.7], **kwargs)

        kwargs.pop('size')
        kwargs.update(scale=None)
        self.t('resize', [c, w], size=11, **kwargs)
        self.t('resize', [c, h, w], size=[13, 20], **kwargs)
        self.t('resize', [c, d, h, w], size=12, **kwargs)
        self.t('resize', [c, d, h, w], size=[3, 4, 5], **kwargs)


if __name__ == '__main__':
    main()

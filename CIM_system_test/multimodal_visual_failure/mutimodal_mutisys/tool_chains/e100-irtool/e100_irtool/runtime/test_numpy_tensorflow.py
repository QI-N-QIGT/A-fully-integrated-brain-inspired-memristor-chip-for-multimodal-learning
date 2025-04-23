from unittest import TestCase, main
from .rt_tests import RTTests
from .tensorflow import TensorflowRuntime

class TestNumpyTensorflowRuntime(RTTests, TestCase):

    @classmethod
    def setup_other(cls):
        return TensorflowRuntime()

    def test_concat(self):
        self.t('concat', [2], [3], axis=1, with_batch=True, channel_pos='first')
        self.t('concat', [2], [3], axis=0, with_batch=False, channel_pos='first')
        self.t('concat', [2, 3], [3, 3], [1, 3], axis=-2, with_batch=True, channel_pos='last')
        self.t('concat', [2, 3], [3, 3], [1, 3], axis=-2, with_batch=False, channel_pos='last')
        self.t('concat', [2, 3], [3, 3], [1, 3], axis=1, with_batch=True, channel_pos='last')
        self.t('concat', [2, 3], [3, 3], [1, 3], axis=1, with_batch=False, channel_pos='first')
        self.t('concat', [2, 3], [3, 3], [1, 3], axis=2, with_batch=True, channel_pos='first')
        self.t('concat', [2, 3], [3, 3], [1, 3], axis=-1, with_batch=True, channel_pos='first')
        self.t('concat', [2, 3], [3, 3], [1, 3], axis=1, with_batch=True, channel_pos='ignore')

    def test_pools(self):
        (c, d, h, w) = (7, 5, 4, 3)

        def run():
            self.t('avg_pool1d', [w], **kwargs)
            self.t('max_pool1d', [w], **kwargs)
            self.t('avg_pool2d', [h, w], **kwargs)
            self.t('max_pool2d', [h, w], **kwargs)
            self.t('avg_pool3d', [d, h, w], **kwargs)
            self.t('max_pool3d', [d, h, w], **kwargs)
        kwargs = dict(kernel=2, stride=2, padding=0, dilation=1, ceil_mode=False, auto_pad=None, channel=c)
        run()
        kwargs.update(padding=None, auto_pad='VALID')
        run()
        kwargs.update(padding=None, auto_pad='SAME')
        run()
        kwargs.update(padding=None, auto_pad='SAME', ceil_mode=True)
        run()
        kwargs.update(padding=[0, 1], auto_pad=None, ceil_mode=False)
        self.t('avg_pool1d', [w], **kwargs)
        self.t('max_pool1d', [w], **kwargs)
        kwargs.update(padding=[0, 0, 0, 1])
        self.t('avg_pool2d', [h, w], **kwargs)
        self.t('max_pool2d', [h, w], **kwargs)
        kwargs.update(padding=[0, 0, 0, 1, 0, 1])
        self.t('avg_pool3d', [d, h, w], **kwargs)
        self.t('max_pool3d', [d, h, w], **kwargs)

    def test_conv(self):
        (ci, co, *dims) = (11, 9, 7, 6, 5)
        k = (3, 2, 1)

        def run(*d):
            for i in d:
                self.t(f'conv{i}d', dims[-i:], weight=self._rand([*k[:i], ci, co]), bias=self._rand([co]), **kwargs)
        kwargs = dict(stride=1, dilation=1, group=1, channel=ci, padding=None, auto_pad='VALID')
        run(1, 2, 3)
        kwargs.update(auto_pad='SAME')
        run(1, 2, 3)
        kwargs.update(stride=2)
        run(1, 2, 3)
        kwargs.update(stride=2, auto_pad='VALID')
        run(1, 2, 3)
        kwargs.update(dilation=2)
        run(1, 2)
        kwargs.update(auto_pad='SAME')
        run(1, 2)
        kwargs.update(stride=2, dilation=1, padding=[1, 0, 1, 1], auto_pad=None)
        run(2)

    def test_conv_transpose(self):
        (ci, co, *dims) = (11, 9, 7, 6, 5)
        k = (3, 2, 2)

        def run(*d):
            for i in d:
                self.t(f'conv_transpose{i}d', dims[-i:], weight=self._rand([*k[:i], co, ci]), bias=self._rand([co]), **kwargs)
        kwargs = dict(stride=1, padding=None, dilation=1, group=1, output_padding=0, channel=ci, auto_pad='VALID')
        run(1, 2, 3)
        kwargs.update(auto_pad='SAME')
        run(1, 2, 3)
        kwargs.update(stride=2)
        run(1, 2, 3)
        kwargs.update(stride=2, auto_pad='VALID')
        run(1, 2, 3)
        kwargs.update(stride=2, padding=[1, 0, 1, 1], auto_pad=None)
        run(2)
        kwargs.update(stride=[1, 2], padding=[1, 0, 1, 1], auto_pad=None)
        run(2)

    def test_resize(self):
        (c, d, h, w) = (2, 5, 6, 7)
        kwargs = dict(mode='nearest', size=None)
        self.t('resize', [w, c], scale=2, **kwargs)
        self.t('resize', [h, w, c], scale=2, **kwargs)
        self.t('resize', [d, h, w, c], scale=2, **kwargs)
        kwargs.pop('size')
        kwargs.update(scale=None)
        self.t('resize', [w, c], size=w * 2, **kwargs)
        self.t('resize', [h, w, c], size=[h * 1, w * 2], **kwargs)
        self.t('resize', [d, h, w, c], size=[d * 1, h * 2, w * 3], **kwargs)
if __name__ == '__main__':
    main()
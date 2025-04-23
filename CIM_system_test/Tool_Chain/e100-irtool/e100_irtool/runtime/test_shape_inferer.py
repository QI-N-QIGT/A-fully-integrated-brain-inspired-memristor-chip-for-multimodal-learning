from unittest import TestCase, main
from .shape_inferer import ShapeInferer
from .numpy import NumpyRuntime
from ..core.op import make_op, enum_op_ids


class TestShapeInferer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runtime = NumpyRuntime(channel_last=False)
        cls.inferer = ShapeInferer()

    def t(self, op_id, *shapes, signed=True, **kwargs):
        si, rt = self.inferer, self.runtime
        op = make_op(op_id, **kwargs)
        ins = [rt.rand((1, *s)) for s in shapes]
        wts = {k: rt.rand(s) for k, s in op.weight_shapes().items()}
        if signed:
            for v in ins:
                v -= .5
                v *= 2
            for v in wts.values():
                v -= .5
                v *= 2
        if op_id in ('acosh',):
            for v in ins:
                v += 1.0
                v *= 10
        if op_id in ('pow',):
            ins[0] += 1.0
        if op_id[:8] in ('logical_', 'bitwise_'):
            dtype = 'int32'
            for v in ins:
                v *= 100
        else:
            dtype = 'float32'
        ins = [rt.as_tensor(v, dtype=dtype) for v in ins]
        wts = {k: rt.as_tensor(v, dtype=dtype) for k, v in wts.items()}
        out = rt.run_op(op.op_id, *ins, **wts, **op.get_attrs())
        if op.op_id == 'constant':
            y1 = [out.shape]
        elif isinstance(out, (tuple, list)):
            y1 = [o.shape[1:] for o in out]
        else:
            y1 = [out.shape[1:]]
        y2 = si.infer_op(op, *shapes, channel_last=rt.channel_last)
        self.assertEqual(y1, y2)

    def test_op_coverage(self):
        op_ids = set().union(*(v for v in self.inferer.all_ops.values()))
        left = tuple(set(enum_op_ids()) - op_ids)
        self.assertFalse(left, f'{left} ops not covered')

    def test_const(self):
        for op_id in ShapeInferer.all_ops['const']:
            self.assertEqual(make_op(op_id, value=1).num_inputs, 0)
        self.t('constant', value=self.runtime.rand([2, 3]))

    def test_unary(self):
        for op_id in ShapeInferer.all_ops['unary']:
            self.assertEqual(make_op(op_id).num_inputs, 1)
            if op_id in ('log', 'sqrt', 'acosh'):
                self.t(op_id, (2, 3), signed=False)
            else:
                self.t(op_id, (2, 3))

    def test_binary(self):
        for op_id in ShapeInferer.all_ops['binary']:
            self.assertEqual(make_op(op_id).num_inputs, 2)
            self.t(op_id, (2, 3), (2, 3))
            self.t(op_id, (2, 3), (2, 1))
            self.t(op_id, (2, 3), (3,))
            self.t(op_id, (2,), (3, 2))

    def test_concat(self):
        self.assertIsNone(make_op('concat', axis=0).num_inputs)
        self.t('concat', (2, 3), (2, 1), (2, 2), axis=1, with_batch=False)
        self.t('concat', (2, 3), (2, 1), (2, 2), axis=2)
        self.t('concat', (3, 2), (1, 2), (2, 2), axis=1, with_batch=False,
               channel_pos="last")
        self.t('concat', (3, 2), (1, 2), (2, 2), axis=-1, with_batch=False,
               channel_pos="last")
        self.t('concat', (3, 2), (1, 2), (2, 2), axis=-1,
               channel_pos="last")
        self.t('concat', (2, 3), (2, 1), (2, 2), axis=1, with_batch=False,
               channel_pos="ignore")

    def test_trans(self):
        self.assertEqual(make_op('reshape', shape=(1,)).num_inputs, 1)
        self.t('reshape', (2, 3, 4), shape=(-1, 3, 2, 2, 2))
        self.t('reshape', (2, 3, 4), shape=(3, 2, 2, 2), with_batch=False)
        self.assertEqual(make_op('flatten').num_inputs, 1)
        self.t('flatten', (2, 3, 4), start_dim=1)
        self.t('flatten', (2, 3, 4), start_dim=0, with_batch=False)
        self.assertEqual(make_op('transpose', perm=(1, 0)).num_inputs, 1)
        self.t('transpose', (2, 3, 4), perm=(0, 2, 1, 3))
        self.t('transpose', (2, 3, 4), perm=(2, 1, 0), with_batch=False)
        self.t('pad', (2, 3), pads=(0, 1))
        self.t('pad', (2, 3), pads=(1, 1))
        self.t('pad', (2, 3, 4), pads=(0, 1, 0, 1))
        self.t('pad', (2, 3, 4), pads=(1, 0, 0, 1))

    def test_norm(self):
        for op_id in ShapeInferer.all_ops['norm']:
            self.assertEqual(make_op(op_id, channel=2).num_inputs, 1)
        self.t('batch_norm', (2, 3, 4, 5, 6), channel=2, signed=False)
        self.t('batch_norm1d', (2, 3), channel=2, signed=False)
        self.t('batch_norm2d', (2, 3, 4), channel=2, signed=False)
        self.t('batch_norm3d', (2, 3, 4, 5), channel=2, signed=False)
        self.t('instance_norm1d', (2, 3), channel=2)
        self.t('instance_norm2d', (2, 3, 4), channel=2)
        self.t('instance_norm3d', (2, 3, 4, 5), channel=2)

    def test_pool(self):
        for op_id in ShapeInferer.all_ops['pool']:
            self.assertEqual(make_op(op_id, kernel=2).num_inputs, 1)
        self.t('avg_pool2d', (2, 4, 5), kernel=(2, 3), stride=(2, 3))
        self.t('maxpool1d', (4, 5), kernel=3, dilation=2, padding=1)
        self.t('global_max_pool3d', (3, 4, 5, 6))

    def test_matmul(self):
        kwargs = dict(in_channel=3, out_channel=4)
        for op_id in ShapeInferer.all_ops['matmul']:
            self.assertEqual(make_op(op_id, **kwargs).num_inputs, 1)
        self.t('fc', (3,), **kwargs)
        self.t('fc', (3,), bias=False, **kwargs)

    def test_conv(self):
        kwargs = dict(in_channel=2, out_channel=3, kernel=3)
        for op_id in ShapeInferer.all_ops['conv']:
            self.assertEqual(make_op(op_id, **kwargs).num_inputs, 1)
        self.t('conv1d', (2, 3), padding=1, **kwargs)
        self.t('conv1d', (2, 3), bias=False, **kwargs)
        self.t('conv2d', (2, 3, 4), stride=2, padding=1, **kwargs)
        self.t('conv2d', (2, 3, 4), padding=1, dilation=2, **kwargs)
        self.t('conv3d', (2, 3, 4, 5), **kwargs)

    def test_conv_transpose(self):
        kwargs = dict(in_channel=2, out_channel=3, kernel=3)
        for op_id in ShapeInferer.all_ops['conv_transpose']:
            self.assertEqual(make_op(op_id, **kwargs).num_inputs, 1)
        self.t('conv_transpose1d', (2, 5), padding=1, **kwargs)
        self.t('conv_transpose2d', (2, 4, 5), padding=1, **kwargs)
        self.t('conv_transpose3d', (2, 4, 5, 6), stride=2, **kwargs)
        self.t('conv_transpose2d', (2, 4, 5), dilation=2, output_padding=1,
               **kwargs)

    def test_resize(self):
        self.t('resize', (2, 5), size=(10,))
        self.t('resize', (2, 5, 6), scale=2)

    def test_split(self):
        self.t('split', (4, 6), axis=1, split=3, with_batch=False)
        self.t('split', (2, 4, 6), axis=2, split=3)
        self.t('split', (2, 4, 6), axis=2, split=3)
        self.t('split', (2, 4, 6), axis=2, split=[1, 2, 1])
        self.t('split', (2, 4, 6), axis=1, split=[1, 2, 1], with_batch=False)


if __name__ == '__main__':
    main()

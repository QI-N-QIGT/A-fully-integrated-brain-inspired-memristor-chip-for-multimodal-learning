#!/usr/bin/env python
import sys
import click
from pathlib import Path
from e100_irtool import BaseIR, make_op, make_layer, UnaryOp
from e100_irtool.core.type_util import is_integer
from e100_irtool.cmd.data import load_pickle, save_pickle
from e100_irtool.tools import flatten_layers    # noqa

ANCHORS = {
    'yolov3': ((10, 13, 16, 30, 33, 23),
               (30, 61, 62, 45, 59, 119),
               (116, 90, 156, 198, 373, 326)),
    'yolov3-tiny': ((10, 14, 23, 27, 37, 58),
                    (81, 82, 135, 169, 344, 319)),
}

STRIDES = {
    'yolov3': (8, 16, 32),
    'yolov3-tiny': (16, 32),
}


class Yolov3XywhOp(UnaryOp):

    op_id = 'yolov3-xywh'
    attrs = ('nclass', 'anchors', 'stride')
    nclass = None
    anchors = None
    stride = None

    def __init__(self, *, nclass, anchors, stride=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('nclass', nclass, is_integer, min_val=1)
        self.set_attr('anchors', anchors)
        self.set_attr('stride', stride, is_integer, min_val=1)
        self._grid = None
        self._anchor_grid = None

    def infer_shape(self, x, *, channel_last):
        no, na, ny, nx, ci, co = self._shapes(x, self.nclass, self.anchors,
                                              channel_last)
        return (co, no)

    def _shapes(self, x, nclass, anchors, channel_last):
        no = nclass + 5
        na = len(anchors) // 2
        if channel_last:
            ny, nx, ci = x
        else:
            ci, ny, nx = x
        co = na * ny * nx
        return no, na, ny, nx, ci, co

    def run_op(self, x, *, nclass, anchors, stride, runtime):
        no, na, ny, nx, ci, co = self._shapes(x.shape[1:], nclass, anchors,
                                              runtime.channel_last)
        if self._grid is None or self._grid.shape != (co,):
            self.make_grid((ny, nx), anchors, stride, runtime=runtime,
                           dtype=x.dtype)
        ba = x.shape[0]
        if runtime.channel_last:
            x = runtime.fn_reshape(x, shape=(ba, ny, nx, na, no),
                                   with_batch=True)
            x = runtime.fn_transpose(x, perm=(0, 3, 1, 2, 4), with_batch=True)
            y = runtime.fn_sigmoid(x)
        else:
            x = runtime.fn_reshape(x, shape=(ba, na, no, ny, nx),
                                   with_batch=True)
            x = runtime.fn_transpose(x, perm=(0, 1, 3, 4, 2), with_batch=True)
            y = runtime.fn_sigmoid(x)
        xy = (y[..., 0:2] * 2 - 0.5 + self._grid) * stride
        wh = (y[..., 2:4] * 2) ** 2 * self._anchor_grid
        if runtime.tensor_is_readonly:
            y = runtime.fn_concat(xy, wh, y[..., 4:], axis=-1, with_batch=True,
                                  channel_pos='ignore')
        else:
            y[..., 0:2] = xy
            y[..., 2:4] = wh
        y = runtime.fn_reshape(y, shape=(ba, co, no), with_batch=True)
        return y

    def make_grid(self, dims, anchors, stride, runtime, dtype):
        import numpy
        ny, nx = dims
        yv, xv = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx),
                                indexing='ij')
        g = numpy.stack((xv, yv), 2).reshape(1, 1, ny, nx, 2)
        a = numpy.array(anchors, dtype=numpy.float32)
        a = a.reshape(1, len(anchors) // 2, 1, 1, 2)
        self._grid = runtime.as_tensor(g, dtype=dtype)
        self._anchor_grid = runtime.as_tensor(a, dtype=dtype)


class Yolov3IR(BaseIR):

    def add_layers(self, tiny):
        self.add_layer('input', type='input', inputs=[dict(ndim=2, channel=3)])
        if tiny:
            self.add_tiny_layers()
        else:
            self.add_full_layers()
        self.add_layer('output', type='output', inputs=['detect'])

    def infer_strides(self):
        from e100_irtool.tools import infer_shapes  # noqa
        s = 256
        ir = self.clone()
        ir.infer_shapes((s, s), dims_only=True, channel_last=False)
        d1 = ir.layers['detect']
        d2 = self.layers['detect']
        strides = []
        for name, layer in d1.layers.items():
            if name.startswith('xywh-'):
                x = layer.inputs[0].shape[-1]
                assert s % x == 0
                stride = s // x
                op = d2.layers[name].op
                strides.append(stride)
                if op.stride is None:
                    op.stride = stride
                else:
                    assert op.stride == stride, \
                        f'layer detect/{name}.stride {op.stride} != {stride}'
        return strides

    def add_tiny_layers(self):
        # backbone
        self.add_layer('b0', layer=self._conv(3, 16, 3), inputs=['input'])
        self.add_layer('b1', op=self._pool(2), inputs=['b0'])
        self.add_layer('b2', layer=self._conv(16, 32, 3), inputs=['b1'])
        self.add_layer('b3', op=self._pool(2), inputs=['b2'])
        self.add_layer('b4', layer=self._conv(32, 64, 3), inputs=['b3'])
        self.add_layer('b5', op=self._pool(2), inputs=['b4'])
        self.add_layer('b6', layer=self._conv(64, 128, 3), inputs=['b5'])
        self.add_layer('b7', op=self._pool(2), inputs=['b6'])
        self.add_layer('b8', layer=self._conv(128, 256, 3), inputs=['b7'])
        self.add_layer('b9', op=self._pool(2), inputs=['b8'])
        self.add_layer('b10', layer=self._conv(256, 512, 3), inputs=['b9'])
        self.add_layer('b11', op=make_op('pad', pads=[0, 0, 1, 1]),
                       inputs=['b10'])
        self.add_layer('b12', op=self._pool(2, 1), inputs=['b11'])
        # head
        self.add_layer('b13', layer=self._conv(512, 1024, 3), inputs=['b12'])
        self.add_layer('b14', layer=self._conv(1024, 256, 1), inputs=['b13'])
        self.add_layer('b15', layer=self._conv(256, 512, 3), inputs=['b14'])
        self.add_layer('b16', layer=self._conv(256, 128, 1), inputs=['b14'])
        self.add_layer('b17', op=self._usmp(2), inputs=['b16'])
        self.add_layer('b18', op=self._concat(1), inputs=['b17', 'b8'])
        self.add_layer('b19', layer=self._conv(384, 256, 3), inputs=['b18'])
        self.add_layer('detect', layer=self._detect('yolov3-tiny', 256, 512),
                       inputs=['b19', 'b15'])

    def add_full_layers(self):
        # backbone
        self.add_layer('b0', layer=self._conv(3, 32, 3), inputs=['input'])
        self.add_layer('b1', layer=self._conv(32, 64, 3, 2), inputs=['b0'])
        self.add_layer('b2', layer=self._bnck(64, 1), inputs=['b1'])
        self.add_layer('b3', layer=self._conv(64, 128, 3, 2), inputs=['b2'])
        self.add_layer('b4', layer=self._bnck(128, 2), inputs=['b3'])
        self.add_layer('b5', layer=self._conv(128, 256, 3, 2), inputs=['b4'])
        self.add_layer('b6', layer=self._bnck(256, 8), inputs=['b5'])
        self.add_layer('b7', layer=self._conv(256, 512, 3, 2), inputs=['b6'])
        self.add_layer('b8', layer=self._bnck(512, 8), inputs=['b7'])
        self.add_layer('b9', layer=self._conv(512, 1024, 3, 2), inputs=['b8'])
        self.add_layer('b10', layer=self._bnck(1024, 4), inputs=['b9'])
        # head
        self.add_layer('b11', layer=self._bnck(1024, 1, False), inputs=['b10'])
        self.add_layer('b12', layer=self._conv(1024, 512, 1), inputs=['b11'])
        self.add_layer('b13', layer=self._conv(512, 1024, 3), inputs=['b12'])
        self.add_layer('b14', layer=self._conv(1024, 512, 1), inputs=['b13'])
        self.add_layer('b15', layer=self._conv(512, 1024, 3), inputs=['b14'])
        self.add_layer('b16', layer=self._conv(512, 256, 1), inputs=['b14'])
        self.add_layer('b17', op=self._usmp(2), inputs=['b16'])
        self.add_layer('b18', op=self._concat(1), inputs=['b17', 'b8'])
        self.add_layer('b19', layer=self._bnck(512, 1, False, c1=768),
                       inputs=['b18'])
        self.add_layer('b20', layer=self._bnck(512, 1, False), inputs=['b19'])
        self.add_layer('b21', layer=self._conv(512, 256, 1), inputs=['b20'])
        self.add_layer('b22', layer=self._conv(256, 512, 3), inputs=['b21'])
        self.add_layer('b23', layer=self._conv(256, 128, 1), inputs=['b21'])
        self.add_layer('b24', op=self._usmp(2), inputs=['b23'])
        self.add_layer('b25', op=self._concat(1), inputs=['b24', 'b6'])
        self.add_layer('b26', layer=self._bnck(256, 1, False, c1=384),
                       inputs=['b25'])
        self.add_layer('b27', layer=self._bnck(256, 2, False), inputs=['b26'])
        self.add_layer('detect', layer=self._detect('yolov3', 256, 512, 1024),
                       inputs=['b27', 'b22', 'b15'])

    def _conv(self, ci, co, k, s=None, fused=True):
        b = make_layer(type='block')
        b.add_layer('inp', type='input', inputs=[dict(channel=ci)])
        b.add_layer('conv', op=make_op('conv2d', in_channel=ci, out_channel=co,
                                       kernel=k, stride=s, padding=k//2),
                    inputs=['inp'])
        if not fused:
            b.add_layer('bn', op=make_op('batch_norm2d', channel=co),
                        inputs=['conv'])
        b.add_layer('act', op='silu', inputs=['conv' if fused else 'bn'])
        b.add_layer('oup', type='output', inputs=['act'])
        return b

    def _bnck(self, c2, n, add=True, c1=None, e=0.5):
        if c1 is None:
            c1 = c2
        c_ = int(c2 * e)
        b = make_layer(type='block', number=n)
        b.add_layer('inp', type='input', inputs=[dict(channel=c1)])
        b.add_layer('cv1', layer=self._conv(c1, c_, 1), inputs=['inp'])
        b.add_layer('cv2', layer=self._conv(c_, c2, 3), inputs=['cv1'])
        if add:
            b.add_layer('add', op='add', inputs=['inp', 'cv2'])
            b.add_layer('oup', type='output', inputs=['add'])
        else:
            b.add_layer('oup', type='output', inputs=['cv2'])
        return b

    def _concat(self, axis, **kwargs):
        return make_op('concat', axis=axis, **kwargs)

    def _usmp(self, s):
        return make_op('resize', scale=s, mode='nearest')

    def _pool(self, k, s=None, p=None):
        if s is None:
            s = k
        return make_op('max_pool2d', kernel=k, stride=s, padding=p)

    def _detect(self, name, *ci, nc=80):
        anchors = ANCHORS[name]
        strides = STRIDES[name]
        no = nc + 5
        na = len(anchors[0]) // 2
        b = make_layer(type='block')
        b.add_layer('inp', type='input', inputs=[dict(channel=c) for c in ci])
        for i, c in enumerate(ci):
            b.add_layer(f'conv-{i}', op=make_op('conv2d', in_channel=c,
                        out_channel=no*na, kernel=1), inputs=[f'inp:{i}'])
            b.add_layer(f'xywh-{i}', op=make_op('yolov3-xywh', nclass=nc,
                        anchors=anchors[i], stride=strides[i]),
                        inputs=[f'conv-{i}'])
        b.add_layer('concat', op=self._concat(1, channel_pos='ignore'),
                    inputs=[f'xywh-{i}' for i in range(len(ci))])
        b.add_layer('oup', type='output', inputs=['concat'])
        return b

    def state_dict_to_weight_name(self, key, tiny):
        k = key.split('.')
        assert k[0] == 'model'
        b = int(k[1])
        nd = 20 if tiny else 28
        assert 0 <= b <= nd
        if b < nd:
            *p, op, w = k[2:]
            if op == 'bn':
                if w.startswith('running_'):
                    w = f'input_{w[8:]}'
                elif w == 'weight':
                    w = 'scale'
                elif w == 'num_batches_tracked':
                    w = None
                else:
                    pass
            elif op == 'conv':
                pass
            else:
                assert False, f'invalid op {op}'
            op = '-'.join((f'b{b}', *p, op))
        else:
            op = k[2]
            if op == 'anchors':
                w = None
            elif op == 'm':
                i, w = k[3:]
                op = f'detect-conv-{i}'
            else:
                assert False, f'invalid op {op}'
        return op, w


def load_pt_model(pt_file):
    from models.common import DetectMultiBackend
    return DetectMultiBackend(pt_file)


@click.group()
def main():
    pass


@main.command()
@click.option('--tiny', is_flag=True, default=False, help='yolov3-tiny')
@click.argument('file', type=Path, required=False)
def gen_ir(file, tiny):
    ir = Yolov3IR()
    ir.add_layers(tiny)
    ir.dump_json(file=file or sys.stdout)


@main.command()
@click.option('--tiny', is_flag=True, default=False, help='yolov3-tiny')
@click.option('--keep', is_flag=True, default=False, help='keep torch tensors')
@click.option('--f32', is_flag=True, default=False, help='convert to float32')
@click.option('--debug', is_flag=True, default=False, help='debug info')
@click.argument('pt_file', type=Path, required=True)
@click.argument('wt_file', type=Path, required=True)
def pt2wt(pt_file, wt_file, tiny, keep, f32, debug):
    import numpy
    import torch

    md = load_pt_model(pt_file).model
    sd = md.state_dict()

    ir = Yolov3IR()
    ir.add_layers(tiny)
    ir.layers = ir.flatten_layers()

    wt = dict()
    for k, v in sd.items():
        n, w = ir.state_dict_to_weight_name(k, tiny)
        if w is None:
            if debug:
                print('.', 'ignore', k, file=sys.stderr)
            continue
        assert n in ir.layers, f'unknown layer {n!r}'
        v = v.detach()
        if keep:
            if f32:
                v = v.to(torch.float32)
        else:
            v = v.numpy()
            if f32:
                v = v.astype(numpy.float32)
        n = f'{n}.{w}'
        if debug:
            print('+', k, '->', n, file=sys.stderr)
        wt[n] = v
    save_pickle(wt, wt_file)


@main.command()
@click.option('--cuda', is_flag=True, default=False)
@click.option('--keep', is_flag=True, default=False)
@click.argument('pt_file', type=Path, required=True)
@click.argument('inp_file', type=Path, required=True)
@click.argument('out_file', type=Path, required=True)
def run_pt(pt_file, inp_file, out_file, cuda, keep):
    import torch
    dev = 'cpu'
    if cuda and torch.cuda.is_available():
        dev = 'cuda:0'
    dev = torch.device(dev)
    mod = load_pt_model(pt_file).to(dev)
    x = load_pickle(inp_file)
    x = torch.as_tensor(x).to(dev)
    if len(x.shape) == 3:
        x = x[None, ...]
    assert len(x.shape) == 4
    y = mod(x)
    if keep:
        y = y.detach().cpu()
    else:
        y = y.detach().cpu().numpy()
    save_pickle(y, out_file)


if __name__ == '__main__':
    main()

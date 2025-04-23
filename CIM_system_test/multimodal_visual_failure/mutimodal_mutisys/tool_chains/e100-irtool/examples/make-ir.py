from e100_irtool import *
from e100_irtool.tools import flatten_layers, layer_graph
from e100_irtool.tools.graph_to_dot import graph_to_dot
ir = make_ir()
image = make_datadef(channel=3, channel_last=True, dtype='uint8')
conv3x3 = make_op('conv2d', kernel=3, padding=1, bias=True, relu=True)
conv1x1 = make_op('conv2d', kernel=1, bias=True, relu=True)
b = make_layer('block')
b.add_layer('in', type='input', inputs=[dict(channel=64)])
b.add_layer('add', op='add', inputs=['in', 'in'])
b.add_layer('out', type='output', inputs=['add'])
block = make_layer('block')
block.add_layer('in', type='input', inputs=[dict(channel=64)])
block.add_layer('x', b.clone(inputs=['in'], number=2))
block.add_layer('conv-1', op=conv3x3.clone(in_channel=64, out_channel=64), inputs=['x'])
block.add_layer('conv-2', op=conv1x1.clone(in_channel=64, out_channel=64), inputs=['conv-1'])
block.add_layer('conv-3', op=conv1x1.clone(in_channel=64, out_channel=64), inputs=['x'])
block.add_layer('add', op='add', inputs=[dict(ref='conv-2', dtype='int4'), 'conv-3'])
block.add_layer('out', type='output', inputs=['add'])
ir.add_layer('image', type='input', inputs=[image.clone()])
ir.add_layer('conv-1', op=conv3x3.clone(in_channel=3, out_channel=64), inputs=['image'])
ir.add_layer('res-blk', block.clone(inputs=['conv-1'], number=3))
ir.add_layer('fc-1', op=make_op('fc', in_channel=64, out_channel=256), inputs=['res-blk'])
ir.add_layer('fc-2', op=make_op('fc', in_channel=256, out_channel=10), inputs=['fc-1'])
ir.add_layer('classes', type='output', inputs=['fc-2'])
ir.add_device('npu', 'a111-npu', number=2)
ir.add_device('xb', 'rram-144k', number=4)
ir.validate_graph()
assert not ir.is_flat_graph()
if 0:
    ir.layers = ir.flatten_layers()
    ir.validate_graph()
    assert ir.is_flat_graph()
if 0:
    ir.dump_json()
if 1:
    g = ir.build_flat_graph()

    def label(name, obj):
        n = getattr(obj, 'number', None)
        return None if n is None else f'{name} [{n}]'
    graph_to_dot(g, label=label)
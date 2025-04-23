from typing import Callable
from ..core import BaseIR, mixin
from .base import BaseRuntime

def _collect_inputs(layer, data):
    for (i, dd) in layer.iter_inputs():
        (name, index) = dd.parse_ref()
        if index is None:
            index = 0
        inputs = data.get(name)
        assert inputs and index < len(inputs), f'invalid ref {dd.ref!r}'
        yield inputs[index]

@mixin(BaseRuntime)
class IRRunner:

    def run_ir(self, ir, inputs, weights, *, outputs=None, callback=None):
        assert isinstance(ir, BaseIR), f'invalid IR type {type(ir)}'
        assert ir.is_flat_graph(), 'IR is not a flat graph'
        (inp, oup) = ir.get_io_layers()
        if isinstance(inputs, dict):
            data = {k: tuple(v) if isinstance(v, (tuple, list)) else (v,) for (k, v) in inputs.items() if v is not None}
        elif isinstance(inputs, (tuple, list)):
            with ir.with_layer(inp) as layer:
                n = len(layer.inputs or ())
                assert len(inputs) == n, f'inputs number {len(inputs)} != {n}'
                data = {inp: tuple(inputs)}
        else:
            data = {inp: (inputs,)}
        for (k, v) in data.items():
            assert k in ir.layers, f'unknown layer {k!r}'
            assert isinstance(v, (tuple, list)), f'invalid layer {k!r} inputs type {type(v)}'
            for (i, x) in enumerate(v):
                assert isinstance(x, self.tensor_type), f'invalid layer {k!r} inputs[{i}] type {type(x)}'
        assert isinstance(weights, dict), f'invalid weights {type(weights)}'
        ons = None
        if outputs is None:
            pass
        elif isinstance(outputs, str):
            ons = {outputs}
        elif outputs is True:
            ons = set(ir.layers) - {inp, oup}
        elif isinstance(outputs, (tuple, list, set)):
            ons = set(outputs)
        else:
            assert False, f'invalid outputs type {type(outputs)}'
        for k in ons or ():
            assert k in ir.layers and k not in (inp, oup), f'invalid output layer {k!r}'
        if callback is not None:
            assert isinstance(callback, Callable), f'invalid callback type {type(callback)}'
        done = False
        for (name, layer) in ir.iter_layers(sorted=True):
            if done:
                continue
            if name in data:
                continue
            if layer.is_io_layer():
                continue
            x = list(_collect_inputs(layer, data))
            wts = dict()
            lname = name
            if layer.type == 'reuse':
                lname = layer.layer
                layer = ir.layers[lname]
            ats = layer.op.get_attrs()
            for k in layer.op.weights:
                wn = f'{lname}.{k}'
                if k not in layer.op.optional_weights:
                    assert wn in weights, f'missing weight {wn}'
                wts[k] = weights.get(wn)
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts, attrs=ats, outputs=None)
            y = self.run_layer(layer, inputs=x, weights=wts, attrs=ats, ir=ir, name=name)
            if not isinstance(y, (tuple, list)):
                y = (y,)
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts, attrs=ats, outputs=y)
            data[name] = tuple(y)
            if ons is not None and all((k in data for k in ons)):
                done = True
        if ons is not None:
            res = {}
            for k in ons:
                v = data[k]
                res[k] = v[0] if len(v) == 1 else v
            if isinstance(outputs, str):
                res = iter(res.values()).next()
        else:
            with ir.with_layer(oup) as layer:
                res = list(_collect_inputs(layer, data))
            if len(res) == 1:
                res = res[0]
        return res
from typing import Callable
from ..core.reg import AbsReg, RegBase
from ..core.ns import ns_push
from ..core.type_util import to_var_token

class RuntimeReg(AbsReg, key='name'):
    pass

class BaseRuntime(RegBase, metaclass=RuntimeReg):
    channel_last = None
    tensor_is_readonly = None

    def __init__(self, *, channel_last=None):
        if channel_last is not None:
            assert self.channel_last is None or self.channel_last == channel_last, 'channel_last conflicts'
            self.channel_last = channel_last
        self.init_backend()
        assert self.channel_last is not None, 'channel_last is not set'

    def init_backend(self):
        pass

    def rand(self, shape):
        raise NotImplementedError

    def as_tensor(self, data, dtype=None):
        raise NotImplementedError

    def run_op(self, op, *args, **kwargs):
        if isinstance(op, str):
            (op, op_id) = (None, op)
        else:
            op_id = op.op_id
        opn = to_var_token(op_id)
        with ns_push(f'op[{op_id!r}]'):
            fn = getattr(self, f'fn_{opn}', None)
            if isinstance(fn, Callable):
                return fn(*args, **kwargs)
            if op is not None:
                fn = getattr(op, 'run_op', None)
                if isinstance(fn, Callable):
                    return fn(*args, runtime=self, **kwargs)
            assert False, f'runner for op {op_id!r} is not found'

    def run_layer(self, layer, *, inputs, weights, attrs, ir=None, name=None):
        assert layer.type == 'op', f'layer type {layer.type} != "op"'
        args = inputs or ()
        kwargs = weights or {}
        if attrs:
            kwargs.update(attrs)
        return self.run_op(layer.op, *args, **kwargs)

def load_runtime(name, **kwargs):
    cls = RuntimeReg.lookup(name)
    assert cls is not None, f'unkown runtime {name!r}'
    return cls(**kwargs)

def enum_runtimes():
    return tuple((k for (k, cls) in RuntimeReg.iter_reg()))
from ..core import UnaryOp

class ExpandOp(UnaryOp):

    op_id = 'expand'
    attrs = ('shape')
    shape = None
    
    def __init__(self, *, shape=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('shape', shape)
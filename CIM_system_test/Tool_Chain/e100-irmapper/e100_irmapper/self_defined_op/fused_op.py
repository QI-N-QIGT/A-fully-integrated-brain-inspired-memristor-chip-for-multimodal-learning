from e100_irtool.ops.conv import *
from e100_irtool.ops.matmul import *
from e100_irtool.ops.math import *
from e100_irtool.ops.trans import *
from e100_irtool.ops.pool import *
from e100_irtool.ops.activate import *
from e100_irtool.core.type_util import to_cls_obj
from e100_irtool.ops.split import *
                   
class fused_conv2d(Conv2dOp):
    
    op_id = 'fused_conv2d'
    relu = None
    # pool = None
    silu = None
    # split = None
    mul = None
    add = None
    
    def __init__(self, *, relu = None, silu = None, mul=None, add=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('silu', silu)
        self.set_attr('mul', mul)
        self.set_attr('add', add)
        # if pool != None:
        #     self.pool = pool

        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)
            
        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)
            
        if mul != None:
            self.mul = to_cls_obj(mul, cls=MulOp)
            
        if add != None:
            self.add = to_cls_obj(silu, cls=AddOp)

class fused_fc(MatMulOp):
    
    op_id = 'fused_fc'
    relu = None
    silu = None
    
    def __init__(self, *, relu = None, silu = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('silu', silu)
        
        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)
        
        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)
    
class fused_add(AddOp):
    
    op_id = 'fused_add'
    relu = None
    pool = None
    split = None
    silu = None
    
    def __init__(self, *, relu = None, pool = None, split =None, silu = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('pool', pool)
        self.set_attr('split', split)
        self.set_attr('silu', silu)
        
        if pool != None:
            self.pool = pool

        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)

        if split != None:
            self.split = to_cls_obj(split, cls=SplitOp)
        
        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)
            
class fused_concat(ConcatOp):
    
    op_id = 'fused_concat'
    relu = None
    pool = None
    split = None
    silu = None
    
    def __init__(self, *, relu = None, pool = None, split = None, silu = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('pool', pool)
        self.set_attr('split', split)
        self.set_attr('silu', silu)
        
        if pool != None:
            self.pool = pool

        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)

        if split != None:
            self.split = to_cls_obj(split, cls=SplitOp)

        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)
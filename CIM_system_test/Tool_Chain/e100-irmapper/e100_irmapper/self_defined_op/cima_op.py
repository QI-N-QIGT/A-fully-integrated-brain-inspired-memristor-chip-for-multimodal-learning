from e100_irtool.core import UnaryOp

class MulAddOp(UnaryOp):
    
    op_id = 'mul_add'

class TypeConversionOp(UnaryOp):
    
    op_id = 'type_conversion'
    
    in_dtype = '4bit'
    out_dtype = '8bit'
    def __init__(self, *, in_dtype = '4bit', out_dtype = '8bit', **kwargs):
        super().__init__(**kwargs)
        assert in_dtype != out_dtype, f'输入类型:{in_dtype} 与 输出类型:{out_dtype} 应该保持不一致 !!!'
        self.set_attr('in_dtype', in_dtype)
        self.set_attr('out_dtype', out_dtype)
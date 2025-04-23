from ..core import UnaryOp, BinaryOp

class SignOp(UnaryOp):
    op_id = 'sign'

class AbsOp(UnaryOp):
    op_id = 'abs'

class NegOp(UnaryOp):
    op_id = 'neg'

class CeilOp(UnaryOp):
    op_id = 'ceil'

class FloorOp(UnaryOp):
    op_id = 'floor'

class AddOp(BinaryOp):
    op_id = 'add'

class SubOp(BinaryOp):
    op_id = 'sub'

class MulOp(BinaryOp):
    op_id = 'mul'

class DivOp(BinaryOp):
    op_id = 'div'

class ModOp(BinaryOp):
    op_id = 'mod'

class PowOp(BinaryOp):
    op_id = 'pow'

class ExpOp(UnaryOp):
    op_id = 'exp'

class LogOp(UnaryOp):
    op_id = 'log'

class SqrtOp(UnaryOp):
    op_id = 'sqrt'

class SinOp(UnaryOp):
    op_id = 'sin'

class CosOp(UnaryOp):
    op_id = 'cos'

class TanOp(UnaryOp):
    op_id = 'tan'

class AsinOp(UnaryOp):
    op_id = 'asin'

class AcosOp(UnaryOp):
    op_id = 'acos'

class AtanOp(UnaryOp):
    op_id = 'atan'

class SinhOp(UnaryOp):
    op_id = 'sinh'

class CoshOp(UnaryOp):
    op_id = 'cosh'

class TanhOp(UnaryOp):
    op_id = 'tanh'

class AsinhOp(UnaryOp):
    op_id = 'asinh'

class AcoshOp(UnaryOp):
    op_id = 'acosh'

class AtanhOp(UnaryOp):
    op_id = 'atanh'

class LogicalNotOp(UnaryOp):
    op_id = 'logical_not'

class LogicalAndOp(BinaryOp):
    op_id = 'logical_and'

class LogicalOrOp(BinaryOp):
    op_id = 'logical_or'

class LogicalXorOp(BinaryOp):
    op_id = 'logical_xor'

class BitwiseNotOp(UnaryOp):
    op_id = 'bitwise_not'

class BitwiseAndOp(BinaryOp):
    op_id = 'bitwise_and'

class BitwiseOrOp(BinaryOp):
    op_id = 'bitwise_or'

class BitwiseXorOp(BinaryOp):
    op_id = 'bitwise_xor'

class EqualOp(BinaryOp):
    op_id = 'equal'

class LessOp(BinaryOp):
    op_id = 'less'

class LessOrEqualOp(BinaryOp):
    op_id = 'less_or_equal'

class GreaterOp(BinaryOp):
    op_id = 'greater'

class GreaterOrEqualOp(BinaryOp):
    op_id = 'greater_or_equal'
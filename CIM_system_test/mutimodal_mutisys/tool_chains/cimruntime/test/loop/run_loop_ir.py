from e100_irtool.core.ir import load_ir
from e100_irtool.tools.loop import *
from e100_irtool.tools import flatten_layers
from cimruntime.c200.c200_rt import C200NumpyRT as CRT
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy as np
ir = load_ir(file=f'ir\\test_ir.yaml')
ir.layers = ir.flatten_layers()
input_data = np.random.randint(1, 10, (2, 4))
in_ = CNT.to_cimtensor(input_data, multi_batch=True)
print()
rt = CRT()
output = rt.run_ir(ir, in_, outputs=['out'])
output = output['out']
if len(output) == 1:
    output = output[0]
re = CNT.scale_recover(output).data
print(re)
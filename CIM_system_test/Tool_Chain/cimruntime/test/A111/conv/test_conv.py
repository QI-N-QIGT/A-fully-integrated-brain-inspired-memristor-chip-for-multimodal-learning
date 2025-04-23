from e100_irtool.core.ir import load_ir
from cimruntime.A111.chip.a111_chip_rt import A111NumpyRT as ANT 
import numpy
# from cimruntime.c200.ir_runner import IRRunner # noqa
import torch.nn.functional as F  
import torch as tc
from e100_irmapper.device.a111 import *
from e100_irmapper.fused_op.op import *
import unittest

class TestA111Runtime(unittest.TestCase):

    def test_run_ir(self):
        
        ir = load_ir(file='test\\A111\\conv\\ir_conv_mapped.yaml')
        
        rt = ANT()
        weights = {'Conv_0.weight':numpy.random.randint(-7,7,size=(32,1,3,3))}
        inputs = numpy.random.randint(0,2,size=(1,1,28,28))
        weights_ = weights['Conv_0.weight']
        output_numpy = F.conv2d(tc.from_numpy(inputs),tc.from_numpy(weights_),stride=1,padding=0)
        output_numpy = F.relu(output_numpy).type(tc.float32)
        output_numpy = F.max_pool2d(output_numpy,kernel_size=2, stride=2).numpy()
        output_a111 = rt.run_ir(ir, inputs, weights=weights)

        diff = (output_numpy - output_a111)
        re_diff = diff / (output_numpy.max() + 1e-6) * 100
        print(f"Numpt results:{output_numpy}")
        print(f"a111 results:{output_a111}")
        print(f"diff mean:{re_diff.mean()}%")
        print(f"diff var:{re_diff.var()}%")
    
if __name__ == "__main__":
    unittest.main()

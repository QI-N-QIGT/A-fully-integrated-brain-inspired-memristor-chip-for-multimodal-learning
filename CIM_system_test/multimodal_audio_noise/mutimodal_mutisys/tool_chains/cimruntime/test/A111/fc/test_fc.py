from e100_irtool.core.ir import load_ir
from cimruntime.A111.chip.a111_chip_rt import A111NumpyRT as ANT
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy
import torch.nn.functional as F
import torch as tc
from e100_irmapper.device.a111 import *
from e100_irmapper.fused_op.op import *
import unittest

class TestA111Runtime(unittest.TestCase):

    def test_run_ir(self):
        mapped_data = {0: numpy.random.randint(-7, 7, size=(320, 128))}
        ir = load_ir(file='test\\A111\\fc\\ir_fc_twolayer_mapped_a111.yaml')
        rt = ANT(mapped_array_data=mapped_data)
        weights = {'MatMul_0.weight': numpy.random.randint(-7, 7, size=(64, 320)), 'MatMul_2.weight': numpy.random.randint(-7, 7, size=(10, 64))}
        inputs = numpy.random.randint(0, 2, size=(1, 320))
        inputs_a111 = CNT(data=inputs)
        weights_1 = weights['MatMul_0.weight']
        weights_2 = weights['MatMul_2.weight']
        output_numpy = F.linear(tc.from_numpy(inputs), tc.from_numpy(weights_1))
        output_numpy = F.relu(output_numpy)
        output_numpy = F.linear(output_numpy, tc.from_numpy(weights_2)).numpy()
        output_a111 = rt.run_ir(ir, inputs_a111, weights=weights)
        output_a111 = CNT.scale_recover(output_a111).data
        diff = output_numpy - output_a111
        re_diff = diff / (output_numpy.max() + 1e-06) * 100
        print()
        print()
        print()
        print()
if __name__ == '__main__':
    unittest.main()
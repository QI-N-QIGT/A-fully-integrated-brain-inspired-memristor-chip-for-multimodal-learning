from cimruntime.c200.c200_rt import C200NumpyRT
from e100_irtool.runtime.numpy import NumpyRuntime
from e100_irtool.core.ir import load_ir
from e100_irtool.cmd.data import *
import unittest

class TestC200NumpyRT(unittest.TestCase):
    def test(self):
        ir = load_ir(file='test\\c200\\ir.yaml')
        rt = C200NumpyRT()
        inputs = load_pickle('test\\c200\\input_ir.np')
        weights = load_pickle('test\\c200\\weight_ir.np')
        output = rt.run_ir(ir, inputs, weights)

if __name__ == "__main__":
    unittest.main()
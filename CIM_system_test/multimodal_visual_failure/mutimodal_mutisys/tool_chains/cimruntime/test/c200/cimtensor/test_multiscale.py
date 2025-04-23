import unittest
import numpy
import torch
import torch.nn as nn
from cimruntime.cimtensor import CIMNumpyTensor as CNT

class TestMultiScale(unittest.TestCase):

    def test_multiscale(self):
        a = numpy.random.randn(2, 3, 1, 2, 2)
        b = CNT.to_cimtensor_quant(data=a, half_level=31, multi_batch=True)
        c = CNT.scale_recover(b).data
        print()
        print()
        print()
if __name__ == '__main__':
    unittest.main()
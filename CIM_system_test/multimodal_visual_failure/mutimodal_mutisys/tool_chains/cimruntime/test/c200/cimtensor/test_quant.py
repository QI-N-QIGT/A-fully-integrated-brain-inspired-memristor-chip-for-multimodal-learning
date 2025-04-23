import unittest
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy

class TestC200NumpyRT(unittest.TestCase):

    def test_binary(self):
        data = numpy.random.randint(-8, 7, size=(10, 10))
        print()
        y1 = CNT.to_cimtensor_quant(data=data, method='Binary')
        print()
        print()
        print()
        print()
        print()

    def test_thr_binary(self):
        data = numpy.random.randint(-8, 7, size=(10, 10))
        print()
        y1 = CNT.to_cimtensor_quant(data=data, method='ThresBinary', thr=5)
        print()
        print()
        print()
        print()
        print()
if __name__ == '__main__':
    unittest.main()
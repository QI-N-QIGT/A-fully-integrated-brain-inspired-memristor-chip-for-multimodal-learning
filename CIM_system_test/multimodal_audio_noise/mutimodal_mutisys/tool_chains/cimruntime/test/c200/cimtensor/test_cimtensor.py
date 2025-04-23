import unittest
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy

class TestC200NumpyRT(unittest.TestCase):

    def test_mul(self):
        y1 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y2 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y3 = CNT.mul(y1, y2)
        print()
        print()
        print()
        print()

    def test_add(self):
        y1 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y2 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y3 = CNT.add(y1, y2)
        print()
        print()
        print()
        print()

    def test_sub(self):
        y1 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y2 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y3 = CNT.sub(y1, y2)
        print()
        print()
        print()
        print()

    def test_div(self):
        y1 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y2 = CNT.to_cimtensor(numpy.array([1, 2, 3]))
        y3 = CNT.div(y1, y2)
        print()
        print()
        print()
        print()

    def test_print()
        print()

    def test_op(self):
        pass
if __name__ == '__main__':
    unittest.main()
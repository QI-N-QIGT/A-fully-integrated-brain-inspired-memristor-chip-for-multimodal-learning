import unittest
import numpy
import torch
import torch.nn as nn
from cimruntime.cimtensor import CIMNumpyTensor as CNT
class TestMultiScale(unittest.TestCase):
    
    def test_multiscale(self):
        # a = numpy.array([[1,2,3],
        #                  [4,5,6],
        #                  [7,8,9]])
        a = numpy.random.randn(2,3,1,2,2)
        b = CNT.to_cimtensor_quant(data=a,half_level=31,multi_batch=True)
        c = CNT.scale_recover(b).data
        
        print(f'original data:\n{a}')
        print(f'CNT tensor value:\n{b}')
        print(f'recover value:\n{c}')
        
if __name__ == "__main__":
    unittest.main()
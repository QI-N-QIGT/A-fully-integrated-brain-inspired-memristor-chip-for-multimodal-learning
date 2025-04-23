import unittest
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy

class TestC200NumpyRT(unittest.TestCase):
    
    def test_binary(self):
        data = numpy.random.randint(-8,7,size=(10,10))
        print(f"原始数据：{data}")
        y1 = CNT.to_cimtensor_quant(data=data,method='Binary')
        print(f"量化数据：{y1}")
        print("======data results=========")
        print(f"data:{y1.data}")
        print(f"shape:{y1.shape}")
        print(f"ndim:{y1.ndim}")
    
    def test_thr_binary(self):
        data = numpy.random.randint(-8,7,size=(10,10))
        print(f"原始数据：{data}")
        y1 = CNT.to_cimtensor_quant(data=data,method='ThresBinary',thr=5)
        print(f"量化数据：{y1}")
        print("======data results=========")
        print(f"data:{y1.data}")
        print(f"shape:{y1.shape}")
        print(f"ndim:{y1.ndim}")


    
if __name__ == "__main__":
    unittest.main()
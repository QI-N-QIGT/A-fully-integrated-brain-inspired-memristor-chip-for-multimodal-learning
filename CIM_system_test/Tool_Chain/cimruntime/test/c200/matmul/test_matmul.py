from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
from e100_irtool.core.ir import load_ir

from cimruntime.c200.c200_rt import C200NumpyRT 
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy
# from cimruntime.c200.ir_runner import IRRunner # noqa

import unittest

class TestC200NumpyRT(unittest.TestCase):

    def test_run_ir(self):
        np.random.seed(19)
        mapped_data = {0:numpy.random.randint(0,15,size=(576,128))}
        adc_scale = [1]*1
        assert len(mapped_data.keys()) == len(adc_scale)
        ir = load_ir(file='test\\c200\\matmul\\ir_matmul_mapped.yaml')
        rt = C200NumpyRT(mapped_array_data = mapped_data, device_adc_scale_data = adc_scale)
        
        inputs = CNT.to_cimtensor(numpy.random.randint(0,2,size=(1,64)))
        weights = mapped_data[0][0:64,0:10] - 8
        output_numpy = inputs.data @  weights
        o1 = CNT.to_cimtensor_quant(data=output_numpy,half_level=127)
        o2 = CNT.scale_recover(o1)
        print(f"numpy quant int :{o1}")
        print(f"numpy quant float :{o2}")
        output_c200 = rt.run_ir(ir, inputs)
        print(f"c200 quant int:{output_c200}")
        print(f"c200 quant flaot:{CNT.scale_recover(output_c200)}")
        assert (output_c200.data == o1.data).all()
if __name__ == "__main__":
    unittest.main()

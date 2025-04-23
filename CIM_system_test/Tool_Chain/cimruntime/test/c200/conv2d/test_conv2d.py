from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
from e100_irtool.core.ir import load_ir

from cimruntime.c200.c200_rt import C200NumpyRT 
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy
# from cimruntime.c200.ir_runner import IRRunner # noqa
import torch.nn.functional as F  
import torch as tc

import unittest

class TestC200NumpyRT(unittest.TestCase):

    def test_run_ir(self):
        mapped_data = {0:numpy.random.randint(0,15,size=(576,128))}
        adc_scale = [1]*1
        assert len(mapped_data.keys()) == len(adc_scale)
        ir = load_ir(file='test\\c200\\conv2d\\ir_conv_mapped.yaml')
        rt = C200NumpyRT(mapped_array_data = mapped_data, device_adc_scale_data = adc_scale)
        inputs = CNT.to_cimtensor(numpy.random.randint(0,2,size=(1,1,3,3)))
        weights = (mapped_data[0][0:9,0:2] - 8).transpose(1,0).reshape(2,1,3,3)
        output_numpy = F.conv2d(input=tc.from_numpy(inputs.data),weight=tc.from_numpy(weights),stride=1,padding=1).numpy()
        output_c200 = rt.run_ir(ir, inputs)
        output_c200 = CNT.scale_recover(output_c200)
        diff = (output_numpy - output_c200.data)
        re_diff = diff / (output_numpy.max() + 1e-6) * 100
        print(f"Numpt results:{output_numpy}")
        print(f"c200 results:{output_c200}")
        print(f"diff mean:{re_diff.mean()}%")
        print(f"diff var:{re_diff.var()}%")
    
if __name__ == "__main__":
    unittest.main()

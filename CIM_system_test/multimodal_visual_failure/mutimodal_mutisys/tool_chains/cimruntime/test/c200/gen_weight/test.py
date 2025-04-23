from e100_irmapper.parser import *
from e100_irmapper.placement import *
from e100_irmapper.search import *
from e100_irmapper.helper import *
from e100_irtool.core.ir import load_ir
from cimruntime.c200.c200_rt import C200NumpyRT
from cimruntime.cimtensor import CIMNumpyTensor as CNT
import numpy
from cimruntime.gen_weight import gen_weight
import unittest

class TestC200NumpyRT(unittest.TestCase):

    def test_matmul(self):
        ir = load_ir(file='test\\c200\\gen_weight\\ir_matmul_mapped.yaml')
        weight = {'fc': numpy.random.randint(0, 2, size=(10, 10))}
        array_data = gen_weight(ir, weight)
        copy_1 = numpy.zeros(shape=(30, 30))
        copy_2 = numpy.zeros(shape=(30, 30))
        for (k, v) in array_data.items():
            if k % 4 == 0:
                if k < 4:
                    copy_1[0:15, 0:15] = v[0:15, 0:15]
                else:
                    copy_2[0:15, 0:15] = v[0:15, 0:15]
            elif k % 4 == 1:
                if k < 4:
                    copy_1[0:15, 15:30] = v[0:15, 15:30]
                else:
                    copy_2[0:15, 15:30] = v[0:15, 15:30]
            elif k % 4 == 2:
                if k < 4:
                    copy_1[15:30, 0:15] = v[15:30, 0:15]
                else:
                    copy_2[15:30, 0:15] = v[15:30, 0:15]
            elif k % 4 == 3:
                if k < 4:
                    copy_1[15:30, 15:30] = v[15:30, 15:30]
                else:
                    copy_2[15:30, 15:30] = v[15:30, 15:30]
        assert (copy_1 == copy_2).all()
        assert (np.tile(weight['fc'].T, [3, 3]) == copy_1).all()

    def test_conv(self):
        ir = load_ir(file='test\\c200\\gen_weight\\ir_conv_mapped.yaml')
        weight = {'conv': numpy.random.randint(0, 2, size=(2, 4, 2, 2))}
        array_data = gen_weight(ir, weight, format='HWC')
        weight_ = weight['conv'].transpose(0, 2, 3, 1)
        weight_ = weight_.reshape(weight_.shape[0], -1, weight_.shape[3])
        weight_1 = weight_[:, :, 0:2].reshape(weight_.shape[0], -1)
        weight_2 = weight_[:, :, 2:4].reshape(weight_.shape[0], -1)
        weight_1 = np.tile(weight_1.transpose(1, 0), [2, 2])
        weight_2 = np.tile(weight_2.transpose(1, 0), [2, 2])
        weight_1_ = weight_1[:, 0:2]
        weight_2_ = weight_1[:, 2:4]
        weight_3_ = weight_2[:, 0:2]
        weight_4_ = weight_2[:, 2:4]
        assert (weight_1_ == array_data[0][0:16, 0:2]).all()
        assert (weight_2_ == array_data[1][0:16, 0:2]).all()
        assert (weight_3_ == array_data[2][0:16, 0:2]).all()
        assert (weight_4_ == array_data[3][0:16, 0:2]).all()
if __name__ == '__main__':
    unittest.main()
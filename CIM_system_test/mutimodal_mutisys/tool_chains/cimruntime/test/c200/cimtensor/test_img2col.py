import unittest
from cimruntime.c200.utils import feature_map_to_input
import numpy
import torch
import torch.nn as nn

class TestImage2Col(unittest.TestCase):

    def test_binary(self):
        data = numpy.random.random((3, 4, 4))
        kernel_size = 3
        padding = 1
        stride = 1
        data_torch = torch.from_numpy(data)
        unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        while len(data_torch.shape) < 4:
            data_torch = data_torch.unsqueeze(dim=0)
        array_torch = unfold(data_torch)
        array_torch = array_torch.numpy().squeeze()
        array_utils = feature_map_to_input(data, kernel_size, stride, padding)
        assert (array_utils == array_torch).all()
if __name__ == '__main__':
    unittest.main()
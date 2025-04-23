import numpy
from .quant import *
import torch

class CIMNumpyTensor:
    

    def __init__(self, *, data, scale=1, max_percent=0, min_percent=0, shift_scale=1):
        if not isinstance(data, numpy.ndarray):
            raise ValueError(f"数据类型暂只支持 ['numpy.ndarray']!!!")
        self.data = data
        self.scale = scale
        self.max_percent = max_percent
        self.min_percent = min_percent
        self.shift_scale = shift_scale

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def get_slice(cls, CNT, start, end, step):
        return cls(data=CNT.data[start:end:step], scale=CNT.scale[start:end:step])

    @property
    def items(self):
        return (self.data, self.scale)

    @property
    def ndim(self):
        return len(self.data.shape)

    @property
    def shape(self):
        return self.data.shape

    def __str__(self):
        return '{}["data":{},"scale":{}]'.format(self.__class__.__name__, self.data, self.scale)

    @classmethod
    def to_cimtensor(cls, data, scale=1, multi_batch=False, max_percent=0, min_percent=0, shift_scale=1):
        if not isinstance(data, numpy.ndarray):
            raise ValueError(f"数据类型暂只支持 ['numpy.ndarray'] !!!")
        if multi_batch:
            if len(data.shape) < 2:
                raise ValueError(f'多维度scale下必须是多个batch，维度需要>=2，目前维度是{len(data.shape)}!!!')
            Batch = data.shape[0]
            data_dim = len(data.shape)
            scale = numpy.ones(Batch) * scale
            shift_scale = numpy.ones(Batch) * shift_scale
            for i in range(data_dim - 1):
                scale = numpy.expand_dims(scale, axis=1)
                shift_scale = numpy.expand_dims(shift_scale, axis=1)
        return cls(data=data, scale=scale, max_percent=max_percent, min_percent=min_percent, shift_scale=shift_scale)

    @classmethod
    def to_cimtensor_quant(cls, *, data=None, half_level=None, method='Uniform', thr=None, multi_batch=False, max_percent=0, min_percent=0, shift_scale=1):
        if not isinstance(data, numpy.ndarray):
            raise ValueError(f"数据类型暂只支持['numpy.ndarray'] !!!")
        if multi_batch:
            if len(data.shape) < 2:
                raise ValueError(f'多维度scale下必须是多个batch，维度需要>=2，目前维度是{len(data.shape)}!!!')
            Batch = data.shape[0]
            data_dim = len(data.shape)
            scale = numpy.ones((Batch,))
            shift_scale = numpy.ones((Batch,))
            for i in range(data_dim - 1):
                scale = numpy.expand_dims(scale, axis=1)
                shift_scale = numpy.expand_dims(shift_scale, axis=1)
            data_batch = numpy.zeros_like(data)
            for i in range(Batch):
                (data_, scale_) = cls.postprocess_elementwise_op(data=data[i], quant=half_level, method=method, thr=thr)
                data_batch[i] = data_
                scale[i] = scale_
            data = data_batch
        else:
            (data, scale) = cls.postprocess_elementwise_op(data=data, quant=half_level, method=method, thr=thr)
        return cls(data=data, scale=scale, max_percent=max_percent, min_percent=min_percent, shift_scale=shift_scale)

    @classmethod
    def scale_recover(cls, ctensor1):
        return cls(data=ctensor1.data / ctensor1.scale, scale=1, max_percent=ctensor1.max_percent, min_percent=ctensor1.min_percent, shift_scale=ctensor1.shift_scale)

    @classmethod
    def preprocess_elementwise_op(cls, ctensor1, ctensor2):
        return (cls.scale_recover(ctensor1).data, cls.scale_recover(ctensor2).data)

    @classmethod
    def postprocess_elementwise_op(cls, data, quant=None, method='Uniform', thr=None):
        if not isinstance(data, numpy.ndarray):
            raise ValueError(f"数据类型暂只支持['numpy.ndarray']!!!")
        if method == 'Uniform':
            if quant != None and isinstance(quant, int):
                (r_data, r_scale) = data_quantization_sym(data, half_level=quant, isint=1)
            else:
                raise ValueError(f'量化level参数{quant}错误!!!')
        elif method == 'Binary':
            r_data = binary_quant(data)
            r_scale = 1
        elif method == 'ThresBinary':
            assert thr != None
            r_data = thres_binary_quant(data, thr=thr)
            r_scale = 1
        else:
            r_data = data
            r_scale = 1
        return (r_data, r_scale)

    @classmethod
    def add(cls, ctensor1, ctensor2, quant=None):
        if not isinstance(ctensor1, cls) or not isinstance(ctensor2, cls):
            raise ValueError(f'{type(ctensor1)} and {type(ctensor2)} are not the same                                 with {cls}!!!')
        (data1, data2) = cls.preprocess_elementwise_op(ctensor1, ctensor2)
        r_data_all = data1 + data2
        (r_data, r_scale) = cls.postprocess_elementwise_op(r_data_all, quant=quant)
        return cls(data=r_data, scale=r_scale)

    def __add__(self, other):
        if str(type(other)) == "<class 'numpy.ndarray'>":
            return CIMNumpyTensor(data=self.data / self.scale + other)
        else:
            return CIMNumpyTensor(data=self.data / self.scale + other.data / other.scale)

    @classmethod
    def mul(cls, ctensor1, ctensor2, quant=None):
        if not isinstance(ctensor1, cls) or not isinstance(ctensor2, cls):
            raise ValueError(f'{ctensor1} and {ctensor2} are not the same                                 with {cls}!!!')
        (data1, data2) = cls.preprocess_elementwise_op(ctensor1, ctensor2)
        r_data_all = data1 * data2
        (r_data, r_scale) = cls.postprocess_elementwise_op(r_data_all, quant=quant)
        return cls(data=r_data, scale=r_scale)

    def __mul__(self, other):
        if str(type(other)) == "<class 'numpy.ndarray'>":
            return CIMNumpyTensor(data=self.data * other, scale=self.scale)
        else:
            return CIMNumpyTensor(data=self.data / self.scale * (other.data / other.scale))

    @classmethod
    def sub(cls, ctensor1, ctensor2, quant=None):
        if not isinstance(ctensor1, cls) or not isinstance(ctensor2, cls):
            raise ValueError(f'{ctensor1} and {ctensor2} are not the same                                 with {cls}!!!')
        (data1, data2) = cls.preprocess_elementwise_op(ctensor1, ctensor2)
        r_data_all = data1 - data2
        (r_data, r_scale) = cls.postprocess_elementwise_op(r_data_all, quant=quant)
        return cls(data=r_data, scale=r_scale)

    def __sub__(self, other):
        if str(type(other)) == "<class 'numpy.ndarray'>":
            return CIMNumpyTensor(data=self.data / self.scale - other)
        else:
            return CIMNumpyTensor(data=self.data / self.scale - other.data / other.scale)

    @classmethod
    def div(cls, ctensor1, ctensor2, quant=None):
        if not isinstance(ctensor1, cls) or not isinstance(ctensor2, cls):
            raise ValueError(f'{ctensor1} and {ctensor2} are not the same                                 with {cls}!!!')
        (data1, data2) = cls.preprocess_elementwise_op(ctensor1, ctensor2)
        r_data_all = data1 / data2
        (r_data, r_scale) = cls.postprocess_elementwise_op(r_data_all, quant=quant)
        return cls(data=r_data, scale=r_scale)

    def __div__(self, other):
        if str(type(other)) == "<class 'numpy.ndarray'>":
            return CIMNumpyTensor(data=self.data / other, scale=self.scale)
        else:
            return CIMNumpyTensor(data=self.data / self.scale / (other.data / other.scale))
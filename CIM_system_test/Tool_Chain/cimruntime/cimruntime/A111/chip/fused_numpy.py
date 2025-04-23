from ...numpy import NumpyRuntime
import numpy
from ...cimtensor import CIMNumpyTensor as CNT
class FusedOpNumpyRuntime(NumpyRuntime):
    
    name = 'FusedOpNumpyRuntime'
    
    def __init__(self):
        self._be = numpy
        self.tensor_type = numpy.ndarray
        
    def fn_fused_conv2d(self, x, **kwargs):
        assert isinstance(x, CNT)
        assert "layer_weight" in kwargs.keys()
        assert "layer_ats" in kwargs.keys()
        assert "layer_info" in kwargs.keys()
        
        layer_info = kwargs["layer_info"]
        layer_weight = kwargs["layer_weight"]
        layer_attr = kwargs["layer_ats"]
        
        in_ = x.data
        in_scale = x.scale
        x = super().fn_conv2d(in_, **layer_weight, **layer_attr)
        
        if layer_info.op.relu != None:
            x = super().fn_relu(x)
        if layer_info.op.pool != None:
            pool_atr = {}
            pool_atr["kernel"] = layer_info.op.pool["kernel"]
            pool_atr["stride"] = layer_info.op.pool["stride"]
            pool_atr["padding"] = layer_info.op.pool["padding"]
            if "dilation" in layer_info.op.pool.keys():
                pool_atr["dilation"] = layer_info.op.pool["dilation"]
            else:
                pool_atr["dilation"] = 1
            if layer_info.op.pool["op_id"] in ['maxpool2d','max_pool2d']:
                x = super().fn_max_pool2d(x, **pool_atr)
            elif layer_info.op.pool["op_id"] in ['global_avg_pool2d']:
                x = super().fn_global_avg_pool2d(x, **pool_atr)
            else:
                raise ValueError(f'暂未实现 池化方式 {layer_info.op.pool.op_id}')
        # print(x.shape)
        # 还原为CIMTensor
        out_, out_scale = CNT.to_cimtensor(data=x).items
        total_scale = in_scale * out_scale
        return CNT(data=out_, scale=total_scale) 
        
    def fn_fused_fc(self, x, **kwargs):
        
        assert isinstance(x, CNT)
        assert "layer_weight" in kwargs.keys()
        assert "layer_info" in kwargs.keys()
        
        layer_info = kwargs["layer_info"]
        layer_weight = kwargs["layer_weight"]
        in_ = x.data
        in_scale = x.scale
        
        x = super().fn_fc(in_, **layer_weight)
        
        if layer_info.op.relu != None:
            x = super().fn_relu(x)
        if layer_info.op.pool != None:
            pool_atr = {}
            pool_atr["kernel"] = layer_info.op.pool["kernel"]
            pool_atr["stride"] = layer_info.op.pool["stride"]
            pool_atr["padding"] = layer_info.op.pool["padding"]
            if "dilation" in layer_info.op.pool.keys():
                pool_atr["dilation"] = layer_info.op.pool["dilation"]
            else:
                pool_atr["dilation"] = 1
            if layer_info.op.pool["op_id"] in ['maxpool2d','max_pool2d']:
                x = super().fn_max_pool2d(x, **pool_atr)
            elif layer_info.op.pool["op_id"] in ['global_avg_pool2d']:
                x = super().fn_global_avg_pool2d(x, **pool_atr)
            else:
                raise ValueError(f'暂未实现 池化方式 {layer_info.op.pool.op_id}')
        
        out_, out_scale = CNT.to_cimtensor(data=x).items
        total_scale = in_scale * out_scale
        
        return CNT(data=out_, scale=total_scale)        
        
    def fn_matmul(self, x, **kwargs):

        assert isinstance(x, CNT)
        in_ = x.data
        in_scale = x.scale
        
        x = super().fn_fc(in_, **kwargs)

        out_, out_scale = CNT.to_cimtensor(data=x).items
        total_scale = in_scale * out_scale
        
        return CNT(data=out_, scale=total_scale)  
# A111 device abstract
from e100_irtool.core.layer import OpLayer
from e100_irtool.core.ref import is_valid_ref
from e100_irtool.core.jsonable import to_json_obj, Jsonable
from e100_irtool.core.type_util import is_integers, is_integer, mixin, \
                                       to_int_tuple, to_cls_obj
from e100_irtool import BaseDevice
from e100_irtool.devices import RramDevice, RramProfile
from itertools import product

class CIMADeviceMappingInfo(Jsonable):

    index = None
    device = None
    address = None

    def __init__(self, *, index, device, address=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('index', to_int_tuple(index, keep_scalar=True),
                      is_integers, not_none=True, min_val=0, ndims=[3])
        # self.set_attr('device', device, is_valid_ref, not_none=True)
        self.set_attr('device', device)
        self.set_attr('address', to_int_tuple(address, keep_scalar=True),
                      is_integers, min_val=0, ndims=[4])


class CIMAMappingInfo(Jsonable):
    
    '''
    mapping时的信息:
        ==============================================================
                            复制平均，抵抗噪声   
        ==============================================================
        col_split_num: 总的列切分次数，如果是SBC，每个复制单元的切分方式相同，如果是CBS，根据切分和地址分配输入
        row_split_num: 总的行切分次数，如果是SBC，每个复制单元的切分方式相同，如果是CBS，根据切分和地址分配输入
        col_repeat_num: 总的列复制次数, A111 暂不支持, 保留参数占位
        row_repeat_num：总的行复制次数
            如果是SBC，则将col_repeat_num与row_repeat_num的乘积作为最终的复制次数；如果是CBS，则按照col_repeat_num进行密堆积复制
            （Note：SBC是CBS在某个切分方式下的特例）
        ===============================================================
                            复制加速，提高并行
        ===============================================================    
        para_diff_array: 通过复制权重的方式放在不同的array中并行
    '''
    
    col_split_num = 1
    row_split_num = 1
    col_repeat_num = 1
    row_repeat_num = 1
    para_diff_array = 1
    in_line_buffer_addr = None
    credit_len = None
    # output_tile_buffer_addr = None
    # in_buf_type = 0
    # out_buf_type = 0
    
    mappings = None

    def __init__(self, *, col_split_num = 1,  row_split_num = 1,
                 col_repeat_num = 1, row_repeat_num = 1, para_diff_array = 1,
                 in_line_buffer_addr = None, credit_len=0,
                #  output_tile_buffer_addr = None , in_buf_type = 0, out_buf_type =0, 
                 mappings = None, **kwargs):
        
        self.set_attr('col_split_num', col_split_num, is_integer, min_val=1)
        self.set_attr('row_split_num', row_split_num, is_integer, min_val=1)
        self.set_attr('col_repeat_num', col_repeat_num, is_integer, min_val=1)
        self.set_attr('row_repeat_num', row_repeat_num, is_integer, min_val=1)
        self.set_attr('para_diff_array', para_diff_array, is_integer, min_val=1)
        self.set_attr('in_line_buffer_addr', in_line_buffer_addr)
        self.set_attr('credit_len', credit_len)
        # self.set_attr('output_tile_buffer_addr', output_tile_buffer_addr)
        # self.set_attr('in_buf_type', in_buf_type, is_integer, min_val=0)
        # self.set_attr('out_buf_type', out_buf_type, is_integer, min_val=0)
        
        if mappings is not None:
            if isinstance(mappings, (tuple, list)):
                d = {}
                for obj in mappings:
                    obj = to_cls_obj(obj, cls=CIMADeviceMappingInfo)
                    d[obj.index] = obj
                # d = {d.index: d for d in
                #      map(DeviceMappingInfo.from_json_obj, mappings)}
                assert len(d) == len(mappings)
                mappings = d
            # print(self.para_diff_array, self.row_split_num, self.col_split_num, list(sorted(d.keys())))
            assert tuple(sorted(mappings)) == tuple(product( 
                    range(self.para_diff_array), range(self.row_split_num), range(self.col_split_num)))
            self.mappings = mappings


    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        if self.mappings is not None:
            d['mappings'] = list(self.mappings.values())
        return to_json_obj(d, **kwargs)

class CIMACalcInfo(Jsonable):
    
    ADC_quant_level = 0
    scale_shift_num = [0]
    scale = [1]
    offset = [0]
    accumulate_shift_num = 0
    data_type = '4bit'
    scale_first = True
    
    def __init__(self, *, ADC_quant_level = 0, scale_shift_num = [0],
                          scale = [1], offset = [0], accumulate_shift_num = 0,
                          data_type = '4bit', scale_first = True,  
                          **kwargs):
        self.set_attr('ADC_quant_level', ADC_quant_level)
        self.set_attr('scale_shift_num', scale_shift_num)
        self.set_attr('scale', scale)
        self.set_attr('offset', offset)
        self.set_attr('accumulate_shift_num', accumulate_shift_num)
        self.set_attr('data_type', data_type)
        self.set_attr('scale_first', scale_first)
        
    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        return to_json_obj(d, **kwargs)

@mixin(OpLayer)
class MappedLayer(Jsonable):

    CIMA_mapping_info = None
    CIMA_calc_info = None

    def __init__(self, *, CIMA_mapping_info=None, CIMA_calc_info=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('CIMA_mapping_info', to_cls_obj(CIMA_mapping_info, CIMAMappingInfo))
        self.set_attr('CIMA_calc_info', to_cls_obj(CIMA_calc_info, CIMACalcInfo))


class CIMAXbDevice(RramDevice):

    kind = 'cima-xb'
    
    profile = dict(
        in_channel = 576,
        out_channel = 128,
        in_bits = 8,          # ?
        out_bits = 8,         # ?
        weight_bits = 4,      # ?
        signed = True,
    )

class CIMAPEClusterDevice(BaseDevice):

    kind = 'cima-pe-cluster'
    
    devices = {
        'xb':{
            'kind': 'cima-xb',
            'number': 16
        }
    }

# class CIMAAdderDevice(BaseDevice):

#     kind = 'cima-adder'


class CIMADRAMDevice(BaseDevice):

    kind = 'cima-dram'

class CIMADMACDevice(RramDevice):

    kind = 'cima-dmac'

    profile = dict(
        in_channel = 256,
        out_channel = 64,
        in_bits = 8,          # ?
        out_bits = 8,         # ?
        weight_bits = 8,      # ?
        signed = True,
    )
    
    
class CIMANodeDevice(BaseDevice):

    kind = 'cima-node'

    devices = {
        'pe-cluster': {
            'kind': 'cima-pe-cluster',
            'number': 4
        },
        # 'adder':{
        #     'kind': 'cima-adder'
        # },
        # 'pooler':{
        #     'kind': 'cima-pooler'
        # },
        'dmac':{
            'kind': 'cima-dmac'
        }
    }
    

            
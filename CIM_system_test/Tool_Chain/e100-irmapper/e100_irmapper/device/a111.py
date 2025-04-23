from e100_irtool import load_ir
from e100_irtool.core.layer import OpLayer
from e100_irtool.core.type_util import is_integers, is_integer, mixin, \
                                       to_int_tuple, to_cls_obj
from e100_irtool.core.ref import is_valid_ref
from e100_irtool.core.jsonable import to_json_obj, Jsonable
from e100_irtool.tools import flatten_layers
from itertools import product

class A111DeviceMappingInfo(Jsonable):

    index = None
    device = None
    address = None

    def __init__(self, *, index, device, address=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('index', to_int_tuple(index, keep_scalar=True),
                      is_integers, not_none=True, min_val=0, ndims=[3])
        self.set_attr('device', device, is_valid_ref, not_none=True)
        self.set_attr('address', to_int_tuple(address, keep_scalar=True),
                      is_integers, min_val=0, ndims=[4])

class A111BufferAddrInfo(Jsonable):

    base = None
    start = None
    end = None

    def __init__(self, *, base, start, end, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('base', base, )
        self.set_attr('start', start,)
        self.set_attr('end', end, )

class A111MappingInfo(Jsonable):
    
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
    input_buffer_addr = None
    output_buffer_addr = None
    in_buf_type = 0
    out_buf_type = 0
    
    mappings = None

    def __init__(self, *, col_split_num = 1,  row_split_num = 1,
                 col_repeat_num = 1, row_repeat_num = 1, para_diff_array = 1,
                 input_buffer_addr = None, output_buffer_addr = None ,
                 in_buf_type = 0, out_buf_type =0,
                 mappings = None, **kwargs):
        
        self.set_attr('col_split_num', col_split_num, is_integer, min_val=1)
        self.set_attr('row_split_num', row_split_num, is_integer, min_val=1)
        self.set_attr('col_repeat_num', col_repeat_num, is_integer, min_val=1)
        self.set_attr('row_repeat_num', row_repeat_num, is_integer, min_val=1)
        self.set_attr('para_diff_array', para_diff_array, is_integer, min_val=1)
        self.set_attr('input_buffer_addr', input_buffer_addr)
        self.set_attr('output_buffer_addr', output_buffer_addr)
        self.set_attr('in_buf_type', in_buf_type, is_integer, min_val=0)
        self.set_attr('out_buf_type', out_buf_type, is_integer, min_val=0)
        
        if mappings is not None:
            if isinstance(mappings, (tuple, list)):
                d = {}
                for obj in mappings:
                    obj = to_cls_obj(obj, cls=A111DeviceMappingInfo)
                    d[obj.index] = obj
                # d = {d.index: d for d in
                #      map(DeviceMappingInfo.from_json_obj, mappings)}
                assert len(d) == len(mappings)
                mappings = d
            # print(self.para_diff_array, self.row_split_num, self.col_split_num, list(sorted(d.keys())))
            
            assert tuple(sorted(mappings)) == tuple(product(
                range(self.para_diff_array), range(self.row_split_num), range(self.col_split_num)))
            self.mappings = mappings

        if input_buffer_addr != None:
            self.input_buffer_addr = to_cls_obj(input_buffer_addr, A111BufferAddrInfo)
        
        if output_buffer_addr != None:
            self.output_buffer_addr = to_cls_obj(output_buffer_addr, A111BufferAddrInfo)
        
    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        if self.mappings is not None:
            d['mappings'] = list(self.mappings.values())
        return to_json_obj(d, **kwargs)

class A111BiasInputInfo(Jsonable):
    
    bias_num = None
    bias_input_value = None

    def __init__(self, *, bias_num = None, bias_input_value = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('bias_num', bias_num)
        self.set_attr('bias_input_value', bias_input_value)

class A111OutputPaddingInfo(Jsonable):
    
    top = 0
    bottom = 0
    left = 0
    right = 0
    
    def __init__(self, *, top=0, bottom=0, left=0, right=0, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('top', top, is_integer)
        self.set_attr('bottom', bottom, is_integer)
        self.set_attr('left', left, is_integer)
        self.set_attr('right', right, is_integer)
        
class A111CalcInfo(Jsonable):
    
    weight_scale = 1
    adc_range = 1
    relu_threshold = 0
    shift_num = 0
    bias_input = None
    buffer_wrap_en = 0
    output_padding = None
    
    def __init__(self, *, weight_scale = 1, adc_range = 1, relu_threshold = 0, shift_num = 0, 
                    bias_input = None, buffer_wrap_en =0, output_padding = None,
                    **kwargs):
        
        self.set_attr('weight_scale', weight_scale)
        self.set_attr('adc_range', adc_range)
        self.set_attr('relu_threshold', relu_threshold)
        self.set_attr('shift_num', shift_num)
        self.set_attr('bias_input', bias_input)
        self.set_attr('buffer_wrap_en', buffer_wrap_en, is_integer)
        self.set_attr('output_padding', output_padding)
        
        if bias_input != None:
            self.bias_input = to_cls_obj(bias_input, cls=A111BiasInputInfo)
        
        if output_padding != None:
            self.output_padding = to_cls_obj(output_padding, cls=A111OutputPaddingInfo)
        
    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        return to_json_obj(d, **kwargs)

@mixin(OpLayer)
class MappedLayer(Jsonable):

    a111_mapping_info = None
    a111_calc_info = None

    def __init__(self, *, a111_mapping_info=None, a111_calc_info=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('a111_mapping_info', to_cls_obj(a111_mapping_info, A111MappingInfo))
        self.set_attr('a111_calc_info', to_cls_obj(a111_calc_info, A111CalcInfo))

# A111 device abstract
from e100_irtool import BaseDevice
from e100_irtool.devices import RramDevice, RramProfile


class A111TileInfo(Jsonable):
    
    tile_mode = 3
    pool0_en = 0
    pool1_en = 0
    pool2_en = 0
    pool3_en = 0
    # in_esram_addr = None
    # out_esram_addr = None
    op_list = None
    # runtime = 'simulation'
    
    def __init__(self, *,  tile_mode = 3, pool0_en = 0, pool1_en = 0, pool2_en = 0,
                pool3_en = 0, op_list = None, in_esram_addr = None, out_esram_addr = None, 
                 **kwargs):
        self.set_attr('tile_mode', tile_mode)
        self.set_attr('pool0_en', pool0_en)
        self.set_attr('pool1_en', pool1_en)
        self.set_attr('pool2_en', pool2_en)
        self.set_attr('pool3_en', pool3_en)
        self.set_attr('op_list', op_list)
        # self.set_attr('in_esram_addr', in_esram_addr)
        # self.set_attr('out_esram_addr', out_esram_addr)
        # self.set_attr('runtime', runtime)
        
    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        return to_json_obj(d, **kwargs)

class A111TileDevice(BaseDevice):

    kind = 'a111-tile'

    devices = {
        'xb': {
            'kind': 'a111-xb',
            'number': 4,
        }
    }

    info = A111TileInfo(
        tile_mode = 3,
        pool0_en = 0,
        pool1_en = 0,
        pool2_en = 0,
        pool3_en = 0,
        op_list = None
    )
    
    def __init__(self, *, info=None, **kwargs):
        super().__init__(**kwargs)
        if info is None:
            info = self.info
        self.set_attr('info', to_cls_obj(info, A111TileInfo))
            

class A111AdderDevice(BaseDevice):

    kind = 'a111-adder'


class A111PoolerDevice(BaseDevice):

    kind = 'a111-pooler'


class A111XbDevice(RramDevice):

    kind = 'a111-xb'

    profile = dict(
        in_channel=320,
        out_channel=128,
        in_bits=8,          # ?
        out_bits=8,         # ?
        weight_bits=4,      # ?
        signed=True,
    )

    # devices = {
    #     'adder': 'a111-adder',
    #     'pooler': 'a111-pooler'
    # }


class A111McuDevice(BaseDevice):

    kind = 'a111-mcu'


class A111NpuDevice(BaseDevice):

    kind = 'a111-npu'

    devices = {
        'a111-tile:0': {
            'kind': 'a111-tile',
        },
        'a111-tile:1': {
            'kind': 'a111-tile',
        },
        'a111-tile:2': {
            'kind': 'a111-tile',
        },
        'a111-tile:3': {
            'kind': 'a111-tile',
        },
        'a111-tile:4': {
            'kind': 'a111-tile',
        },
        'a111-tile:5': {
            'kind': 'a111-tile',
        },
        'a111-tile:6': {
            'kind': 'a111-tile',
        },
        'a111-tile:7': {
            'kind': 'a111-tile',
        },
        'a111-tile:8': {
            'kind': 'a111-tile',
        },
        'a111-tile:9': {
            'kind': 'a111-tile',
        },
        'a111-tile:10': {
            'kind': 'a111-tile',
        },
        'a111-tile:11': {
            'kind': 'a111-tile',
        },
        'mcu': {
            'kind': 'a111-mcu',
        }
    }
    
    ip = None
    
    def __init__(self, *, ip=None, **kwargs):
        super().__init__(**kwargs)
        if ip is None:
            ip = self.ip
        self.set_attr('ip', ip)
        

from e100_irtool import load_ir
from e100_irtool.core.layer import OpLayer
from e100_irtool.core.type_util import is_integers, is_integer, mixin, \
                                       to_int_tuple, to_cls_obj
from e100_irtool.core.ref import is_valid_ref
from e100_irtool.core.jsonable import to_json_obj, Jsonable
from e100_irtool.tools import flatten_layers
from itertools import product

class C200DeviceMappingInfo(Jsonable):

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


class C200MappingInfo(Jsonable):
    '''
    mapping时的信息:
        ==============================================================
                            复制平均，抵抗噪声   
        ==============================================================
        SBC : Split before clone ; CBS : clone before split
        col_split_num: 总的列切分次数，如果是SBC，每个复制单元的切分方式相同，如果是CBS，根据切分和地址分配输入
        row_split_num: 总的行切分次数，如果是SBC，每个复制单元的切分方式相同，如果是CBS，根据切分和地址分配输入
        col_repeat_num: 总的列复制次数
        row_repeat_num：总的行复制次数
            如果是SBC，则将col_repeat_num与row_repeat_num的乘积作为最终的复制次数；如果是CBS，则按照col_repeat_num进行密堆积复制
            （Note：SBC是CBS在某个切分方式下的特例）
        ===============================================================
                            复制加速，提高并行
        ===============================================================    
        para_same_array: 通过window copy的方式放在同一个array中并行
        para_diff_array: 通过复制权重的方式放在不同的array中并行
    '''
    
    col_split_num = 1
    row_split_num = 1
    col_repeat_num = 1
    row_repeat_num = 1
    para_same_array = 1
    para_diff_array = 1
    
    runtime = 'simulation'
    
    mappings = None

    def __init__(self, *, col_split_num = 1,  row_split_num = 1,
                 col_repeat_num = 1, row_repeat_num = 1, para_same_array = 1, para_diff_array = 1, 
                 mappings = None, runtime = 'simulation', **kwargs):
        
        self.set_attr('col_split_num', col_split_num, is_integer, min_val=1)
        self.set_attr('row_split_num', row_split_num, is_integer, min_val=1)
        self.set_attr('col_repeat_num', col_repeat_num, is_integer, min_val=1)
        self.set_attr('row_repeat_num', row_repeat_num, is_integer, min_val=1)
        self.set_attr('para_same_array', para_same_array, is_integer, min_val=1)
        self.set_attr('para_diff_array', para_diff_array, is_integer, min_val=1)
        self.set_attr('runtime', runtime)
        
        if mappings is not None:
            if isinstance(mappings, (tuple, list)):
                d = {}
                for obj in mappings:
                    obj = to_cls_obj(obj, cls=C200DeviceMappingInfo)
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

class C200CalcInfo(Jsonable):
    
    weight_scale = 1
    input_quant_scale = 1.0
    activation_bits = 1
    assigned_output_quant_scale = 1
    output_quant_mode = 1
    it_time = 1
    reg_shift_mode = 0
    output_half_level = 0
    shift_expansion_mode = 'bit_shift'
    noise_scale = 0.0
    adc_clamp = False
    adc_quant = False
    adc_offset = 0
    ADC_LUT = False
    fit_k = 1
    fit_b = 0
    
    def __init__(self, *, weight_scale = 1, input_quant_scale = 1.0, activation_bits = 1, 
                 assigned_output_quant_scale = 1, output_quant_mode = 1, it_time = 1, reg_shift_mode = 0,
                 output_half_level = 0, shift_expansion_mode = 'bit_shift', 
                 noise_scale=noise_scale, adc_offset=0, adc_clamp = False, adc_quant = False, 
                 ADC_LUT = False, fit_k = 1, fit_b = 0,
                 **kwargs):
        self.set_attr('weight_scale', weight_scale)
        self.set_attr('input_quant_scale', input_quant_scale)
        self.set_attr('activation_bits', activation_bits)
        self.set_attr('assigned_output_quant_scale', assigned_output_quant_scale)
        self.set_attr('output_quant_mode', output_quant_mode, is_integer, min_val= 1)
        self.set_attr('it_time', it_time, is_integer, min_val=1 )
        self.set_attr('reg_shift_mode', reg_shift_mode)
        self.set_attr('output_half_level', output_half_level, is_integer, min_val = 0)
        self.set_attr('shift_expansion_mode', shift_expansion_mode)
        self.set_attr('noise_scale', noise_scale)
        self.set_attr('adc_offset', adc_offset)
        self.set_attr('adc_clamp', adc_clamp)
        self.set_attr('adc_quant', adc_quant)
        self.set_attr('ADC_LUT', ADC_LUT)
        self.set_attr('fit_k', fit_k)
        self.set_attr('fit_b', fit_b)
        
    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        return to_json_obj(d, **kwargs)

@mixin(OpLayer)
class MappedLayer(Jsonable):

    c200_mapping_info = None
    c200_calc_info = None

    def __init__(self, *, c200_mapping_info=None, c200_calc_info=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('c200_mapping_info', to_cls_obj(c200_mapping_info, C200MappingInfo))
        self.set_attr('c200_calc_info', to_cls_obj(c200_calc_info, C200CalcInfo))

from e100_irtool.devices.rram import Rram144kDevice

class Rram144kDeviceCluster(Rram144kDevice):
    
    kind = 'rram-144k-cluster'
    
    ip = None

    def __init__(self, *, ip=None, **kwargs):
        super().__init__(**kwargs)
        if ip is None:
            ip = self.ip
        self.set_attr('ip', ip)
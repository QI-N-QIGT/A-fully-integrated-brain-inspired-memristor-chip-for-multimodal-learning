from e100_irtool.core.ir import load_ir, BaseIR
from .helper import *
from .device import *
import re

class IrParser(object):

    def __init__(self, ir=None, cpu_layer=None, specify_device_id_list=None, masked_device_id_list=None):
        
        if isinstance(ir, BaseIR):
            self.ir = ir
        elif isinstance(ir, str):
            self.ir = load_ir(ir)
        else:
            raise ValueError('缺少ir，请输入ir object或者ir file！！！')
        self.node_weight = {}
        self.node_info = {}
        self.hardware_config = {}
        self.cpu_layer = cpu_layer
        self.specify_device_id_list = specify_device_id_list
        self.masked_device_id_list = masked_device_id_list
        self.parser()

    def parser(self):
        
        for i in self.ir.layers.keys():
            layer = self.ir.layers[i]
            if layer.type == 'op':
                if self.cpu_layer != None and i in self.cpu_layer:
                    continue
                op_type = layer.op.op_id
                if op_type in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
                    self.node_weight[i] = get_conv_shape(layer.op)
                    temp1 = get_conv_info(layer)
                    temp1.update(dict(weight_shape=get_conv_shape(layer.op)))
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
                elif op_type in ['linear', 'matmul', 'fc', 'fused_fc']:
                    self.node_weight[i] = get_linear_shape(layer.op)
                    temp1 = get_linear_info(layer)
                    temp1.update(dict(weight_shape=get_linear_shape(layer.op)))
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
                elif op_type in ['split', 'concat', 'fused_concat']:
                    temp1 = get_split_concat_info(layer)
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
                elif op_type in ['add', 'fused_add']:
                    temp1 = get_add_info(layer)
                    ref = []
                    get_layer_ref(layer.inputs, self.ir.layers, ref)
                    temp1.update(dict(ref=ref))
                    self.node_info[i] = temp1
        device = DeviceParser(self.ir.devices, self.specify_device_id_list, self.masked_device_id_list)
        self.hardware_config = device.info

class DeviceParser(object):

    def __init__(self, device, specify_device_id_list, masked_device_id_list):
        
        self.device = device
        self.specify_device_id_list = specify_device_id_list
        self.masked_device_id_list = masked_device_id_list
        self.info = {}
        self.parser()

    def extract_a111_tile_and_xb(self, input_string):
        pattern1 = 'Tile:(\\d+)-XB:(\\d+)'
        pattern2 = 'Tile:(\\d+)'
        match1 = re.match(pattern1, input_string, re.IGNORECASE)
        match2 = re.match(pattern2, input_string, re.IGNORECASE)
        if match1:
            tile_number = int(match1.group(1))
            xb_number = int(match1.group(2))
            if 0 <= tile_number <= 5 and 0 <= xb_number <= 7:
                return (tile_number, xb_number)
            else:
                raise ValueError(f'Tile number: {tile_number}, xb number: {xb_number}, 数值大小不符合要求。')
        elif match2:
            tile_number = int(match2.group(1))
            if 0 <= tile_number <= 5:
                return (tile_number, None)
            else:
                raise ValueError(f'Tile number: {tile_number}, 数值大小不符合要求。')
        else:
            raise ValueError("格式不匹配，请使用 'Tile:数字-XB:数字' 格式")

    def parser(self):
        profile = []
        self.get_device_profile(self.device, profile)
        full_name = []
        full_name = self.get_device_full_name(full_name, None, self.device)
        if 'cima' in full_name[0]:
            count = 0
            for name_ in full_name:
                if 'cima-xb' in name_:
                    count += 1
            device_name = list(self.device.keys())[0]
            mesh_height = self.device[device_name].height
            mesh_width = self.device[device_name].width
        specified_device_id_list = []
        if self.specify_device_id_list != None:
            assert isinstance(self.specify_device_id_list, list)
            for tile_id in self.specify_device_id_list:
                if 'a111-tile' in full_name[0]:
                    specified_device_id_list.append(f'a111-tile:{tile_id * 2}')
                    specified_device_id_list.append(f'a111-tile:{tile_id * 2 + 1}')
                elif 'rram-144k' in full_name[0]:
                    device_kind = full_name[0].split('.')[1].split(':')[0]
                    specified_device_id_list.append(f'{device_kind}:{tile_id}')
        masked_device_id_list = []
        if self.masked_device_id_list != None:
            for device_id in self.masked_device_id_list:
                if 'a111-tile' in full_name[0]:
                    (tile_id, xb_id) = self.extract_a111_tile_and_xb(device_id)
                    if xb_id != None:
                        res_tile_id = xb_id // 4
                        if xb_id >= 4:
                            mapping_xb_id = xb_id - 4
                        else:
                            mapping_xb_id = xb_id
                        mapping_tile_id = 2 * tile_id + res_tile_id
                        if mapping_xb_id in [0, 1]:
                            masked_device_id_list.append(f'a111-tile:{mapping_tile_id}.a111-xb:0')
                            masked_device_id_list.append(f'a111-tile:{mapping_tile_id}.a111-xb:1')
                        elif mapping_xb_id in [2, 3]:
                            masked_device_id_list.append(f'a111-tile:{mapping_tile_id}.a111-xb:2')
                            masked_device_id_list.append(f'a111-tile:{mapping_tile_id}.a111-xb:3')
                    else:
                        mapping_tile_id = [2 * tile_id, 2 * tile_id + 1]
                        for t_id in mapping_tile_id:
                            for xb_id in range(4):
                                masked_device_id_list.append(f'a111-tile:{t_id}.a111-xb:{xb_id}')
                elif 'rram-144k' in full_name[0]:
                    device_kind = full_name[0].split('.')[1].split(':')[0]
                    masked_device_id_list.append(f'{device_kind}:{tile_id}')
                elif 'cima' in full_name[0]:
                    (core_h_index, core_w_index) = device_id
                    core_index = core_h_index * mesh_width + core_w_index
                    device_kind = full_name[0].split('.')[1].split(':')[0]
                    masked_device_id_list.append(f'{device_kind}:{core_index}')
                else:
                    raise ValueError(f'暂不支持 device: {full_name[0]} !!!')
        num = 0
        all_xb_name = []
        for name in full_name:
            if 'xb' in name or 'rram' in name or 'dmac' in name:
                if specified_device_id_list != []:
                    if 'a111-tile' in name:
                        tile_name = name.split('.')[2]
                        if tile_name in specified_device_id_list:
                            num += 1
                            all_xb_name.append(name)
                    elif 'rram-144k' in name:
                        tile_name = name.split('.')[1]
                        if tile_name in specified_device_id_list:
                            num += 1
                            all_xb_name.append(name)
                    else:
                        raise ValueError(f'暂不支持 device: {name} !!!')
                elif masked_device_id_list != []:
                    if 'a111-tile' in name:
                        device_name = '.'.join(name.split('.')[2:])
                        if device_name in masked_device_id_list:
                            continue
                        else:
                            num += 1
                            all_xb_name.append(name)
                    elif 'rram-144k' in name or 'cima' in name:
                        tile_name = name.split('.')[1]
                        if tile_name in masked_device_id_list:
                            continue
                        else:
                            num += 1
                            all_xb_name.append(name)
                    else:
                        raise ValueError(f'暂不支持 device: {name} !!!')
                else:
                    num += 1
                    all_xb_name.append(name)
        rram_profile = profile[0]
        dac_num = rram_profile.in_channel
        if 'dac_num' in rram_profile.__dict__.keys():
            dac_num = rram_profile.dac_num
        adc_num = rram_profile.out_channel
        if 'adc_num' in rram_profile.__dict__.keys():
            adc_num = rram_profile.adc_num
        dac_precision = 4
        if 'dac_precision' in rram_profile.__dict__.keys():
            dac_precision = rram_profile.dac_precision
        self.info = {'name': all_xb_name, 'xb_number': num, 'xb_shape': [rram_profile.out_channel, rram_profile.in_channel], 'adc_num': adc_num, 'dac_num': dac_num, 'dac_precision': dac_precision}
        if len(profile) > 1:
            dmac_profile = profile[1]
            self.info.update(dict(dmac_shape=[dmac_profile.out_channel, dmac_profile.in_channel]))

    def get_device_name(self, name, device):
        
        t = []
        for i in device.keys():
            t.append(device[i].kind)
            if device[i].devices != None:
                self.get_device_name(name, device[i].devices)
        name.append(t)
        return name

    def get_device_profile(self, device, profile):
        
        for i in device.keys():
            if 'profile' in device[i].__dict__.keys():
                profile.append(device[i].profile)
            if device[i].devices != None:
                self.get_device_profile(device[i].devices, profile)

    def get_device_number(self, number, device):
        
        t = []
        for i in device.keys():
            if 'number' in device[i].__dict__.keys():
                t.append(device[i].number)
            else:
                t.append(1)
            if device[i].devices != None:
                self.get_device_number(number, device[i].devices)
        number.append(t)
        return number

    def get_device_full_name(self, name, prefix, device):
        
        count = 0
        for i in device.keys():
            num = 1
            if 'number' in device[i].__dict__.keys():
                num = device[i].number
            for j in range(num):
                if prefix != None:
                    if 'a111-tile' == device[i].kind:
                        suffix = prefix + '.' + device[i].kind + f':{count}'
                        count += 1
                    else:
                        suffix = prefix + '.' + device[i].kind + f':{j}'
                else:
                    suffix = i + '.' + device[i].kind + f':{j}'
                if device[i].devices != None:
                    self.get_device_full_name(name, suffix, device[i].devices)
                else:
                    name.append(suffix)
        return name
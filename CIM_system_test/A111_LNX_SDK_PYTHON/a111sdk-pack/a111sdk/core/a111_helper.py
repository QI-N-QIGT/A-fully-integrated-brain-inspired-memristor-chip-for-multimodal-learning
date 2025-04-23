from .a111_ffi import lib, ffi
import numpy as np
import csv

class TileOpObj:
    
    def __init__(self,
                tile_id, xb_id_list, *,
                
                tile_mode = 3, bypass_mode = 0, bypass_sel = 0, rsv0 = 0, 
                pool_en_list = [0,0,0,0],
                xb_arr_sel = 0, mcu_en_list = [1, 1 ,1, 1], 
                rsv1 = 0x38, slave_mode = 0, mcu_mode = 1, res_load = 0, res_en =0 ,bp_mode = 0, # tile mode 
                
                xbg_mode_list = [0,0,0,0], xbg_para_type_list = [0,0,0,0], xbg_op_mode_list = [0,0,0,0],
                xbg_calc_mode_list=[0,0,0,0], xbg_in_pix_type_list=[3,3,3,3], xbg_out_pix_type_list = [3,3,3,3], 
                xbg_kernel_type_list=[0,0,0,0], xbg_pool_mode_list=[0,0,0,0], xbg_toggle_en0_list=[0,0,0,0], xbg_toggle_bit0_list=[0,0,0,0],
                xbg_tile_buf_en0_list=[0,0,0,0], xbg_tile_cal_en0_list=[0,0,0,0], xbg_fcn_en0_list=[0,0,0,0], xbg_out_kernel_type_list=[1,1,1,1],
                xbg_bias_en_list=[0,0,0,0], xbg_relu_en_list=[0,0,0,0], xbg_bit_mode_list=[0,0,0,0], # xbg mode

                xb_start_column_list = [0], xb_column_num_list = [3], # xb column
                xb_start_row_list = [0], # xb row
                input_addr_list = [0x0,0x0,0x0,0x0], input_len_list = [0x140,0x0,0x0,0x0], in_img_size_list = [[1,1],[1,1],[1,1],[1,1]] , # input 
                output_addr_list = [0x68080000, 0x0, 0x0, 0x0] , out_img_size_list = [[1,1],[0,0],[0,0],[0,0]], xbg_axi_cnt_list = [0x40, 0x0, 0x0, 0x0], # output 
                linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0], linebuf_width_list = [0x140, 0x0, 0x0, 0x0], # linebuf
                relu_th_list = [0x0, 0x0, 0x0, 0x0], act_mode_list = [0x0, 0x0, 0x0, 0x0], shift_list = [0x0, 0x0, 0x0, 0x0], # sfu
                adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], # xb adc range
                res_in_sel = 0, res_out_sel = 0, res_9bit_en = 0, # res_9bit , when relu == false, res_9bit_en = 1
                xb_bias_input_value_list = [[0]], # xb bias input value
                pad_en_list = [0,0,0,0], # padding
                tile_skip = 0 # stride
                ):

        self.tile_id = tile_id
        self.xb_id_list = xb_id_list
        # tile mode
        self.tile_mode = tile_mode
        self.bypass_mode = bypass_mode
        self.bypass_sel = bypass_sel
        self.rsv0 = rsv0
        self.pool_en_list = pool_en_list
        self.xb_arr_sel = xb_arr_sel
        self.mcu_en_list = mcu_en_list
        self.rsv1 = rsv1
        self.slave_mode = slave_mode
        self.mcu_mode = mcu_mode
        self.res_load = res_load
        self.res_en = res_en
        self.bp_mode = bp_mode   
        # xbg mode
        self.xbg_mode_list = xbg_mode_list
        self.xbg_para_type_list = xbg_para_type_list
        self.xbg_op_mode_list = xbg_op_mode_list
        self.xbg_calc_mode_list = xbg_calc_mode_list
        self.xbg_in_pix_type_list = xbg_in_pix_type_list 
        self.xbg_out_pix_type_list = xbg_out_pix_type_list
        self.xbg_kernel_type_list = xbg_kernel_type_list
        self.xbg_pool_mode_list = xbg_pool_mode_list
        self.xbg_toggle_en0_list = xbg_toggle_en0_list
        self.xbg_toggle_bit0_list = xbg_toggle_bit0_list
        self.xbg_tile_buf_en0_list = xbg_tile_buf_en0_list 
        self.xbg_tile_cal_en0_list = xbg_tile_cal_en0_list 
        self.xbg_fcn_en0_list = xbg_fcn_en0_list 
        self.xbg_out_kernel_type_list = xbg_out_kernel_type_list
        self.xbg_bias_en_list = xbg_bias_en_list 
        self.xbg_relu_en_list = xbg_relu_en_list 
        self.xbg_bit_mode_list = xbg_bit_mode_list
        # xb column
        self.xb_start_column_list = xb_start_column_list
        self.xb_column_num_list = xb_column_num_list
        # xb row
        self.xb_start_row_list = xb_start_row_list
        # input 
        self.input_addr_list = input_addr_list 
        self.input_len_list = input_len_list 
        self.in_img_size_list = in_img_size_list 
        # output 
        self.output_addr_list = output_addr_list  
        self.out_img_size_list = out_img_size_list 
        self.xbg_axi_cnt_list = xbg_axi_cnt_list 
        # linebuf
        self.linebuf_addr_offset_list = linebuf_addr_offset_list 
        self.linebuf_width_list = linebuf_width_list 
        # sfu
        self.relu_th_list = relu_th_list
        self.act_mode_list = act_mode_list 
        self.shift_list = shift_list
        # xb adc range
        self.adc_range_list = adc_range_list
        # res_9bit , when relu == false, res_9bit_en = 1
        self.res_in_sel = res_in_sel 
        self.res_out_sel = res_out_sel 
        self.res_9bit_en = res_9bit_en
        # xb bias input value
        self.xb_bias_input_value_list = xb_bias_input_value_list 
        # padding
        self.pad_en_list = pad_en_list 
        # stride
        self.tile_skip = tile_skip

def pad(x, stride = 1):
    assert len(x.shape) == 4
    b, c, h, w = x.shape
    y = np.zeros((b, c, h+2 * stride, w+2 * stride),dtype=np.uint8)
    y[:,:,stride:-1*stride,stride:-1*stride] = x[:,:,:,:]
    return y

# @qinqi implementation
# 给定pt形式的数组，制作rram格式的数组，即pt顺序的格式转为rram4i+j的格式
def pt_sequence_2_rram_discretization(pt_sequence):
    pt_sequence_row, pt_sequence_colum = pt_sequence.shape
    rram_discretization = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum[:, :pt_sequence_colum] = pt_sequence
    # 遍历127次，对应索引为：pt0-rram0,pt1-rram4,pt31-rram124,pt32-rram1,pt126-rram123,pt127-rram127
    for rram_colum in range(127):
        mapping_index = (4 * rram_colum) % 127
        rram_discretization[:, mapping_index] = pt_sequence_128colum[:, rram_colum]
    # 最后一次需要单独赋值，pt127-rram127
    rram_discretization[:, 127] = pt_sequence_128colum[:, 127]
    return rram_discretization

# @qinqi implementation
### run
# pt_weight is from the given csv file, you can read it using pandas library
# pt_weight_2_rram = trans_pt_weight_2_rram(pt_weight).T
def trans_pt_weight_2_rram(pt_weight, pos_sa = 5, neg_sa = 5):
    # 对于pt的3值权重，映射到rram上需要改变具体值，也就是rram = pt x pos_sa 或者rram = pt x neg_sa

    row, colum = pt_weight.shape
    # 转换原始pt权重为2T2R权重
    rram_weight = np.zeros([row * 2, colum])
    pos_weight = np.zeros_like(pt_weight)
    neg_weight = np.zeros_like(pt_weight)
    flag = pt_weight > 0
    pos_weight[flag] = pos_sa
    flag = pt_weight < 0
    neg_weight[flag] = neg_sa
    rram_weight[::2, :] = pos_weight
    rram_weight[1::2, :] = neg_weight
    # 根据芯片mapping策略，重构rram权重（每隔4列存一个数据，满列操作，即128列都用）
    sub_mapping_weight = pt_sequence_2_rram_discretization(rram_weight)
    # 补全其余行的数据，最终芯片mapping的权重需求为640x128的矩阵
    mapping_weight = np.zeros([640, 128])
    mapping_weight[:rram_weight.shape[0]] = sub_mapping_weight
    mapping_weight = mapping_weight.astype(np.uint8)
    return mapping_weight

# @qinqi implementation
def trans_pt_data_2_rram(pt_data, dac_value = 119, data_len= 320):
    row, column = pt_data.shape
    rram_data = np.zeros([row, data_len])
    rram_data[:row, :column] = pt_data * dac_value
    rram_data = rram_data.astype(np.uint8)
    return rram_data

def load_csv(file_path):
    in_data_all = []
    with open(file_path) as f:
        d = csv.reader(f)
        for t in d:
            data_ = []
            for i in t:
                data_.append(int(i))
            in_data_all.append(data_)
    return np.array(in_data_all)

def load_txt(fn, dtype='int32'):
    assert fn
    return np.loadtxt(fn, dtype= dtype, delimiter=',', ndmin = 2)

def save_txt(fn, data):
    assert fn
    np.savetxt(fn, data, delimiter=',', fmt='%d')

def dump_serial(tile_id, file):
    
    print(' 寄存器脚本生成 ===>')
    
    # 配置路由
    router_reg = [0xd0000050]
    # 配置模拟寄存器
    xb_power_ctrl_reg = [0x400, 0x404, 0x408, 0x40c]
    xb_adc_range_reg = [0x414]
    xb_dac_trim_reg = [0x4a0, 0x4a4, 0x4a8, 0x4ac]
    auto_zero_ctrl_reg = [0x4c0, 0x4c4]
    ana_cycle_ctrl_reg = [0x4e0]
    calibra_trim_reg = [0x540]
    forward_calib_cycle_reg = [0x544]
    calibra_ctrl_reg = [0x54c]
    xb0_4_bl_trim_reg = [0x420, 0x424, 0x428, 0x42c]
    xb1_5_bl_trim_reg = [0x430, 0x434, 0x438, 0x43c]
    xb2_6_bl_trim_reg = [0x440, 0x444, 0x448, 0x44c]
    xb3_7_bl_trim_reg = [0x450, 0x454, 0x458, 0x45c]
    # 配置控制寄存器
    tile_mode_reg = [0x10]
    xbg_mode_reg = [0x08, 0x0c, 0x208, 0x20c]
    xb_column_addr_reg = [0x410]
    xb_bias_input_value_reg = [0x1c0, 0x1c4, 0x1c8, 0x1cc, 0x1d0, 0x1d4, 0x1d8, 0x1dc,
                               0x1e0, 0x1e4, 0x1e8, 0x1ec, 0x1f0, 0x1f4, 0x1f8, 0x1fc,
                               0x3c0, 0x3c4, 0x3c8, 0x3cc, 0x3d0, 0x3d4, 0x3d8, 0x3dc,
                               0x3e0, 0x3e4, 0x3e8, 0x3ec, 0x3f0, 0x3f4, 0x3f8, 0x3fc,]
    slave_addr_reg = [0x00, 0x04, 0x200, 0x204]
    img_in_reg = [0x30, 0x34, 0x230, 0x234]
    img_out_reg = [0x38, 0x3c, 0x238, 0x23c]
    tile_buf_addr_reg = [0x54,0x5c, 0x254, 0x25c]
    tlle_buf_used_size_reg = [0x58, 0x60, 0x258, 0x260]
    fifo_threshold_reg = [0x64, 0x68]
    fcn_len_reg = [0x6c, 0x26c]
    linebuf_reg = [0x70, 0x74, 0x270, 0x274]
    in_buf_type_reg = [0x78, 0x278]
    out_buf_type_reg = [0x7c, 0x27c]
    axi_cnt_reg = [0x40, 0x44, 0x240, 0x244]
    sfu_reg = [0x48, 0x4c, 0x248, 0x24c]
    pad_reg = [0x14]
    resnet_reg = [0x2c]
    bl_start_reg = [0x50]
    #配置esram == > tile buffer传输寄存器
    mcu_write_addr_reg = [0xe0]
    mcu_write_len_reg = [0xe4]
    mcu_read_ctrl_reg = [0xe8]
    cmd_que_len_reg = [0x80, 0x88, 0x90, 0x98, 0xa0, 0xa8, 0xb0, 0xb8]
    cmd_que_addr_reg = [0x84, 0x8c, 0x94, 0x9c, 0xa4, 0xac, 0xb4, 0xbc]
   
    #配置开始计算寄存器
    reset_reg = [0x24, 0x224]
    int_req_reg = [0xec]
    #adc暂存结果
    adc_results_reg = [0x680, 0x684, 0x688, 0x68c, 0x690, 0x694, 0x698, 0x69c, 0x6a0]

    base_addr = 0
    if tile_id == 0:
        base_addr = lib.TILE0_CTRL_BASE_ADDR
    elif tile_id == 1:
        base_addr = lib.TILE1_CTRL_BASE_ADDR
    elif tile_id == 2:
        base_addr = lib.TILE2_CTRL_BASE_ADDR
    elif tile_id == 3:
        base_addr = lib.TILE3_CTRL_BASE_ADDR
    elif tile_id == 4:
        base_addr = lib.TILE4_CTRL_BASE_ADDR
    elif tile_id == 5:
        base_addr = lib.TILE5_CTRL_BASE_ADDR
    else:
        raise ValueError(f'{tile_id} 超过tile的个数 !!!')
    
    with open(file, 'w') as f:

        print('start session', file=f)

        print('''
        #####################################################
        ####                                             ####
        ####             配置路由寄存器                   ####
        ####                                             ####
        #####################################################
        ''', file=f)
        for i in router_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        #####################################################
        ####                                             ####
        ####             配置模拟信号寄存器                ####
        ####                                             ####
        #####################################################
        ''', file=f)
        print('## XB电源 & XB使能', file=f)
        for i in xb_power_ctrl_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## ADC输出范围', file=f)
        for i in xb_adc_range_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## DAC trim', file=f)
        for i in xb_dac_trim_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## AUTO ZERO CTRL', file=f)
        for i in auto_zero_ctrl_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Analog CYCLE', file=f)
        for i in ana_cycle_ctrl_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Calibration trim', file=f)
        for i in calibra_trim_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 前向 Calibration cycle', file=f)
        for i in forward_calib_cycle_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Calibration 控制', file=f)
        for i in calibra_ctrl_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB0&4 BL trim', file=f)
        for i in xb0_4_bl_trim_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB1&5 BL trim', file=f)
        for i in xb1_5_bl_trim_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB2&6 BL trim', file=f)
        for i in xb2_6_bl_trim_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB3&7 BL trim', file=f)
        for i in xb3_7_bl_trim_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        #####################################################
        ####                                             ####
        ####             配置TILE控制寄存器               ####
        ####                                             ####
        #####################################################
        ''', file=f)
        print('## TILE mode', file=f)
        for i in tile_mode_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB group mode', file=f)
        for i in xbg_mode_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB 列计算地址', file=f)
        for i in xb_column_addr_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB bias input value', file=f)
        for i in xb_bias_input_value_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## SLAVE 地址（xbg 输出地址）', file=f)
        for i in slave_addr_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输入图像大小', file=f)
        for i in img_in_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输出图像大小', file=f)
        for i in img_out_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Tile buffer 地址', file=f)
        for i in tile_buf_addr_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 使用的 Tile buffer 大小', file=f)
        for i in tlle_buf_used_size_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## FIFO master & slave 阈值', file=f)
        for i in fifo_threshold_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 全连接模式数据长度 ', file=f)
        for i in fcn_len_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Line Buffer 地址 & 长度 （卷积的图片只算宽度，区别于Tile Buffer） ', file=f)
        for i in linebuf_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输入数据 buffer 的类型 （ 0~7 对应 2K~32K）', file=f)
        for i in in_buf_type_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输出数据 buffer 的类型 （ 0~7 对应 2K~32K） ', file=f)
        for i in out_buf_type_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## TILE AXI的数据个数（输出的数据量，整幅图大小包括padding） ', file=f)
        for i in axi_cnt_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 激活函数&移位控制 ', file=f)
        for i in sfu_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## padding控制 ', file=f)
        for i in pad_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 残差连接控制 ', file=f)
        for i in resnet_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## BL 起始行地址 ', file=f)
        for i in bl_start_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        #####################################################
        ####                                             ####
        ####             配置MCU传输寄存器                ####
        ####                                             ####
        #####################################################
        ''', file=f)
        print('## MCU 写到Tile Buffer的地址 ', file=f)
        for i in mcu_write_addr_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU 写到Tile Buffer的数据长度 ', file=f)
        for i in mcu_write_len_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU 读数的基地址 & 读数的次数', file=f)
        for i in mcu_read_ctrl_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU CMD que 读数地址（最多8个通路）', file=f)
        for i in cmd_que_addr_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU CMD que 数据长度（最多8个通路）', file=f)
        for i in cmd_que_len_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        #####################################################
        ####                                             ####
        ####             配置开始计算寄存器               ####
        ####                                             ####
        #####################################################
        ''', file=f)
        print('## 开始计算触发 （reset）', file=f)
        for i in reset_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('set_ic %#x %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 中断响应 ', file=f)
        for i in int_req_reg:
            val = ffi.new("uint32_t*")
            re = lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('wait_ic %#x 1 [3] '% (addr), file=f)
        f.write('\n')
        print('end session', file=f)
    # re = lib.a111_drv_deinit()
    print('寄存器脚本保存成功 ！！！')
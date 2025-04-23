# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from .a111_regs import *
from .a111_enum import *
from . import a111_ffi

"""
#
##
###
####
#####
######################################  LIB A111 SDK API FOR PYTHON ##########################################
#####
####
###
##
#
"""

##############################################################################
#
#                     POWER CONTROL FOR A111 CORE BOARD
#
##############################################################################

def open_a111():
    """打开设备(调用linux系统调用open())，并为A111核心板上电；
        为方便使用，将上电功能整合入设备打开函数中；
    """
    t = a111_ffi.lib.a111_sdk_open_device()
    if t:
        raise ValueError('设备打开失败！！！')
    print("设备已打开！")


def close_a111():
    """关闭设备(调用linux系统调用close())，并关闭A111核心板电源；
        为方便使用，将下电功能整合入设备关闭函数中；
    """
    t = a111_ffi.lib.a111_sdk_close_device()
    if t:
        raise ValueError('设备关闭失败！！！')
    print("设备已关闭！")


def a111_power_on():
    """为A111核心板上电；(需要设备已打开)"""
    t = a111_ffi.lib.a111_power_on()
    if t:
        raise ValueError('A111 Power On 失败！！！')

    return 0


def a111_power_off():
    """为A111核心板下电；(需要设备已打开)"""
    t = a111_ffi.lib.a111_power_off()
    if t:
        raise ValueError('A111 Power Off 失败！！！')

    return 0


def a111_hw_reset():
    """硬件复位"""
    t = a111_ffi.lib.a111_sdk_hw_reset()
    if t:
        raise ValueError('A111 硬件复位 失败！！！')

    return 0


##############################################################################
#
#                               GET BOARD/SDK INFO
#
##############################################################################

def a111_get_hw_prd():
    """获取硬件产品ID，该信息位于FPGA的寄存器中"""
    buf = a111_ffi.ffi.new("char[]", 6)
    t = a111_ffi.lib.a111_sdk_get_hw_prd(buf)
    if t:
        raise ValueError('获取hw prd失败！！！')
    
    return a111_ffi.ffi.string(buf)


def a111_get_hw_ver():
    """获取硬件版本号，该信息位于FPGA的寄存器中"""
    buf = a111_ffi.ffi.new("char[]", 6)
    t = a111_ffi.lib.a111_sdk_get_hw_ver(buf)
    if t:
        raise ValueError('获取hw ver失败！！！')
    
    return a111_ffi.ffi.string(buf)


def a111_get_sw_ver():
    """获取驱动软件版本号，该信息位于驱动源码中"""
    buf = a111_ffi.ffi.new("char[]", 6)
    t = a111_ffi.lib.a111_sdk_get_sw_ver(buf)
    if t:
        raise ValueError('获取sw ver失败！！！')
    
    return a111_ffi.ffi.string(buf)


def a111_get_sw_btime():
    """获取驱动软件编译时间，该信息位于驱动源码中"""
    buf = a111_ffi.ffi.new("char[]", 32)
    t = a111_ffi.lib.a111_sdk_get_sw_btime(buf)
    if t:
        raise ValueError('获取sw built time失败！！！')
    
    return a111_ffi.ffi.string(buf)


##############################################################################
#
#                          READ/WRITE TILE REGISTER
#
##############################################################################

# tile_id: TILE0/TILE1/TILE2/TILE3/TILE4/TILE5
# tile_reg: tile 寄存器的相对地址，参考芯片手册
def a111_read_tile_reg32(tile_id, tile_reg):
    """读取tile寄存器的值"""
    t_val = a111_ffi.ffi.new('uint32_t *', 0)
    t = a111_ffi.lib.a111_read_tile_reg(tile_id, tile_reg, t_val)
    if t:
        raise ValueError('读取Tile寄存器值失败！！！')

    return t_val[0]

def a111_write_tile_reg32(tile_id, tile_reg, t_val):
    """将值写入tile寄存器"""
    t = a111_ffi.lib.a111_write_tile_reg(tile_id, tile_reg, t_val)
    if t:
        raise ValueError('写入Tile寄存器值失败！！！')

    return 0

##############################################################################
#
#                    SET/GET VALUE TO/FROM VOLTAGE SOURCE
#
##############################################################################

# vref: VOUTA/VOUTB/VOUTC/VOUTD/VOUTE
def a111_vsource_set(vref, val):
    """设置电压源"""
    t = a111_ffi.lib.a111_vsource_set(vref, val)
    if t:
        raise ValueError('A111电压源设置失败！！！')
    
    return 0

def a111_vsource_get(vref):
    """读取电压源"""
    val = a111_ffi.ffi.new("float *", 0)
    t = a111_ffi.lib.a111_vsource_get(vref, val)
    if t:
        raise ValueError('A111电压源获取失败！！！')
    
    return int(val[0])


##############################################################################
#
#                    SET/GET VALUE TO/FROM CURRENT SOURCE
#
##############################################################################

# tile_id: TILE0/TILE1/TILE2/TILE3/TILE4/TILE5， 与电压源不同，每个TILE都有其独立的电流源
# iref: IREFN_HVDAC/IREF_IDAC_ADC/IREF_OTA/IREF_IDAC_SA
def a111_isource_set(tile_id, iref, val):
    """设置电流源"""
    t = a111_ffi.lib.a111_isource_set(tile_id, iref, val)
    if t:
        raise ValueError('A111电流源设置失败！！！')
    
    return 0

def a111_isource_get(tile_id, iref):
    """读取电流源"""
    val = a111_ffi.ffi.new("float *", 0)
    t = a111_ffi.lib.a111_isource_get(tile_id, iref, val)
    if t:
        raise ValueError('A111电流源获取失败！！！')
    
    return int(val[0])


##############################################################################
#
#                    SET/GET VALUE TO/FROM CLOCK AND PLL
#
##############################################################################

def a111_clock_set(clk, div, bypass):
    """设置时钟，clk为时钟类型：HFCLK，MCLK，ACLK，ADC_CLK，PERI_CLK，APB_CLK"""
    t = a111_ffi.lib.a111_clock_set(clk, div, bypass)
    if t:
        raise ValueError('A111时钟设置失败！！！')

    return

def a111_clock_get(clk):
    """读取时钟配置"""
    div = a111_ffi.ffi.new("uint8_t *", 0)
    bypass = a111_ffi.ffi.new("bool *", 0)
    t = a111_ffi.lib.a111_clock_get(clk, div, bypass)
    if t:
        raise ValueError('A111时钟获取失败！！！')
    
    return (div[0], bypass[0])

# reg_val类型为 class reg_pll_ctl
def a111_pll_set(pll, reg_val):
    """设置锁相环寄存器，pll为锁相环类型：PLL0，PLL1，PLL2"""
    t = a111_ffi.lib.a111_pll_set(pll, reg_val.bits)
    if t:
        raise ValueError('A111锁相环设置失败！！！')

    return

# 返回值类型为 class reg_pll_ctl
def a111_pll_get(pll):
    """读取锁相环寄存器的值"""
    reg_val = a111_ffi.ffi.new("uint32_t *", 0)
    t = a111_ffi.lib.a111_pll_get(pll, reg_val)
    if t:
        raise ValueError('A111锁相环获取失败！！！')
    val = reg_pll_ctl()
    val.bits = reg_val[0]
    return val

def a111_pll_clk_init(mode):
    """
    初始化锁相环与时钟模式
    CLK_MODE_MAP = 0        MAPPING模式
    CLK_MODE_CALC = 1       计算模式
    """

    # 默认为计算模式
    m = 1
    if mode == 0:
        m = CLK_MODE_MAP
    else:
        m = CLK_MODE_CALC

    re = a111_ffi.lib.a111_pll_clk_init(m)
    if re:
        raise ValueError('锁相环与时钟模式设定失败！！！')
    else:
        print('锁相环与时钟模式设定成功！！！')



##############################################################################
#
#                    INIT HARDWARE SYSTEM(VSRC/ISRC/CLK/PLL)
#
##############################################################################
def a111_hw_sys_init():
    """
    初始化系统硬件，包含电压源、电流源、时钟、锁相环等；（需要先打开设备==》》"open_a111()"）
    """
    re = a111_ffi.lib.a111_sdk_init()
    if re:
        raise ValueError('A111硬件初始化失败！！！')
    
    return

##############################################################################
#
#                    READ/WRITE DATA FROM/TO ESRAM
#
##############################################################################

def get_value_from_esram(data_addr, data_len):
    ''''
    从esram中读出数据，返回numpy数组
    data_addr：相对地址，从0开始；
    data_len：数据长度，单位是字节；
    '''

    read_data = a111_ffi.ffi.new(f"uint8_t [{data_len}]")
    # print('esram 读数地址 : %#x, 读数长度 : %d'%(data_addr, data_len))
    re = a111_ffi.lib.a111_read_data_from_eSRAM(data_addr, read_data, data_len)
    data = np.zeros((data_len,),dtype=np.uint8)
    for i in range(data_len):
        data[i] = read_data[i]

    return data

def set_input_to_esram(in_data, in_data_esram_addr):
    '''
    in_data: 输入数据， numpy数组
    in_data_esram_addr: 输入esram地址偏移量， 基地址为0x68000000
    '''
    if len(in_data.shape) > 1:
        in_data = in_data.flatten()
    in_data_len = in_data.shape[0]
    in_data_ptr = a111_ffi.ffi.cast("uint8_t*", in_data.__array_interface__["data"][0])
    re = a111_ffi.lib.a111_write_data_to_eSRAM(in_data_esram_addr, in_data_ptr, in_data_len)
    if re:
        raise ValueError('写入数据失败！！！')

    # # 校验输入结果
    # read_data = a111_ffi.ffi.new(f"uint8_t [{in_data_len}]")
    # re = a111_ffi.lib.a111_read_data_from_eSRAM(in_data_esram_addr, read_data, in_data_len)
    # data = np.zeros((in_data_len,),dtype=np.uint8)
    # for i in range(in_data_len):
    #     data[i] = read_data[i]
    # if not (data == in_data).all():
    #     # print(data)
    #     # print(in_data)
    #     # print(f'input data = {in_data},\n read_data = {data}')
    #     raise ValueError("写入与读出的数据不一致！！！")
    
    # 输入 int clear
    re = a111_ffi.lib.a111_set_reg32(a111_ffi.lib.INOUT_CTRL_BASE_ADDR + a111_ffi.lib.INOUT_INT_CLR, 0x1)
    if re:
        raise ValueError('INOUT INT CLEAR FAIL!!!')

    # print('输入数据成功，数据长度: %d， 起始地址：%#x !!!' %( in_data_len, 0x68000000 + in_data_esram_addr))


##############################################################################
#
#                               RRAM OPERATIONS
#
##############################################################################

def a111_read_weight(tile_id, xb_id, addr=[0, 0, 640, 128], verbose=True):

    # 初始化权重阵列
    weight = np.zeros((addr[3],addr[2]),dtype=np.uint8)
    # weight地址
    w_data = a111_ffi.ffi.cast("uint8_t*", weight.__array_interface__["data"][0])
    
    # 初始化
    open_a111()
    
    # 初始化时钟 calc
    re = a111_ffi.lib.a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')

    # 设置tile_id
    tile_id = tile_id
    
    # 设置xb_id
    xb_id = xb_id

    # 设置行列起始地址以及行列数量
    row_start = addr[0]
    r_cnt = addr[2]
    col_start = addr[1]
    c_cnt = addr[3]

    print(f"开始 read Tile[{tile_id}].XB[{xb_id}].[{row_start},{r_cnt},{col_start},{c_cnt}] ======>")

    re = a111_ffi.lib.a111_read_array(tile_id, xb_id, row_start, r_cnt, col_start, c_cnt, w_data)
    if re:
        print(re)
        raise ValueError('权重read失败！！！')
    
    # if file == None:
    #     file = f'tile[{tile_id}]_xb[{xb_id}]_weight.txt'
    # save_txt(file, weight.T)

    if verbose:
        print(f'=====================================')
        print(f'tile[{tile_id}].xb[{xb_id}]: ')
        print(f'    weight max: {weight.max()}')
        print(f'    weight min: {weight.min()}')
        print(f'    weight mean: {weight.mean()}')

    # a111_ffi.lib.a111_drv_deinit()

    a111_hw_reset()
    
    return weight.T

def a111_mapping_weight(weight, tile_id, xb_id, addr=[0, 0, 640, 128], program_times=1, tolerance = 1):
    
    h, w = weight.shape
    weight = weight.astype(np.uint8)
    # 转置和芯片写入权重的顺序保持一致
    weight1 = weight.T.copy()
    w_data = a111_ffi.ffi.cast("uint8_t*", weight1.__array_interface__["data"][0])
    print(f'weight max :{weight.max()}')
    print(f'weight min :{weight.min()}')
    print(f'weight shape:{weight.shape}')

    # 初始化
    open_a111()
    
    # 初始化时钟 PROGRAM
    re = a111_ffi.lib.a111_pll_clk_init(CLK_MODE_MAP)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')

    # 设置tile_id
    tile_id = tile_id
    
    # 设置xb_id
    xb_id = xb_id
    
    # 设置行列起始地址以及行列数量
    row_start = addr[0]
    r_cnt = addr[2]
    col_start = addr[1]
    c_cnt = addr[3]
    assert h == r_cnt
    assert w == c_cnt
    # 选择容忍度与模式
    mode = 2
    t1 = time.time()
    for i in range(program_times):
        print(f'开始 program 第 {i} 次======>')
        rate = a111_ffi.lib.a111_program_array(tile_id, xb_id, row_start, r_cnt, col_start, c_cnt, tolerance, w_data, mode)
        if rate < 0.1:
            print('The xbs mapping times out.')
            break
    t2 = time.time()
    print(f'Program times : {program_times}, Program cost time : {t2-t1}s')

    # lib.a111_drv_deinit()

    a111_hw_reset()
    return rate

##############################################################################
#
#                    SET/GET VALUE TO/FROM TILE REGISTER
#
##############################################################################

def set_tile_xb_dac_trim(tile_id, xb_id_list, bltrim_value= [0x77777777, 0x77777777, 0x77777777, 0x77777777]):
    '''
    tile_id: int 0~5
    xb_id: int 0~7
    '''
    bltrim_ptr = a111_ffi.ffi.new("uint32_t []", bltrim_value)
    assert len(xb_id_list) <= 8

    for xb_id in xb_id_list:
        assert isinstance(xb_id, int) and xb_id <= 7
        re = a111_ffi.lib.a111_clr_xb_dac_trim_val(tile_id, xb_id)
        if re:
            raise ValueError("clear dac trim 错误！！！")
        
        re = a111_ffi.lib.a111_set_xb_bl_calc_dac_trim(tile_id, xb_id, bltrim_ptr)
        if re:
            raise ValueError("set dac trim 错误！！！")


def set_tile_xb_adc_range(tile_id, xb_id_list, adc_range_list=[7,7,7,7,7,7,7,7], adc_bypass_id = None):
    """
    设置ADC量程
    """    
    adc_range = 0x0
    assert len(adc_range_list) <= 8

    for index in range(len(xb_id_list)):
        adc_value = adc_range_list[index]
        xb_id = xb_id_list[index]
        assert adc_value <= 7
        assert xb_id <= 7 and xb_id >= 0
        adc_range = adc_range + adc_value * 16**(xb_id % 4)
    
    if adc_bypass_id != None:
        assert isinstance(adc_bypass_id, list)
        for i in adc_bypass_id:
            assert isinstance(i, int)
            assert i <= 7 and i >= 0
            adc_range = adc_range + 8 * 16**(i)

    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ADC_RG_SEL, adc_range)
    if re:
        raise ValueError("set xb adc range 错误！！！")

def set_tile_xb_enable(tile_id, xb_id_list):
    '''
    tile_id : int 0~5
    xb_id_list: list, 0~7
    '''
    ANA_CTRL = [0xfe00fe, 0xfe00fe, 0xfe00fe, 0xfe00fe]
    # 2,3 共享一个power信号
    IsOpen = False
    for xb_id in xb_id_list:
        assert xb_id <= 7 and xb_id >= 0
        assert isinstance(xb_id, int)
        re = xb_id % 2
        div = xb_id // 2

        if xb_id in [2, 3]:
            if not IsOpen:
                # 2号xb和3号xb控制共享
                ANA_CTRL[1] = 0xb800b8
                IsOpen = True
            if re == 0:
                # enable [11] xb: 0, 2, 4, 6
                ANA_CTRL[div]  = ANA_CTRL[div] | 0x800
            elif re == 1:
                # enable [27] xb: 1, 3, 5, 7
                ANA_CTRL[div]  = ANA_CTRL[div] | 0x8000000

            continue
        
        if re == 0:
            ANA_CTRL[div] = ANA_CTRL[div] - 0x46
            # enable [11] xb: 0, 2, 4, 6
            ANA_CTRL[div]  = ANA_CTRL[div] | 0x800
        elif re == 1:
            ANA_CTRL[div] = ANA_CTRL[div] - 0x46 * ( 16 ** (4))
            # enable [27] xb: 1, 3, 5, 7
            ANA_CTRL[div]  = ANA_CTRL[div] | 0x8000000
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL0, ANA_CTRL[0])
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL1, ANA_CTRL[1])
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL2, ANA_CTRL[2])
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL3, ANA_CTRL[3])
    if re:
        raise ValueError('寄存器配置错误！！！')

def set_tile_analog_signal(tile_id, 
                            adc_dout_read_cycle = 1, # 默认值
                            adc_fs_h = 2,
                            adc_fs_l = 2,
                            az_cyc0 = 0x0f,
                            az_cyc1 = 1,
                            az_cyc2 = 0x0f,
                            az_cyc3 = 1,
                            az_cyc6 = 6,
                            fcali_cyc_cycle = 0x7ffff):
    '''
    模拟默认值，暂不需修改
    '''

    # union python
    cali_ctl = reg_cali_ctl()
    cali_ctl.adc_dout_read_cycle = adc_dout_read_cycle
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CALI_CTRL, cali_ctl.bits)

    cyc_ctl0 = reg_cyc_ctl0()
    cyc_ctl0.adc_fs_h = adc_fs_h
    cyc_ctl0.adc_fs_l = adc_fs_l
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CYC_CTRL0, cyc_ctl0.bits)

    az_ctl0 = reg_az_ctl0()
    az_ctl0.az_cyc0 = az_cyc0
    az_ctl0.az_cyc1 = az_cyc1
    az_ctl0.az_cyc2 = az_cyc2
    az_ctl0.az_cyc3 = az_cyc3
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_AZ_CTRL0, az_ctl0.bits)

    az_ctl1 = reg_az_ctl1()
    az_ctl1.az_cyc6 = az_cyc6
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_AZ_CTRL1, az_ctl1.bits)

    fcali_cyc = reg_fcali_cyc()
    fcali_cyc.cycle = fcali_cyc_cycle
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCALI_CYC, fcali_cyc.bits)


def set_router_table(direction=0x04):
    '''
    direction: int 0~4; 0: north, 1: east, 2: south, 3: west, 4: tile;
    '''
    assert direction <= 0x4
    re = a111_ffi.lib.a111_set_reg32(0xd0000050, direction)


def set_tile_mode(tile_id, tile_mode = 3, bypass_mode = 0, bypass_sel = 0, 
                 rsv0 = 0, pool0_en = 0, pool1_en = 0, pool2_en = 0, pool3_en = 0,
                 xb_arr_sel = 0, mcu_en0 = 1, mcu_en1 = 1, mcu_en2 = 1, mcu_en3 = 1, 
                 rsv1 = 0x38, slave_mode = 0, mcu_mode = 1, res_load = 0, res_en =0 ,
                 bp_mode = 0
                 ):
    
    tile_ctl2 = reg_tile_ctl2()
    tile_ctl2.tile_mode = tile_mode
    tile_ctl2.bypass_mode = bypass_mode
    tile_ctl2.bypass_sel = bypass_sel
    tile_ctl2.rsv0 = rsv0
    tile_ctl2.pool0_en = pool0_en
    tile_ctl2.pool1_en = pool1_en
    tile_ctl2.pool2_en = pool2_en
    tile_ctl2.pool3_en = pool3_en
    tile_ctl2.xb_arr_sel = xb_arr_sel
    tile_ctl2.mcu_en0 = mcu_en0
    tile_ctl2.mcu_en1 = mcu_en1
    tile_ctl2.mcu_en2 = mcu_en2
    tile_ctl2.mcu_en3 = mcu_en3
    tile_ctl2.rsv1 = rsv1
    tile_ctl2.slave_mode = slave_mode
    tile_ctl2.mcu_mode = mcu_mode
    tile_ctl2.res_load = res_load
    tile_ctl2.res_en = res_en
    tile_ctl2.bp_mode = bp_mode
    re = a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL2, tile_ctl2.bits)

def set_tile_xbg_mode(tile_id, tile_mode, xbg_mode_list = [0,0,0,0], xbg_para_type_list = [0,0,0,0], xbg_op_mode_list = [0,0,0,0],
                      xbg_calc_mode_list=[3,0,0,0], xbg_in_pix_type_list=[3,3,3,3], xbg_out_pix_type_list = [3,3,3,3], 
                      xbg_kernel_type_list=[0,0,0,0], xbg_pool_mode_list=[0,0,0,0], xbg_toggle_en0_list=[0,0,0,0], xbg_toggle_bit0_list=[1,0,0,0],
                      xbg_tile_buf_en0_list=[1,0,0,0], xbg_tile_cal_en0_list=[1,0,0,0], xbg_fcn_en0_list=[1,0,0,0], xbg_out_kernel_type_list=[1,0,0,0],
                      xbg_bias_en_list=[0,0,0,0], xbg_relu_en_list=[0,0,0,0], xbg_bit_mode_list=[0,0,0,0]):
    
    if tile_mode == 0:
        # 所有xb计算一层
        assert len(xbg_mode_list) >= 1
        assert len(xbg_para_type_list) >= 1
        assert len(xbg_op_mode_list) >= 1
        assert len(xbg_calc_mode_list) >= 1
        assert len(xbg_in_pix_type_list) >= 1
        assert len(xbg_out_pix_type_list) >= 1
        assert len(xbg_kernel_type_list) >= 1
        assert len(xbg_pool_mode_list) >= 1
        assert len(xbg_toggle_en0_list) >= 1
        assert len(xbg_toggle_bit0_list) >= 1
        assert len(xbg_tile_buf_en0_list) >= 1
        assert len(xbg_tile_cal_en0_list) >= 1
        assert len(xbg_fcn_en0_list) >= 1
        assert len(xbg_out_kernel_type_list) >= 1
        assert len(xbg_bias_en_list) >= 1
        assert len(xbg_relu_en_list) >= 1
        assert len(xbg_bit_mode_list) >= 1

        tile_ctl0 = reg_tile_ctl0()
        tile_ctl0.tile_mode = xbg_mode_list[0]
        tile_ctl0.para_type = xbg_para_type_list[0]
        tile_ctl0.op_mode = xbg_op_mode_list[0]
        tile_ctl0.calc_mode = xbg_calc_mode_list[0]
        tile_ctl0.pixel_type_in = xbg_in_pix_type_list[0]
        tile_ctl0.pixel_type_out = xbg_out_pix_type_list[0]
        tile_ctl0.ker_type_in = xbg_kernel_type_list[0]
        tile_ctl0.pool_mode = xbg_pool_mode_list[0]
        tile_ctl0.toggle_en = xbg_toggle_en0_list[0]
        tile_ctl0.toggle_bit = xbg_toggle_bit0_list[0]
        tile_ctl0.buf_en = xbg_tile_buf_en0_list[0]
        tile_ctl0.cal_en = xbg_tile_cal_en0_list[0]
        tile_ctl0.fcn_en = xbg_fcn_en0_list[0]
        tile_ctl0.ker_type_out = xbg_out_kernel_type_list[0]
        tile_ctl0.bias_en = xbg_bias_en_list[0]
        tile_ctl0.rev_en = xbg_relu_en_list[0]
        tile_ctl0.bit_mode = xbg_bit_mode_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, tile_ctl0.bits)
    
    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(xbg_mode_list) >= 2
        assert len(xbg_para_type_list) >= 2
        assert len(xbg_op_mode_list) >= 2
        assert len(xbg_calc_mode_list) >= 2
        assert len(xbg_in_pix_type_list) >= 2
        assert len(xbg_out_pix_type_list) >= 2
        assert len(xbg_kernel_type_list) >= 2
        assert len(xbg_pool_mode_list) >= 2
        assert len(xbg_toggle_en0_list) >= 2
        assert len(xbg_toggle_bit0_list) >= 2
        assert len(xbg_tile_buf_en0_list) >= 2
        assert len(xbg_tile_cal_en0_list) >= 2
        assert len(xbg_fcn_en0_list) >= 2
        assert len(xbg_out_kernel_type_list) >= 2
        assert len(xbg_bias_en_list) >= 2
        assert len(xbg_relu_en_list) >= 2
        assert len(xbg_bit_mode_list) >= 2 

        tile_ctl0 = reg_tile_ctl0()
        tile_ctl0.tile_mode = xbg_mode_list[0]
        tile_ctl0.para_type = xbg_para_type_list[0]
        tile_ctl0.op_mode = xbg_op_mode_list[0]
        tile_ctl0.calc_mode = xbg_calc_mode_list[0]
        tile_ctl0.pixel_type_in = xbg_in_pix_type_list[0]
        tile_ctl0.pixel_type_out = xbg_out_pix_type_list[0]
        tile_ctl0.ker_type_in = xbg_kernel_type_list[0]
        tile_ctl0.pool_mode = xbg_pool_mode_list[0]
        tile_ctl0.toggle_en = xbg_toggle_en0_list[0]
        tile_ctl0.toggle_bit = xbg_toggle_bit0_list[0]
        tile_ctl0.buf_en = xbg_tile_buf_en0_list[0]
        tile_ctl0.cal_en = xbg_tile_cal_en0_list[0]
        tile_ctl0.fcn_en = xbg_fcn_en0_list[0]
        tile_ctl0.ker_type_out = xbg_out_kernel_type_list[0]
        tile_ctl0.bias_en = xbg_bias_en_list[0]
        tile_ctl0.rev_en = xbg_relu_en_list[0]
        tile_ctl0.bit_mode = xbg_bit_mode_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, tile_ctl0.bits)

        tile_ctl1 = reg_tile_ctl0()
        tile_ctl1.tile_mode = xbg_mode_list[1]
        tile_ctl1.para_type = xbg_para_type_list[1]
        tile_ctl1.op_mode = xbg_op_mode_list[1]
        tile_ctl1.calc_mode = xbg_calc_mode_list[1]
        tile_ctl1.pixel_type_in = xbg_in_pix_type_list[1]
        tile_ctl1.pixel_type_out = xbg_out_pix_type_list[1]
        tile_ctl1.ker_type_in = xbg_kernel_type_list[1]
        tile_ctl1.pool_mode = xbg_pool_mode_list[1]
        tile_ctl1.toggle_en = xbg_toggle_en0_list[1]
        tile_ctl1.toggle_bit = xbg_toggle_bit0_list[1]
        tile_ctl1.buf_en = xbg_tile_buf_en0_list[1]
        tile_ctl1.cal_en = xbg_tile_cal_en0_list[1]
        tile_ctl1.fcn_en = xbg_fcn_en0_list[1]
        tile_ctl1.ker_type_out = xbg_out_kernel_type_list[1]
        tile_ctl1.bias_en = xbg_bias_en_list[1]
        tile_ctl1.rev_en = xbg_relu_en_list[1]
        tile_ctl1.bit_mode = xbg_bit_mode_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL3, tile_ctl1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        # 所有xb分为两层
        assert len(xbg_mode_list) >= 3
        assert len(xbg_para_type_list) >= 3
        assert len(xbg_op_mode_list) >= 3
        assert len(xbg_calc_mode_list) >= 3
        assert len(xbg_in_pix_type_list) >= 3
        assert len(xbg_out_pix_type_list) >= 3
        assert len(xbg_kernel_type_list) >= 3
        assert len(xbg_pool_mode_list) >= 3
        assert len(xbg_toggle_en0_list) >= 3
        assert len(xbg_toggle_bit0_list) >= 3
        assert len(xbg_tile_buf_en0_list) >= 3
        assert len(xbg_tile_cal_en0_list) >= 3
        assert len(xbg_fcn_en0_list) >= 3
        assert len(xbg_out_kernel_type_list) >= 3
        assert len(xbg_bias_en_list) >= 3
        assert len(xbg_relu_en_list) >= 3
        assert len(xbg_bit_mode_list) >= 3 

        tile_ctl0 = reg_tile_ctl0()
        tile_ctl0.tile_mode = xbg_mode_list[0]
        tile_ctl0.para_type = xbg_para_type_list[0]
        tile_ctl0.op_mode = xbg_op_mode_list[0]
        tile_ctl0.calc_mode = xbg_calc_mode_list[0]
        tile_ctl0.pixel_type_in = xbg_in_pix_type_list[0]
        tile_ctl0.pixel_type_out = xbg_out_pix_type_list[0]
        tile_ctl0.ker_type_in = xbg_kernel_type_list[0]
        tile_ctl0.pool_mode = xbg_pool_mode_list[0]
        tile_ctl0.toggle_en = xbg_toggle_en0_list[0]
        tile_ctl0.toggle_bit = xbg_toggle_bit0_list[0]
        tile_ctl0.buf_en = xbg_tile_buf_en0_list[0]
        tile_ctl0.cal_en = xbg_tile_cal_en0_list[0]
        tile_ctl0.fcn_en = xbg_fcn_en0_list[0]
        tile_ctl0.ker_type_out = xbg_out_kernel_type_list[0]
        tile_ctl0.bias_en = xbg_bias_en_list[0]
        tile_ctl0.rev_en = xbg_relu_en_list[0]
        tile_ctl0.bit_mode = xbg_bit_mode_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, tile_ctl0.bits)

        tile_ctl1 = reg_tile_ctl0()
        tile_ctl1.tile_mode = xbg_mode_list[1]
        tile_ctl1.para_type = xbg_para_type_list[1]
        tile_ctl1.op_mode = xbg_op_mode_list[1]
        tile_ctl1.calc_mode = xbg_calc_mode_list[1]
        tile_ctl1.pixel_type_in = xbg_in_pix_type_list[1]
        tile_ctl1.pixel_type_out = xbg_out_pix_type_list[1]
        tile_ctl1.ker_type_in = xbg_kernel_type_list[1]
        tile_ctl1.pool_mode = xbg_pool_mode_list[1]
        tile_ctl1.toggle_en = xbg_toggle_en0_list[1]
        tile_ctl1.toggle_bit = xbg_toggle_bit0_list[1]
        tile_ctl1.buf_en = xbg_tile_buf_en0_list[1]
        tile_ctl1.cal_en = xbg_tile_cal_en0_list[1]
        tile_ctl1.fcn_en = xbg_fcn_en0_list[1]
        tile_ctl1.ker_type_out = xbg_out_kernel_type_list[1]
        tile_ctl1.bias_en = xbg_bias_en_list[1]
        tile_ctl1.rev_en = xbg_relu_en_list[1]
        tile_ctl1.bit_mode = xbg_bit_mode_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL3, tile_ctl1.bits)

        tile_ctl2 = reg_tile_ctl0()
        tile_ctl2.tile_mode = xbg_mode_list[2]
        tile_ctl2.para_type = xbg_para_type_list[2]
        tile_ctl2.op_mode = xbg_op_mode_list[2]
        tile_ctl2.calc_mode = xbg_calc_mode_list[2]
        tile_ctl2.pixel_type_in = xbg_in_pix_type_list[2]
        tile_ctl2.pixel_type_out = xbg_out_pix_type_list[2]
        tile_ctl2.ker_type_in = xbg_kernel_type_list[2]
        tile_ctl2.pool_mode = xbg_pool_mode_list[2]
        tile_ctl2.toggle_en = xbg_toggle_en0_list[2]
        tile_ctl2.toggle_bit = xbg_toggle_bit0_list[2]
        tile_ctl2.buf_en = xbg_tile_buf_en0_list[2]
        tile_ctl2.cal_en = xbg_tile_cal_en0_list[2]
        tile_ctl2.fcn_en = xbg_fcn_en0_list[2]
        tile_ctl2.ker_type_out = xbg_out_kernel_type_list[2]
        tile_ctl2.bias_en = xbg_bias_en_list[2]
        tile_ctl2.rev_en = xbg_relu_en_list[2]
        tile_ctl2.bit_mode = xbg_bit_mode_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL4, tile_ctl2.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(xbg_mode_list) >= 4
        assert len(xbg_para_type_list) >= 4
        assert len(xbg_op_mode_list) >= 4
        assert len(xbg_calc_mode_list) >= 4
        assert len(xbg_in_pix_type_list) >= 4
        assert len(xbg_out_pix_type_list) >= 4
        assert len(xbg_kernel_type_list) >= 4
        assert len(xbg_pool_mode_list) >= 4
        assert len(xbg_toggle_en0_list) >= 4
        assert len(xbg_toggle_bit0_list) >= 4
        assert len(xbg_tile_buf_en0_list) >= 4
        assert len(xbg_tile_cal_en0_list) >= 4
        assert len(xbg_fcn_en0_list) >= 4
        assert len(xbg_out_kernel_type_list) >= 4
        assert len(xbg_bias_en_list) >= 4
        assert len(xbg_relu_en_list) >= 4
        assert len(xbg_bit_mode_list) >= 4 

        tile_ctl0 = reg_tile_ctl0()
        tile_ctl0.tile_mode = xbg_mode_list[0]
        tile_ctl0.para_type = xbg_para_type_list[0]
        tile_ctl0.op_mode = xbg_op_mode_list[0]
        tile_ctl0.calc_mode = xbg_calc_mode_list[0]
        tile_ctl0.pixel_type_in = xbg_in_pix_type_list[0]
        tile_ctl0.pixel_type_out = xbg_out_pix_type_list[0]
        tile_ctl0.ker_type_in = xbg_kernel_type_list[0]
        tile_ctl0.pool_mode = xbg_pool_mode_list[0]
        tile_ctl0.toggle_en = xbg_toggle_en0_list[0]
        tile_ctl0.toggle_bit = xbg_toggle_bit0_list[0]
        tile_ctl0.buf_en = xbg_tile_buf_en0_list[0]
        tile_ctl0.cal_en = xbg_tile_cal_en0_list[0]
        tile_ctl0.fcn_en = xbg_fcn_en0_list[0]
        tile_ctl0.ker_type_out = xbg_out_kernel_type_list[0]
        tile_ctl0.bias_en = xbg_bias_en_list[0]
        tile_ctl0.rev_en = xbg_relu_en_list[0]
        tile_ctl0.bit_mode = xbg_bit_mode_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, tile_ctl0.bits)

        tile_ctl1 = reg_tile_ctl0()
        tile_ctl1.tile_mode = xbg_mode_list[1]
        tile_ctl1.para_type = xbg_para_type_list[1]
        tile_ctl1.op_mode = xbg_op_mode_list[1]
        tile_ctl1.calc_mode = xbg_calc_mode_list[1]
        tile_ctl1.pixel_type_in = xbg_in_pix_type_list[1]
        tile_ctl1.pixel_type_out = xbg_out_pix_type_list[1]
        tile_ctl1.ker_type_in = xbg_kernel_type_list[1]
        tile_ctl1.pool_mode = xbg_pool_mode_list[1]
        tile_ctl1.toggle_en = xbg_toggle_en0_list[1]
        tile_ctl1.toggle_bit = xbg_toggle_bit0_list[1]
        tile_ctl1.buf_en = xbg_tile_buf_en0_list[1]
        tile_ctl1.cal_en = xbg_tile_cal_en0_list[1]
        tile_ctl1.fcn_en = xbg_fcn_en0_list[1]
        tile_ctl1.ker_type_out = xbg_out_kernel_type_list[1]
        tile_ctl1.bias_en = xbg_bias_en_list[1]
        tile_ctl1.rev_en = xbg_relu_en_list[1]
        tile_ctl1.bit_mode = xbg_bit_mode_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL1, tile_ctl1.bits)

        tile_ctl2 = reg_tile_ctl0()
        tile_ctl2.tile_mode = xbg_mode_list[2]
        tile_ctl2.para_type = xbg_para_type_list[2]
        tile_ctl2.op_mode = xbg_op_mode_list[2]
        tile_ctl2.calc_mode = xbg_calc_mode_list[2]
        tile_ctl2.pixel_type_in = xbg_in_pix_type_list[2]
        tile_ctl2.pixel_type_out = xbg_out_pix_type_list[2]
        tile_ctl2.ker_type_in = xbg_kernel_type_list[2]
        tile_ctl2.pool_mode = xbg_pool_mode_list[2]
        tile_ctl2.toggle_en = xbg_toggle_en0_list[2]
        tile_ctl2.toggle_bit = xbg_toggle_bit0_list[2]
        tile_ctl2.buf_en = xbg_tile_buf_en0_list[2]
        tile_ctl2.cal_en = xbg_tile_cal_en0_list[2]
        tile_ctl2.fcn_en = xbg_fcn_en0_list[2]
        tile_ctl2.ker_type_out = xbg_out_kernel_type_list[2]
        tile_ctl2.bias_en = xbg_bias_en_list[2]
        tile_ctl2.rev_en = xbg_relu_en_list[2]
        tile_ctl2.bit_mode = xbg_bit_mode_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL3, tile_ctl2.bits)

        tile_ctl3 = reg_tile_ctl0()
        tile_ctl3.tile_mode = xbg_mode_list[3]
        tile_ctl3.para_type = xbg_para_type_list[3]
        tile_ctl3.op_mode = xbg_op_mode_list[3]
        tile_ctl3.calc_mode = xbg_calc_mode_list[3]
        tile_ctl3.pixel_type_in = xbg_in_pix_type_list[3]
        tile_ctl3.pixel_type_out = xbg_out_pix_type_list[3]
        tile_ctl3.ker_type_in = xbg_kernel_type_list[3]
        tile_ctl3.pool_mode = xbg_pool_mode_list[3]
        tile_ctl3.toggle_en = xbg_toggle_en0_list[3]
        tile_ctl3.toggle_bit = xbg_toggle_bit0_list[3]
        tile_ctl3.buf_en = xbg_tile_buf_en0_list[3]
        tile_ctl3.cal_en = xbg_tile_cal_en0_list[3]
        tile_ctl3.fcn_en = xbg_fcn_en0_list[3]
        tile_ctl3.ker_type_out = xbg_out_kernel_type_list[3]
        tile_ctl3.bias_en = xbg_bias_en_list[3]
        tile_ctl3.rev_en = xbg_relu_en_list[3]
        tile_ctl3.bit_mode = xbg_bit_mode_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL4, tile_ctl3.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_xbg_output_addr(tile_id, tile_mode, output_addr_list):
    
    if tile_mode == 0:
        # 所有xb计算一层
        assert len(output_addr_list) >= 1
        slave_addr0 = reg_slave_addr()
        slave_addr0.addr = output_addr_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR0, slave_addr0.bits)
    
    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(output_addr_list) >= 2
        slave_addr0 = reg_slave_addr()
        slave_addr1 = reg_slave_addr()
        slave_addr0.addr = output_addr_list[0]
        slave_addr1.addr = output_addr_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR0, slave_addr0.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR2, slave_addr1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(output_addr_list) >= 3
        slave_addr0 = reg_slave_addr()
        slave_addr2 = reg_slave_addr()
        slave_addr3 = reg_slave_addr()
        slave_addr0.addr = output_addr_list[0]
        slave_addr2.addr = output_addr_list[1]
        slave_addr3.addr = output_addr_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR0, slave_addr0.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR2, slave_addr2.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR3, slave_addr3.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(output_addr_list) >= 4
        slave_addr0 = reg_slave_addr()
        slave_addr1 = reg_slave_addr()
        slave_addr2 = reg_slave_addr()
        slave_addr3 = reg_slave_addr()
        slave_addr0.addr = output_addr_list[0]
        slave_addr1.addr = output_addr_list[1]
        slave_addr2.addr = output_addr_list[2]
        slave_addr3.addr = output_addr_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR0, slave_addr0.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR1, slave_addr1.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR2, slave_addr2.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SLAVE_ADDR3, slave_addr3.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_xbg_input_addr(tile_id, tile_mode, input_addr_list, input_len_list, buf_type_list=[0x6, 0x6, 0x6, 0x6]):

    '''
    input_addr_list: 输入起始地址偏移量，基地址由tile_id 决定
    input_len_list: 如果是卷积，需要考虑 padding 的情况；
    '''

    buf_size_list = []
    buf_size_type = [0x800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x4000, 0x8000]
    # index = 0
    # for i in range(len(input_len_list)):
    #     if input_len_list[i] > 0x4000:
    #         buf_size_list.append(0x8000)
    #     elif input_len_list[i] > 0x3000:
    #         buf_size_list.append(0x4000)
    #     elif input_len_list[i] > 0x2800:
    #         buf_size_list.append(0x3000)
    #     elif input_len_list[i] > 0x2000:
    #         buf_size_list.append(0x2800)
    #     elif input_len_list[i] > 0x1800:
    #         buf_size_list.append(0x2000)
    #     elif input_len_list[i] > 0x1000:
    #         buf_size_list.append(0x1800)
    #     elif input_len_list[i] > 0x800:
    #         buf_size_list.append(0x1000)
    #     else:
    #         buf_size_list.append(0x800)
    for i in buf_type_list:
        assert i <= 7 and i >= 0
        buf_size_list.append(buf_size_type[i])

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(input_addr_list) >= 1
        assert len(input_len_list) >= 1
        tile_buf_ctl0 = reg_tile_buf_ctl()
        tile_buf_ctl0.addr_start = input_addr_list[0]
        tile_buf_ctl0.addr_end = input_addr_list[0] + buf_size_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL00, tile_buf_ctl0.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL10, input_len_list[0])
        # 全连接模式下的输入大小也一起配，具体选择哪个由fcn_en信号决定
        fcn_ctl0 = reg_fcn_ctl()
        fcn_ctl0.len0 = input_len_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCN_CTRL0, fcn_ctl0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(input_addr_list) >= 2
        assert len(input_len_list) >= 2

        tile_buf_ctl0 = reg_tile_buf_ctl()
        tile_buf_ctl0.addr_start = input_addr_list[0]
        tile_buf_ctl0.addr_end = input_addr_list[0] + buf_size_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL00, tile_buf_ctl0.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL10, input_len_list[0])

        tile_buf_ctl1 = reg_tile_buf_ctl()
        tile_buf_ctl1.addr_start = input_addr_list[1]
        tile_buf_ctl1.addr_end = input_addr_list[1] + buf_size_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL40, tile_buf_ctl1.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL50, input_len_list[1])

        # 全连接模式下的输入大小也一起配，具体选择哪个由fcn_en信号决定
        fcn_ctl0 = reg_fcn_ctl()
        fcn_ctl0.len0 = input_len_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCN_CTRL0, fcn_ctl0.bits)

        fcn_ctl1 = reg_fcn_ctl()
        fcn_ctl1.len0 = input_len_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCN_CTRL1, fcn_ctl1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(input_addr_list) >= 3
        assert len(input_len_list) >= 3
        
        tile_buf_ctl0 = reg_tile_buf_ctl()
        tile_buf_ctl0.addr_start = input_addr_list[0]
        tile_buf_ctl0.addr_end = input_addr_list[0] + buf_size_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL00, tile_buf_ctl0.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL10, input_len_list[0])

        tile_buf_ctl1 = reg_tile_buf_ctl()
        tile_buf_ctl1.addr_start = input_addr_list[1]
        tile_buf_ctl1.addr_end = input_addr_list[1] + buf_size_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL40, tile_buf_ctl1.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL50, input_len_list[1])

        tile_buf_ctl2 = reg_tile_buf_ctl()
        tile_buf_ctl2.addr_start = input_addr_list[2]
        tile_buf_ctl2.addr_end = input_addr_list[2] + buf_size_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL60, tile_buf_ctl2.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL70, input_len_list[2])

        # 全连接模式下的输入大小也一起配，具体选择哪个由fcn_en信号决定
        fcn_ctl0 = reg_fcn_ctl()
        fcn_ctl0.len0 = input_len_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCN_CTRL0, fcn_ctl0.bits)
        
        fcn_ctl1 = reg_fcn_ctl()
        fcn_ctl1.len0 = input_len_list[1]
        fcn_ctl1.len1 = input_len_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCN_CTRL1, fcn_ctl1.bits)

    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(input_addr_list) >= 4
        assert len(input_len_list) >= 4

        tile_buf_ctl0 = reg_tile_buf_ctl()
        tile_buf_ctl0.addr_start = input_addr_list[0]
        tile_buf_ctl0.addr_end = input_addr_list[0] + buf_size_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL00, tile_buf_ctl0.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL10, input_len_list[0])

        tile_buf_ctl1 = reg_tile_buf_ctl()
        tile_buf_ctl1.addr_start = input_addr_list[1]
        tile_buf_ctl1.addr_end = input_addr_list[1] + buf_size_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL20, tile_buf_ctl1.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL30, input_len_list[1])

        tile_buf_ctl2 = reg_tile_buf_ctl()
        tile_buf_ctl2.addr_start = input_addr_list[2]
        tile_buf_ctl2.addr_end = input_addr_list[2] + buf_size_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL40, tile_buf_ctl2.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL50, input_len_list[2])

        tile_buf_ctl3 = reg_tile_buf_ctl()
        tile_buf_ctl3.addr_start = input_addr_list[3]
        tile_buf_ctl3.addr_end = input_addr_list[3] + buf_size_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL60, tile_buf_ctl3.bits)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL70, input_len_list[3])

        # 全连接模式下的输入大小也一起配，具体选择哪种模式由fcn_en信号决定
        fcn_ctl0 = reg_fcn_ctl()
        fcn_ctl0.len0 = input_len_list[0]
        fcn_ctl0.len1 = input_len_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCN_CTRL0, fcn_ctl0.bits)
        
        fcn_ctl1 = reg_fcn_ctl()
        fcn_ctl1.len0 = input_len_list[2]
        fcn_ctl1.len1 = input_len_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_FCN_CTRL1, fcn_ctl1.bits)

    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_xbg_input_img_size(tile_id, tile_mode, img_size_list = [[1,1],[1,1],[1,1],[1,1]]):
    '''
    img_size_list: list - [[height,width],[height,width],...]
    '''
    # 校验img_size 是否包含h，w
    for i in img_size_list:
        assert len(i) == 2

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(img_size_list) >= 1
        
        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN0, img_in0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(img_size_list) >= 2
        
        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN0, img_in0.bits)

        img_in1 = reg_img()
        img_in1.width = img_size_list[1][1]
        img_in1.height = img_size_list[1][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN2, img_in1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(img_size_list) >= 3
       
        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN0, img_in0.bits)

        img_in1 = reg_img()
        img_in1.width = img_size_list[1][1]
        img_in1.height = img_size_list[1][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN2, img_in1.bits)

        img_in2 = reg_img()
        img_in2.width = img_size_list[2][1]
        img_in2.height = img_size_list[2][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN3, img_in2.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(img_size_list) >= 4

        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN0, img_in0.bits)

        img_in1 = reg_img()
        img_in1.width = img_size_list[1][1]
        img_in1.height = img_size_list[1][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN1, img_in1.bits)
        
        img_in2 = reg_img()
        img_in2.width = img_size_list[2][1]
        img_in2.height = img_size_list[2][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN2, img_in2.bits)

        img_in3 = reg_img()
        img_in3.width = img_size_list[3][1]
        img_in3.height = img_size_list[3][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_IN3, img_in3.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_xbg_output_img_size(tile_id, tile_mode, img_size_list = [[1,1],[1,1],[1,1],[1,1]]):
    '''
    img_size_list: list - [[height,width],[height,width],...]
    '''
    # 校验img_size 是否包含h，w
    for i in img_size_list:
        assert len(i) == 2

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(img_size_list) >= 1
        
        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT0, img_in0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(img_size_list) >= 2
        
        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT0, img_in0.bits)

        img_in1 = reg_img()
        img_in1.width = img_size_list[1][1]
        img_in1.height = img_size_list[1][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT2, img_in1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(img_size_list) >= 3
       
        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT0, img_in0.bits)

        img_in1 = reg_img()
        img_in1.width = img_size_list[1][1]
        img_in1.height = img_size_list[1][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT2, img_in1.bits)

        img_in2 = reg_img()
        img_in2.width = img_size_list[2][1]
        img_in2.height = img_size_list[2][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT3, img_in2.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(img_size_list) >= 4

        img_in0 = reg_img()
        img_in0.width = img_size_list[0][1]
        img_in0.height = img_size_list[0][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT0, img_in0.bits)

        img_in1 = reg_img()
        img_in1.width = img_size_list[1][1]
        img_in1.height = img_size_list[1][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT1, img_in1.bits)
        
        img_in2 = reg_img()
        img_in2.width = img_size_list[2][1]
        img_in2.height = img_size_list[2][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT2, img_in2.bits)

        img_in3 = reg_img()
        img_in3.width = img_size_list[3][1]
        img_in3.height = img_size_list[3][0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IMG_OUT3, img_in3.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_fifo_threshold(tile_id, master_fifo_thres = 0x20, slave_fifo_thres = 0x20):
    
    fifo_thresh1 = reg_fifo_thresh()
    fifo_thresh1.thresh = master_fifo_thres
    fifo_thresh2 = reg_fifo_thresh()
    fifo_thresh2.thresh = slave_fifo_thres
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_WRM_FIFO_TH, fifo_thresh1.bits)
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_WRS_FIFO_TH, fifo_thresh2.bits)


def set_tile_linebuffer_width(tile_id, tile_mode, linebuf_addr_offset_list, linebuf_width_list):
    '''
    linebuf_addr_offset: list, tile buffer offset 0x0000-0x7ffff base: 0x78000000.
    linebuf_width_list: list, img width data including the padding data and input channel number.
    '''

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(linebuf_addr_offset_list) >= 1
        assert len(linebuf_width_list) >= 1

        buf_ctl0 = reg_buf_ctl()
        buf_ctl0.offset = linebuf_addr_offset_list[0]
        buf_ctl0.length = linebuf_width_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL01, buf_ctl0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(linebuf_addr_offset_list) >= 2
        assert len(linebuf_width_list) >= 2

        buf_ctl0 = reg_buf_ctl()
        buf_ctl0.offset = linebuf_addr_offset_list[0]
        buf_ctl0.length = linebuf_width_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL01, buf_ctl0.bits)

        buf_ctl1 = reg_buf_ctl()
        buf_ctl1.offset = linebuf_addr_offset_list[1]
        buf_ctl1.length = linebuf_width_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL41, buf_ctl1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(linebuf_addr_offset_list) >= 3
        assert len(linebuf_width_list) >= 3

        buf_ctl0 = reg_buf_ctl()
        buf_ctl0.offset = linebuf_addr_offset_list[0]
        buf_ctl0.length = linebuf_width_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL01, buf_ctl0.bits)

        buf_ctl1 = reg_buf_ctl()
        buf_ctl1.offset = linebuf_addr_offset_list[1]
        buf_ctl1.length = linebuf_width_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL41, buf_ctl1.bits)

        buf_ctl2 = reg_buf_ctl()
        buf_ctl2.offset = linebuf_addr_offset_list[2]
        buf_ctl2.length = linebuf_width_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL51, buf_ctl2.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(linebuf_addr_offset_list) >= 4
        assert len(linebuf_width_list) >= 4

        buf_ctl0 = reg_buf_ctl()
        buf_ctl0.offset = linebuf_addr_offset_list[0]
        buf_ctl0.length = linebuf_width_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL01, buf_ctl0.bits)

        buf_ctl1 = reg_buf_ctl()
        buf_ctl1.offset = linebuf_addr_offset_list[1]
        buf_ctl1.length = linebuf_width_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL11, buf_ctl1.bits)

        buf_ctl2 = reg_buf_ctl()
        buf_ctl2.offset = linebuf_addr_offset_list[2]
        buf_ctl2.length = linebuf_width_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL41, buf_ctl2.bits)

        buf_ctl3 = reg_buf_ctl()
        buf_ctl3.offset = linebuf_addr_offset_list[3]
        buf_ctl3.length = linebuf_width_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL51, buf_ctl3.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')


def set_tile_in_buf_type(tile_id, tile_mode, buf_type_list = [0x06, 0x06, 0x06, 0x06]):
    '''
    buf_type_list: list.
    '''
    # 校验buf_type 是否都小于 7
    for i in buf_type_list:
        assert i <= 0x7

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(buf_type_list) >= 1

        buf_type0 = reg_buf_ctl_type()
        buf_type0.type0 = buf_type_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL21, buf_type0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(buf_type_list) >= 2

        buf_type0 = reg_buf_ctl_type()
        buf_type0.type0 = buf_type_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL21, buf_type0.bits)

        buf_type1 = reg_buf_ctl_type()
        buf_type1.type0 = buf_type_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL61, buf_type1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(buf_type_list) >= 3

        buf_type0 = reg_buf_ctl_type()
        buf_type0.type0 = buf_type_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL21, buf_type0.bits)

        buf_type1 = reg_buf_ctl_type()
        buf_type1.type0 = buf_type_list[1]
        buf_type1.type1 = buf_type_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL61, buf_type1.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(buf_type_list) >= 4

        buf_type0 = reg_buf_ctl_type()
        buf_type0.type0 = buf_type_list[0]
        buf_type0.type1 = buf_type_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL21, buf_type0.bits)

        buf_type2 = reg_buf_ctl_type()
        buf_type2.type0 = buf_type_list[2]
        buf_type2.type1 = buf_type_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL61, buf_type2.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_xb_calc_column(tile_id, xb_id_list, xb_start_column_list, xb_column_num_list):
 
    # 校验xb_id_list 是否都小于 7
    for i in xb_id_list:
        assert i <= 7

    assert len(xb_id_list) <= 8
    assert len(xb_start_column_list) <= 8
    assert len(xb_column_num_list) <= 8

    xb_column_addr_ctl = reg_addr_ctl()

    for index in range(len(xb_id_list)):
        xb_id = xb_id_list[index]

        start_column = xb_start_column_list[index]
        column_num = xb_column_num_list[index]
        
        assert start_column <= 0x3
        assert column_num <= 0x3

        if xb_id == 0:
            xb_column_addr_ctl.xb0_start = start_column
            xb_column_addr_ctl.xb0_size = column_num
        elif xb_id == 1:
            xb_column_addr_ctl.xb1_start = start_column
            xb_column_addr_ctl.xb1_size = column_num
        elif xb_id == 2:
            xb_column_addr_ctl.xb2_start = start_column
            xb_column_addr_ctl.xb2_size = column_num
        elif xb_id == 3:
            xb_column_addr_ctl.xb3_start = start_column
            xb_column_addr_ctl.xb3_size = column_num
        elif xb_id == 4:
            xb_column_addr_ctl.xb4_start = start_column
            xb_column_addr_ctl.xb4_size = column_num
        elif xb_id == 5:
            xb_column_addr_ctl.xb5_start = start_column
            xb_column_addr_ctl.xb5_size = column_num
        elif xb_id == 6:
            xb_column_addr_ctl.xb6_start = start_column
            xb_column_addr_ctl.xb6_size = column_num
        elif xb_id == 7:
            xb_column_addr_ctl.xb7_start = start_column
            xb_column_addr_ctl.xb7_size = column_num
    
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ADDR_CTRL, xb_column_addr_ctl.bits)

def set_tile_xb_calc_row(tile_id, xb_id_list, xb_start_row_list = [0]):
 
    # 校验xb_id_list 是否都小于 7
    for i in xb_id_list:
        assert i <= 7

    assert len(xb_id_list) <= 8
    assert len(xb_start_row_list) <= 8
    xb_row_addr_ctl = reg_bl_start()

    for index in range(len(xb_start_row_list)):
        xb_id = xb_id_list[index]
        start_row = xb_start_row_list[index]
        assert start_row <= 0x3
        if xb_id == 0:
            xb_row_addr_ctl.bl0_start = start_row
        elif xb_id == 1:
            xb_row_addr_ctl.bl1_start = start_row
        elif xb_id == 2:
            xb_row_addr_ctl.bl2_start = start_row
        elif xb_id == 3:
            xb_row_addr_ctl.bl3_start = start_row
        elif xb_id == 4:
            xb_row_addr_ctl.bl4_start = start_row
        elif xb_id == 5:
            xb_row_addr_ctl.bl5_start = start_row
        elif xb_id == 6:
            xb_row_addr_ctl.bl6_start = start_row
        elif xb_id == 7:
            xb_row_addr_ctl.bl7_start = start_row
    
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BL_START, xb_row_addr_ctl.bits)

def set_tile_xb_bias_input_value(tile_id, xb_id_list, xb_start_row_list, xb_bias_input_value_list):
 
    # 校验xb_id_list 是否都小于 7
    for i in xb_id_list:
        assert i <= 7

    assert len(xb_id_list) <= 8
    assert len(xb_start_row_list) <= 8
    layer1_bias_input_value = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
    layer2_bias_input_value = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
    layer3_bias_input_value = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
    layer4_bias_input_value = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
    for index in range(len(xb_bias_input_value_list)):
        xb_id = xb_id_list[index]
        start_row = xb_start_row_list[index]
        bias_input_value = xb_bias_input_value_list[index]
        assert start_row <= 0x3
        if xb_id == 0:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer1_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer1_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer1_bias_input_value[0] +=  bias_input_value[0]
                layer1_bias_input_value[1] +=  bias_input_value[1]
                layer1_bias_input_value[2] +=  bias_input_value[2]
                layer1_bias_input_value[3] +=  bias_input_value[3]
        elif xb_id == 1:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer1_bias_input_value[0] +=  bias_input_value[0] * 16 **(4)
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer1_bias_input_value[1] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer1_bias_input_value[4] +=  bias_input_value[0]
                layer1_bias_input_value[5] +=  bias_input_value[1]
                layer1_bias_input_value[6] +=  bias_input_value[2]
                layer1_bias_input_value[7] +=  bias_input_value[3]
        elif xb_id == 2:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer2_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer2_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer2_bias_input_value[0] +=  bias_input_value[0]
                layer2_bias_input_value[1] +=  bias_input_value[1]
                layer2_bias_input_value[2] +=  bias_input_value[2]
                layer2_bias_input_value[3] +=  bias_input_value[3]
        elif xb_id == 3:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer2_bias_input_value[0] +=  bias_input_value[0] * 16 **(4)
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer2_bias_input_value[1] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer2_bias_input_value[4] +=  bias_input_value[0]
                layer2_bias_input_value[5] +=  bias_input_value[1]
                layer2_bias_input_value[6] +=  bias_input_value[2]
                layer2_bias_input_value[7] +=  bias_input_value[3]
        elif xb_id == 4:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer3_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer3_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer3_bias_input_value[0] +=  bias_input_value[0]
                layer3_bias_input_value[1] +=  bias_input_value[1]
                layer3_bias_input_value[2] +=  bias_input_value[2]
                layer3_bias_input_value[3] +=  bias_input_value[3]
        elif xb_id == 5:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer3_bias_input_value[0] +=  bias_input_value[0] * 16 **(4)
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer3_bias_input_value[1] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer3_bias_input_value[4] +=  bias_input_value[0]
                layer3_bias_input_value[5] +=  bias_input_value[1]
                layer3_bias_input_value[6] +=  bias_input_value[2]
                layer3_bias_input_value[7] +=  bias_input_value[3]
        elif xb_id == 6:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer4_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer4_bias_input_value[0] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer4_bias_input_value[0] +=  bias_input_value[0]
                layer4_bias_input_value[1] +=  bias_input_value[1]
                layer4_bias_input_value[2] +=  bias_input_value[2]
                layer4_bias_input_value[3] +=  bias_input_value[3]
        elif xb_id == 7:
            if start_row == 1:
                assert len(bias_input_value) == 1
                assert bias_input_value[0] <= 0xffff
                layer4_bias_input_value[0] +=  bias_input_value[0] * 16 **(4)
            elif start_row == 2:
                assert len(bias_input_value) == 1
                layer4_bias_input_value[1] +=  bias_input_value[0]
            elif start_row == 3:
                assert len(bias_input_value) == 4
                layer4_bias_input_value[4] +=  bias_input_value[0]
                layer4_bias_input_value[5] +=  bias_input_value[1]
                layer4_bias_input_value[6] +=  bias_input_value[2]
                layer4_bias_input_value[7] +=  bias_input_value[3]
    
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_0, layer1_bias_input_value[0])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_1, layer1_bias_input_value[1])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_2, layer1_bias_input_value[2])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_3, layer1_bias_input_value[3])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_4, layer1_bias_input_value[4])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_5, layer1_bias_input_value[5])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_6, layer1_bias_input_value[6])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS0_7, layer1_bias_input_value[7])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_0, layer2_bias_input_value[0])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_1, layer2_bias_input_value[1])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_2, layer2_bias_input_value[2])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_3, layer2_bias_input_value[3])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_4, layer2_bias_input_value[4])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_5, layer2_bias_input_value[5])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_6, layer2_bias_input_value[6])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS1_7, layer2_bias_input_value[7])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_0, layer3_bias_input_value[0])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_1, layer3_bias_input_value[1])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_2, layer3_bias_input_value[2])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_3, layer3_bias_input_value[3])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_4, layer3_bias_input_value[4])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_5, layer3_bias_input_value[5])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_6, layer3_bias_input_value[6])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS2_7, layer3_bias_input_value[7])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_0, layer4_bias_input_value[0])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_1, layer4_bias_input_value[1])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_2, layer4_bias_input_value[2])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_3, layer4_bias_input_value[3])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_4, layer4_bias_input_value[4])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_5, layer4_bias_input_value[5])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_6, layer4_bias_input_value[6])
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BIAS3_7, layer4_bias_input_value[7])

def set_tile_xbg_output_axi_cnt(tile_id, tile_mode, xbg_axi_cnt_list):

    '''
    xbg_axi_cnt_list: list, [value,...], value = （输出图像宽 + 输出图像padding宽）*（输出图像高 + 输出图像padding高） * 输出像素类型 * op_mode_cnt0 .
    '''

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(xbg_axi_cnt_list) >= 1

        in_axi_cnt0 = reg_in_axi_cnt()
        in_axi_cnt0.cnt = xbg_axi_cnt_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT0, in_axi_cnt0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(xbg_axi_cnt_list) >= 2

        in_axi_cnt0 = reg_in_axi_cnt()
        in_axi_cnt0.cnt = xbg_axi_cnt_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT0, in_axi_cnt0.bits)

        in_axi_cnt1 = reg_in_axi_cnt()
        in_axi_cnt1.cnt = xbg_axi_cnt_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT2, in_axi_cnt1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(xbg_axi_cnt_list) >= 3

        in_axi_cnt0 = reg_in_axi_cnt()
        in_axi_cnt0.cnt = xbg_axi_cnt_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT0, in_axi_cnt0.bits)

        in_axi_cnt1 = reg_in_axi_cnt()
        in_axi_cnt1.cnt = xbg_axi_cnt_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT2, in_axi_cnt1.bits)

        in_axi_cnt2 = reg_in_axi_cnt()
        in_axi_cnt2.cnt = xbg_axi_cnt_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT3, in_axi_cnt2.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(xbg_axi_cnt_list) >= 4

        in_axi_cnt0 = reg_in_axi_cnt()
        in_axi_cnt0.cnt = xbg_axi_cnt_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT0, in_axi_cnt0.bits)

        in_axi_cnt1 = reg_in_axi_cnt()
        in_axi_cnt1.cnt = xbg_axi_cnt_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT1, in_axi_cnt1.bits)

        in_axi_cnt2 = reg_in_axi_cnt()
        in_axi_cnt2.cnt = xbg_axi_cnt_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT2, in_axi_cnt2.bits)

        in_axi_cnt3 = reg_in_axi_cnt()
        in_axi_cnt3.cnt = xbg_axi_cnt_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_IN_AXI_CNT3, in_axi_cnt3.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')
    
def set_tile_sfu_ctl(tile_id, tile_mode, relu_th_list = [0,0,0,0], act_mode_list = [0,0,0,0], shift_list = [4,4,4,4]):

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(relu_th_list) >= 1
        assert len(act_mode_list) >= 1
        assert len(shift_list) >= 1

        sfu_ctl0 = reg_sfu_ctl()
        sfu_ctl0.relu_th = relu_th_list[0]
        sfu_ctl0.act_mode = act_mode_list[0]
        sfu_ctl0.shift = shift_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL0, sfu_ctl0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(relu_th_list) >= 2
        assert len(act_mode_list) >= 2
        assert len(shift_list) >= 2

        sfu_ctl0 = reg_sfu_ctl()
        sfu_ctl0.relu_th = relu_th_list[0]
        sfu_ctl0.act_mode = act_mode_list[0]
        sfu_ctl0.shift = shift_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL0, sfu_ctl0.bits)

        sfu_ctl1 = reg_sfu_ctl()
        sfu_ctl1.relu_th = relu_th_list[1]
        sfu_ctl1.act_mode = act_mode_list[1]
        sfu_ctl1.shift = shift_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL2, sfu_ctl1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(relu_th_list) >= 3
        assert len(act_mode_list) >= 3
        assert len(shift_list) >= 3

        sfu_ctl0 = reg_sfu_ctl()
        sfu_ctl0.relu_th = relu_th_list[0]
        sfu_ctl0.act_mode = act_mode_list[0]
        sfu_ctl0.shift = shift_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL0, sfu_ctl0.bits)

        sfu_ctl1 = reg_sfu_ctl()
        sfu_ctl1.relu_th = relu_th_list[1]
        sfu_ctl1.act_mode = act_mode_list[1]
        sfu_ctl1.shift = shift_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL2, sfu_ctl1.bits)

        sfu_ctl2 = reg_sfu_ctl()
        sfu_ctl2.relu_th = relu_th_list[2]
        sfu_ctl2.act_mode = act_mode_list[2]
        sfu_ctl2.shift = shift_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL3, sfu_ctl2.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(relu_th_list) >= 4
        assert len(act_mode_list) >= 4
        assert len(shift_list) >= 4

        sfu_ctl0 = reg_sfu_ctl()
        sfu_ctl0.relu_th = relu_th_list[0]
        sfu_ctl0.act_mode = act_mode_list[0]
        sfu_ctl0.shift = shift_list[0]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL0, sfu_ctl0.bits)

        sfu_ctl1 = reg_sfu_ctl()
        sfu_ctl1.relu_th = relu_th_list[1]
        sfu_ctl1.act_mode = act_mode_list[1]
        sfu_ctl1.shift = shift_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL1, sfu_ctl1.bits)

        sfu_ctl2 = reg_sfu_ctl()
        sfu_ctl2.relu_th = relu_th_list[2]
        sfu_ctl2.act_mode = act_mode_list[2]
        sfu_ctl2.shift = shift_list[2]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL2, sfu_ctl2.bits)

        sfu_ctl3 = reg_sfu_ctl()
        sfu_ctl3.relu_th = relu_th_list[3]
        sfu_ctl3.act_mode = act_mode_list[3]
        sfu_ctl3.shift = shift_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_SFU_CTRL3, sfu_ctl3.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_out_buf_type(tile_id, tile_mode, buf_type_list = [0x06, 0x06, 0x06, 0x06], 
                          buf_wrap_en_list = [0, 0, 0, 0], buf_tog_en_list = [0, 0, 0, 0],
                          ):
    '''
    buf_type_list: list.
    '''
    # 校验buf_type 是否都小于 7
    for i in buf_type_list:
        assert i <= 0x7

    if tile_mode == 0:
        # 所有xb计算一层
        assert len(buf_type_list) >= 1

        buf_ctl3_en0 = reg_buf_ctl3_en()
        buf_ctl3_en0.wrap_en0 = buf_wrap_en_list[0]
        buf_ctl3_en0.tog_en0 = buf_tog_en_list[0]
        buf_ctl3_en0.type0 = buf_type_list[0]
        # for esram
        buf_ctl3_en0.wrap_en2 = 1
        buf_ctl3_en0.type2 = buf_type_list[-1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL31, buf_ctl3_en0.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        assert len(buf_type_list) >= 2

        buf_ctl3_en0 = reg_buf_ctl3_en()
        buf_ctl3_en0.wrap_en0 = buf_wrap_en_list[0]
        buf_ctl3_en0.tog_en0 = buf_tog_en_list[0]
        buf_ctl3_en0.type0 = buf_type_list[0]
        # for esram
        buf_ctl3_en0.wrap_en2 = 1
        buf_ctl3_en0.type2 = buf_type_list[-1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL31, buf_ctl3_en0.bits)

        buf_ctl7_en1 = reg_buf_ctl7_en()
        buf_ctl7_en1.wrap_en3 = buf_wrap_en_list[1]
        buf_ctl7_en1.tog_en3 = buf_tog_en_list[1]
        buf_ctl7_en1.type3 = buf_type_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL71, buf_ctl7_en1.bits)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        assert len(buf_type_list) >= 3

        buf_ctl3_en0 = reg_buf_ctl3_en()
        buf_ctl3_en0.wrap_en0 = buf_wrap_en_list[0]
        buf_ctl3_en0.tog_en0 = buf_tog_en_list[0]
        buf_ctl3_en0.type0 = buf_type_list[0]
        # for esram
        buf_ctl3_en0.wrap_en2 = 1
        buf_ctl3_en0.type2 = buf_type_list[-1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL31, buf_ctl3_en0.bits)

        buf_ctl7_en1 = reg_buf_ctl7_en()
        buf_ctl7_en1.wrap_en3 = buf_wrap_en_list[1]
        buf_ctl7_en1.tog_en3 = buf_tog_en_list[1]
        buf_ctl7_en1.type3 = buf_type_list[1]
        buf_ctl7_en1.wrap_en4 = buf_wrap_en_list[1]
        buf_ctl7_en1.tog_en4 = buf_tog_en_list[1]
        buf_ctl7_en1.type4 = buf_type_list[1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL71, buf_ctl7_en1.bits)
    
    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        assert len(buf_type_list) >= 4

        buf_ctl3_en0 = reg_buf_ctl3_en()
        buf_ctl3_en0.wrap_en0 = buf_wrap_en_list[0]
        buf_ctl3_en0.tog_en0 = buf_tog_en_list[0]
        buf_ctl3_en0.type0 = buf_type_list[0]
        buf_ctl3_en0.wrap_en1 = buf_wrap_en_list[1]
        buf_ctl3_en0.tog_en1 = buf_tog_en_list[1]
        buf_ctl3_en0.type1 = buf_type_list[1]
        # for esram 
        buf_ctl3_en0.wrap_en2 = 1
        buf_ctl3_en0.type2 = buf_type_list[-1]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL31, buf_ctl3_en0.bits)

        buf_ctl7_en1 = reg_buf_ctl7_en()
        buf_ctl7_en1.wrap_en3 = buf_wrap_en_list[2]
        buf_ctl7_en1.tog_en3 = buf_tog_en_list[2]
        buf_ctl7_en1.type3 = buf_type_list[2]
        buf_ctl7_en1.wrap_en4 = buf_wrap_en_list[3]
        buf_ctl7_en1.tog_en4 = buf_tog_en_list[3]
        buf_ctl7_en1.type4 = buf_type_list[3]
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_BUF_CTRL71, buf_ctl7_en1.bits)
    
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def set_tile_resnet_ctl(tile_id, res_in_sel = 1, res_out_sel = 1, res_9bit_en = 0, res_shift = 0,
                        tile_skip = 0, pool_eng_ker = 0, pool_eng_shift = 0, pool_eng_size = 0,
                        pool_eng_en = 0, res_out_mode = 0 ):
    ''''
    TODO
    '''
    res_ctl = reg_res_ctl()
    # res_ctl.bits = 0
    res_ctl.res_in_sel = res_in_sel
    res_ctl.res_out_sel = res_out_sel
    res_ctl.res_9bit_en = res_9bit_en
    res_ctl.res_shift = res_shift
    res_ctl.tile_skip = tile_skip
    res_ctl.pool_eng_ker = pool_eng_ker
    res_ctl.pool_eng_shift = pool_eng_shift
    res_ctl.pool_eng_size = pool_eng_size
    res_ctl.pool_eng_en = pool_eng_en
    res_ctl.res_out_mode = res_out_mode
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESNET_CTRL, res_ctl.bits)

def set_tile_pad_ctl(tile_id, pad_en_list = [0,0,0,0]) :
    '''
    默认选中某一个layer padding时, 其输出结果上下左右都需要padding
    '''
    pad_ctl = reg_padding_ctl()
    pad_en0 = pad_en_list[0]
    pad_en1 = pad_en_list[1]
    pad_en2 = pad_en_list[2]
    pad_en3 = pad_en_list[3] 
    pad_ctl.pad_en0 = pad_en0
    pad_ctl.pad_en1 = pad_en1
    pad_ctl.pad_en2 = pad_en2
    pad_ctl.pad_en3 = pad_en3
    
    if pad_en0:
        pad_ctl.pad0_top = 1
        pad_ctl.pad0_bot = 1
        pad_ctl.pad0_left = 1
        pad_ctl.pad0_right = 1
    if pad_en1:
        pad_ctl.pad1_top = 1
        pad_ctl.pad1_bot = 1
        pad_ctl.pad1_left = 1
        pad_ctl.pad1_right = 1
    if pad_en2:
        pad_ctl.pad2_top = 1
        pad_ctl.pad2_bot = 1
        pad_ctl.pad2_left = 1
        pad_ctl.pad2_right = 1
    if pad_en3:
        pad_ctl.pad3_top = 1
        pad_ctl.pad3_bot = 1
        pad_ctl.pad3_left = 1
        pad_ctl.pad3_right = 1
    
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_PADDING_CTRL, pad_ctl.bits)

def set_mcu_send_data_to_tile(tile_id, rd_addr_offset, rd_length, rd_addr_base, rd_cycle, write_addr, write_length):
    '''
    rd_addr_base: base addr value.0x68
    rd_addr_offset: offset addr value. 0x2000 for dummy
    rd_length: data length.
    rd_cycle: read cycle <= 12. default = 1
    write_addr: tile buf addr.
    write_length: write data length.
    '''
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CMD_QUE0, rd_length)
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CMD_QUE1, rd_addr_offset)

    rd_ctl = reg_read_ctl()
    rd_ctl.base_addr = rd_addr_base
    rd_ctl.read_cycle = rd_cycle
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_READ_CTRL, rd_ctl.bits)

    wr_addr = reg_write_addr()
    wr_addr.addr = write_addr 

    wr_len = reg_write_len()
    wr_len.len = write_length

    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_WR_ADDR, wr_addr.bits)
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_WR_LEN, wr_len.bits)

def set_tile_cali(tile_id, xb_id_list):
    cali_tri = reg_cali_tri()
    cali_tri.fcalc_cali = 1
    for xb_id in xb_id_list:
        if xb_id == 0:
            cali_tri.xb0_tri = 1
        elif xb_id == 1:
            cali_tri.xb1_tri = 1
        elif xb_id == 2:
            cali_tri.xb2_tri = 1
        elif xb_id == 3:
            cali_tri.xb3_tri = 1
        elif xb_id == 4:
            cali_tri.xb4_tri = 1
        elif xb_id == 5:
            cali_tri.xb5_tri = 1
        elif xb_id == 6:
            cali_tri.xb6_tri = 1
        elif xb_id == 7:
            cali_tri.xb7_tri = 1
        else:
            raise ValueError(f"xb_id {xb_id} 超出范围(0~7)")
    # print('calib reg: %#x' % cali_tri.bits)

    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CALI_TRI, cali_tri.bits)
    
def set_tile_run(tile_id, tile_mode):

    # # 清tile的中断
    # a111_ffi.lib.a111_clr_tile_interrupt(tile_id)

    if tile_mode == 0:
        # 所有xb计算一层
        reset_ctl = reg_reset()
        reset_ctl.mcu_fetch_start = 1
        reset_ctl.buf_trans0 = 1
        reset_ctl.buf_trans2 = 1
        
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET0, reset_ctl.bits)

    elif tile_mode == 1:
        # 所有xb分为两层
        reset_ctl = reg_reset()
        reset_ctl.mcu_fetch_start = 1
        reset_ctl.buf_trans0 = 1
        reset_ctl.buf_trans2 = 1
        
        # set reg 0x224 buf_trans3 = 1
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET1, 0x100)
        # run
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET0, reset_ctl.bits)
        

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        reset_ctl = reg_reset()
        reset_ctl.mcu_fetch_start = 1
        reset_ctl.buf_trans0 = 1
        reset_ctl.buf_trans2 = 1
        
        # set reg 0x224 buf_trans3 = 1 , buf_trans4 = 1
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET1, 0x300)
        # run
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET0, reset_ctl.bits)
        

    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        reset_ctl = reg_reset()
        reset_ctl.mcu_fetch_start = 1
        reset_ctl.buf_trans0 = 1
        reset_ctl.buf_trans1 = 1
        reset_ctl.buf_trans2 = 1
        
        # set reg 0x224 buf_trans3 = 1 , buf_trans4 = 1
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET1, 0x300)
        # run
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET0, reset_ctl.bits)
        
    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

    # print('mcu start reg: %#x' % reset_ctl.bits)

def IsCalcDone(tile_id, output_xb_id = 0, log_file=f'Interrupt_reg_error/Register_file.txt', pool_en=False):
    
    assert output_xb_id <= 7
    assert isinstance(output_xb_id,int)

    if pool_en:
        interrupt_sig = 0x2
        
    else:
        interrupt_sig = 0x0
        if (output_xb_id // 2) == 0:
            interrupt_sig = 0x4
        elif (output_xb_id // 2) == 1:
            interrupt_sig = 0x8
        elif (output_xb_id // 2) == 2:
            interrupt_sig = 0x10
        elif (output_xb_id // 2) == 3:
            interrupt_sig = 0x20
        else:
            raise ValueError('中断信号')
    
    t_val = a111_ffi.ffi.new('uint32_t *', 0)
    expired = 0
    
    while(not t_val[0] and expired <= 1000):
        a111_ffi.lib.a111_read_tile_reg(tile_id, a111_ffi.lib.TILE_INT_REQ, t_val)
        t_val[0] = t_val[0] & interrupt_sig
        expired +=1
        time.sleep(0.00000001)

    if (expired > 1000):
        print(" !!! 接收计算中断失败 !!! ")
        file_path = log_file.split('/')
        file_path = '/'.join(file_path[0:-1])
        print(f" !!! 保存当前寄存器配置 !!!")
        print(f" !!! 寄存器文件路径：[{file_path}] !!!")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dump_TileReg(tile_id = tile_id, file=log_file)
        
        # reset hardware 
        a111_hw_reset()
        # ret = a111_ffi.lib.a111_drv_deinit()
        exit(1)

    # 清理中断信号
    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_INT_CLR, interrupt_sig)

    return 1

def Reset_TileOp(tile_id, tile_mode):

    if tile_mode == 0:
        # 所有xb计算一层
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL0, 0)

    elif tile_mode == 1:
        # 所有xb分为两层
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL0, 0)

        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL3, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL2, 0)

    elif tile_mode == 2:
        # 所有xb分为三层， 第一层为4xb, 后两层分别为2xb
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL0, 0)

        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL3, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL2, 0)

        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL4, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL3, 0)

    elif tile_mode == 3:
        # 所有xb分为四层， 每层2xb
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL0, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL0, 0)
        
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL1, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL1, 0)

        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL3, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL2, 0)

        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_CTRL4, 0x0)
        a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_ANA_CTRL3, 0)

    else:
        raise ValueError(f'tile mode {tile_mode} 不支持 ！！！')

def TileOp( tile_id, xb_id_list, *,
            
            tile_mode = 3, bypass_mode = 0, bypass_sel = 0, rsv0 = 0, 
            pool0_en = 0, pool1_en = 0, pool2_en = 0, pool3_en = 0,
            xb_arr_sel = 0, mcu_en0 = 1, mcu_en1 = 1, mcu_en2 = 1, mcu_en3 = 1, 
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
            adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], adc_bypass_id = None, # xb adc range
            res_in_sel = 0, res_out_sel = 0, res_9bit_en = 0, # res_9bit , when relu == false, res_9bit_en = 1
            xb_bias_input_value_list = [[0]], # xb bias input value
            pad_en_list = [0,0,0,0], # padding
            tile_skip = 0 # stride
            ):
    
    # 清tile的中断
    a111_ffi.lib.a111_clr_tile_interrupt(tile_id)

    # 1.配置tile的路由
    set_router_table()

    # 2.配置tile模拟信号
    set_tile_analog_signal(tile_id)

    # 3.配置tile中各xb计算信号，
    set_tile_xb_enable(tile_id, xb_id_list) # power on and enable
    set_tile_xb_dac_trim(tile_id, xb_id_list)
    set_tile_xb_adc_range(tile_id, xb_id_list, adc_range_list = adc_range_list, adc_bypass_id = adc_bypass_id)
    set_tile_xb_calc_column(tile_id, xb_id_list, xb_start_column_list, xb_column_num_list)
    set_tile_xb_calc_row(tile_id, xb_id_list, xb_start_row_list)
    set_tile_xb_bias_input_value(tile_id, xb_id_list, xb_start_row_list, xb_bias_input_value_list)

    # 4.配置tile的工作模式
    set_tile_mode(tile_id, tile_mode, bypass_mode = bypass_mode, bypass_sel = bypass_sel, rsv0 = rsv0, 
            pool0_en = pool0_en, pool1_en = pool1_en, pool2_en = pool2_en, pool3_en = pool3_en,
            xb_arr_sel = xb_arr_sel, mcu_en0 = mcu_en0, mcu_en1 = mcu_en1, mcu_en2 = mcu_en2, mcu_en3 = mcu_en3, 
            rsv1 = rsv1, slave_mode = slave_mode, mcu_mode = mcu_mode, res_load = res_load, res_en =res_en ,bp_mode = bp_mode)
    
    # 5.配置tile中各层的工作模式
    set_tile_xbg_mode(tile_id, tile_mode, xbg_mode_list =  xbg_mode_list, xbg_para_type_list = xbg_para_type_list , 
                      xbg_op_mode_list = xbg_op_mode_list, xbg_calc_mode_list= xbg_calc_mode_list, xbg_in_pix_type_list=xbg_in_pix_type_list,
                      xbg_out_pix_type_list = xbg_out_pix_type_list, xbg_kernel_type_list=xbg_kernel_type_list, xbg_pool_mode_list=xbg_pool_mode_list,
                      xbg_toggle_en0_list = xbg_toggle_en0_list, xbg_toggle_bit0_list = xbg_toggle_bit0_list, xbg_tile_buf_en0_list=xbg_tile_buf_en0_list,
                      xbg_tile_cal_en0_list = xbg_tile_cal_en0_list, xbg_fcn_en0_list = xbg_fcn_en0_list, xbg_out_kernel_type_list = xbg_out_kernel_type_list,
                      xbg_bias_en_list = xbg_bias_en_list, xbg_relu_en_list = xbg_relu_en_list, xbg_bit_mode_list=xbg_bit_mode_list)

    # 6.配置tile中各层的输入输出 ，全连接与卷积同时配置
    set_tile_xbg_input_addr(tile_id, tile_mode, input_addr_list, input_len_list)
    set_tile_xbg_input_img_size(tile_id, tile_mode, in_img_size_list)
    set_tile_xbg_output_addr(tile_id, tile_mode, output_addr_list)
    set_tile_xbg_output_img_size(tile_id, tile_mode, out_img_size_list)
    set_tile_xbg_output_axi_cnt(tile_id, tile_mode, xbg_axi_cnt_list)
    
    # 7.配置tile计算时的buffer类型
    set_tile_in_buf_type(tile_id, tile_mode)
    set_tile_out_buf_type(tile_id, tile_mode)

    # 8.配置tile中各层的linebuffer width
    set_tile_linebuffer_width(tile_id,tile_mode, linebuf_addr_offset_list, linebuf_width_list)

    # 9.配置tile的FIFO 阈值
    set_tile_fifo_threshold(tile_id)

    # 10.配置tile的激活函数,移位
    set_tile_sfu_ctl(tile_id, tile_mode, relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list)

    # 11.配置tile的padding
    set_tile_pad_ctl(tile_id, pad_en_list)

    # 12.配置tile的resnet ctl
    set_tile_resnet_ctl(tile_id, res_in_sel = res_in_sel , res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, tile_skip = tile_skip)


def get_tileop_results_FPGA(out_addr_esram, out_len, num_type = '8bit', out_len_effective = [0,128], raw_data=False, op_type='FC', bypass= False):
    
    # 最后输出 32byte 对齐
    assert out_len % 32 == 0
    # 最后输出地址 256 byte 对齐
    assert out_addr_esram % 256 == 0
    # 有效输出
    assert len(out_len_effective) == 2
    assert (out_len_effective[0] + out_len_effective[1])  <= out_len
    
    rd_buf = a111_ffi.ffi.new(f"int16_t [{out_len}]")
    a111_ffi.lib.a111_read_data_from_eSRAM_9bit_trans( out_addr_esram, rd_buf, out_len)
 
    # FPGA 实现与原始版本不一致， 当输出‘9bit’时， out_len需要除 2
    if num_type == '9bit':
        out_len = out_len // 2
    
    # # 初始化输出
    # output = np.zeros((out_len))
    
    # 从ffi buffer中取数
    tmp = a111_ffi.ffi.buffer(rd_buf, (out_len * 2))
    output = np.frombuffer(tmp, dtype=np.int16, count=out_len)
    
    # 保存原始数据 
    raw_data_value = ""
    if raw_data:
        for i in range(out_len):
            # output[i] = rd_buf[i]
            # if raw_data:
            str_ = str(hex(int(rd_buf[i])))[2:]
            str_ = str_.zfill(2)
            # 打印出的raw_data为高位在前，低位在后；而真实硬件的ADC的输出码字是低位在前，高位在后
            raw_data_value += str_ + ','
    
    if op_type == 'FC':
        if raw_data:
            return output[out_len_effective[0]:out_len_effective[1]], raw_data_value

        return output[out_len_effective[0]:out_len_effective[1]]
    elif op_type == 'CONV':
        if raw_data:
            return output, raw_data_value
        return output
    else:
        raise ValueError(f"不支持的 op type: {op_type} !!! 仅支持 ['FC', 'CONV'] !!!")

def get_tileop_results(out_addr_esram, out_len, num_type = '8bit', out_len_effective = [0,128], raw_data=False, op_type='FC', bypass= False):
    
    # 最后输出 32byte 对齐
    assert out_len % 32 == 0
    # 最后输出地址 256 byte 对齐
    assert out_addr_esram % 256 == 0
    # 有效输出
    assert len(out_len_effective) == 2
    assert (out_len_effective[0] + out_len_effective[1])  <= out_len
    
    rd_buf = a111_ffi.ffi.new(f"uint8_t[{out_len}]")
    a111_ffi.lib.a111_read_data_from_eSRAM( out_addr_esram, rd_buf, out_len)
     
    # 初始化输出
    output = np.zeros((out_len))

    raw_data_value = ""
    for i in range(out_len):
        # # print(rd_buf[i])
        output[i] = rd_buf[i]
        if raw_data:
            str_ = str(hex(int(rd_buf[i])))[2:]
            str_ = str_.zfill(2)
            # 打印出的raw_data为高位在前，低位在后；而真实硬件的ADC的输出码字是低位在前，高位在后
            raw_data_value += str_ + ','

    # bypass模式下 输出每一bit的权重值
    w_bypass = [1, 2, 4, 6, 12, 20, 36, 62, 112]
    
    if num_type == '9bit':
        # 以9bit输出时，将 最终的 out_len 对齐到32 byte， 超过的部分将被均分
        num_ = out_len // 32
        # 每次取32byte的中的前18byte（9bit * 16）
        byte_num_batch = 18 
        # 初始化新的向量
        out_value = np.zeros((out_len // 2,))
        # 将内存中 8bit 的数转换为 9bit
        for i in range(num_):
            bin_code = ""
            for j in range(byte_num_batch):
                str_ = str(bin(int(output[i * 32 + j])))[2:]
                if len(str_) < 8:
                    str_ = str_.zfill(8)
                # bin code 的结果与raw_data值的顺序是相反，例如raw_data顺序为0a, 1b, 2c; 
                # 则bin code 的输出结果为 2c, 1b, 0a；reverse之后为正常顺序，
                # 原因是为了方便匹配相邻两个8bit的值形成9bit，对应硬件的输出
                bin_code = str_ + bin_code
            # 逆序
            bin_code = ''.join(reversed(bin_code))
            assert len(bin_code) == (16 * 9)
            for k in range(16):
                value = 0
                sign_bit = int(bin_code[k*9 + 8])
                for m in range(8):
                    if bypass:
                        # 0 --> -w ; 1 --> +w
                        value += (2 * int(bin_code[9*k + m]) - 1) * w_bypass[m]
                    else:
                        value += int(bin_code[9*k + m]) * 2 ** (m)
                if bypass:
                    value += (2 * sign_bit - 1) * w_bypass[8] 
                    out_value[i*16 + k] = value
                else: 
                    if sign_bit == 0:
                        out_value[i*16 + k] = value
                    else:
                        out_value[i*16 + k] =  value - 256

        output = out_value
    elif num_type == '8bit':
        pass
    else:
        raise ValueError(f"不支持 num type: {num_type}, 只有 '8bit' 与 '9bit' 两种模式!!!")
    
    if op_type == 'FC':
        if raw_data:
            return output[out_len_effective[0]:out_len_effective[1]], raw_data_value

        return output[out_len_effective[0]:out_len_effective[1]]
    elif op_type == 'CONV':
        if raw_data:
            return output, raw_data_value
        return output
    else:
        raise ValueError(f"不支持的 op type: {op_type} !!! 仅支持 ['FC', 'CONV'] !!!")


def dump_TileReg(tile_id, file):

    print('寄存器状态快照 ===>')

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
                               0x3e0, 0x3e4, 0x3e8, 0x3ec, 0x3f0, 0x3f4, 0x3f8, 0x3fc, ]
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
        base_addr = a111_ffi.lib.TILE0_CTRL_BASE_ADDR
    elif tile_id == 1:
        base_addr = a111_ffi.lib.TILE1_CTRL_BASE_ADDR
    elif tile_id == 2:
        base_addr = a111_ffi.lib.TILE2_CTRL_BASE_ADDR
    elif tile_id == 3:
        base_addr = a111_ffi.lib.TILE3_CTRL_BASE_ADDR
    elif tile_id == 4:
        base_addr = a111_ffi.lib.TILE4_CTRL_BASE_ADDR
    elif tile_id == 5:
        base_addr = a111_ffi.lib.TILE5_CTRL_BASE_ADDR
    else:
        raise ValueError(f'{tile_id} 超过tile的个数 !!!')
    
    val = a111_ffi.ffi.new("uint32_t *")
    with open(file, 'w') as f:
        print('''
        *****************************************************
        ****                                             ****
        ****             配置模拟信号寄存器                ****
        ****                                             ****
        *****************************************************
        ''', file=f)
        print('## XB电源 & XB使能', file=f)
        for i in xb_power_ctrl_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## ADC输出范围', file=f)
        for i in xb_adc_range_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## DAC trim', file=f)
        for i in xb_dac_trim_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## AUTO ZERO CTRL', file=f)
        for i in auto_zero_ctrl_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Analog CYCLE', file=f)
        for i in ana_cycle_ctrl_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Calibration trim', file=f)
        for i in calibra_trim_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 前向 Calibration cycle', file=f)
        for i in forward_calib_cycle_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Calibration 控制', file=f)
        for i in calibra_ctrl_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB0&4 BL trim', file=f)
        for i in xb0_4_bl_trim_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB1&5 BL trim', file=f)
        for i in xb1_5_bl_trim_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB2&6 BL trim', file=f)
        for i in xb2_6_bl_trim_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB3&7 BL trim', file=f)
        for i in xb3_7_bl_trim_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        *****************************************************
        ****                                             ****
        ****             配置TILE控制寄存器               ****
        ****                                             ****
        *****************************************************
        ''', file=f)
        print('## TILE mode', file=f)
        for i in tile_mode_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB group mode', file=f)
        for i in xbg_mode_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB 列计算地址', file=f)
        for i in xb_column_addr_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## XB bias input value', file=f)
        for i in xb_bias_input_value_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## SLAVE 地址（xbg 输出地址）', file=f)
        for i in slave_addr_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输入图像大小', file=f)
        for i in img_in_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输出图像大小', file=f)
        for i in img_out_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Tile buffer 地址', file=f)
        for i in tile_buf_addr_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 使用的 Tile buffer 大小', file=f)
        for i in tlle_buf_used_size_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## FIFO master & slave 阈值', file=f)
        for i in fifo_threshold_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 全连接模式数据长度 ', file=f)
        for i in fcn_len_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## Line Buffer 地址 & 长度 （卷积的图片只算宽度，区别于Tile Buffer 大小的配置） ', file=f)
        for i in linebuf_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输入数据 buffer 的类型 （ 0~7 对应 2K~32K）', file=f)
        for i in in_buf_type_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 输出数据 buffer 的类型 （ 0~7 对应 2K~32K） ', file=f)
        for i in out_buf_type_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## TILE AXI的数据个数（输出的数据量，整幅图大小包括padding） ', file=f)
        for i in axi_cnt_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 激活函数&移位控制 ', file=f)
        for i in sfu_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## padding控制 ', file=f)
        for i in pad_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 残差连接控制 ', file=f)
        for i in resnet_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## BL 起始行地址 ', file=f)
        for i in bl_start_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        *****************************************************
        ****                                             ****
        ****             配置MCU传输寄存器                ****
        ****                                             ****
        *****************************************************
        ''', file=f)
        print('## MCU 写到缓存 (Esram:0x68-- or Tile Buffer:0x78--)的地址 ', file=f)
        for i in mcu_write_addr_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU 写到缓存 (Esram:0x68-- or Tile Buffer:0x78--)的数据长度 ', file=f)
        for i in mcu_write_len_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU 读数的基地址 & 读数的次数', file=f)
        for i in mcu_read_ctrl_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU CMD que 读数地址（最多8个通路）', file=f)
        for i in cmd_que_addr_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## MCU CMD que 数据长度（最多8个通路）', file=f)
        for i in cmd_que_len_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        *****************************************************
        ****                                             ****
        ****             配置开始计算寄存器               ****
        ****                                             ****
        *****************************************************
        ''', file=f)
        print('## 开始计算触发 （reset）', file=f)
        for i in reset_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('## 中断响应 ', file=f)
        for i in int_req_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
        f.write('\n')
        print('''
        *****************************************************
        ****                                             ****
        ****             ADC结果暂存寄存器                ****
        ****                                             ****
        *****************************************************
        ''', file=f)
        print('## ADC结果暂存 ', file=f)
        for i in adc_results_reg:
            val = a111_ffi.ffi.new("uint32_t*")
            re = a111_ffi.lib.a111_read_tile_reg(tile_id, i, val)
            addr = base_addr + i
            if re:
                raise ValueError('读取寄存器 %#x 失败'%addr)
            print('addr: %#x, val: %#x '% (addr, val[0]), file=f)
    print('寄存器状态保存成功 ！！！')


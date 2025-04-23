import time
import math
from .a111_helper import *
from .libA111SDK import *

time_record =  time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime(time.time()))

def FC_one_layer(tile_id, xb_id_list, input_data, output_column, calc_mode = 3, adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], 
                 shift_list = [0x3, 0x3, 0x3, 0x3], relu = True, bypass = False, dump_reg = False, 
                 dump_serial_script = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        xb_id_list : 计算一层所需要的所有xb, 单次计算的数目只支持1,2,4
        input_data : numpy数组, 输入为320 (单个XB的行数) 的倍数, 用几个xb输入扩展为几倍, 支持batch
        output_column: 指定输出的列地址，[起始列，列数], 必须在 0 ~ 128 范围内
        calc_mode: 选择计算模式，0：高4bit，1：低4bit，3：8bit。
        adc_range_list: 设置每个XB的adc的range，和xb_id_list 一一对应
        shift_list: 设置每个XB group的移位数值
        relu: 输出是否relu
        bypass: 输出是否直接使用ADC的结果, 不经过移位, relu等数字单元
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
    输出：
        输出结果, numpy数组
    '''
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')
    
    # 判断tile_id是否合理
    assert isinstance(tile_id, int)
    assert tile_id >= 0 and tile_id <=5
    
    # 判断xb_id_list是否合理
    if len(xb_id_list) not in [1,2,4]:
        raise ValueError(f'输入xb_id：{xb_id_list}不满足 1层计算 要求!!!')
    
    # 判断输入数据是否合理
    assert 0 < len(input_data.shape) <= 2
    if len(input_data.shape) == 1:
        # batch_num = 1
        # data_len = input_data.shape[0]
        batch_len = input_data.shape[0]
        input_data = np.expand_dims(input_data, axis=0)
    else:
        batch_num = input_data.shape[0]
        batch_len = input_data.shape[1]
        # data_len = input_data.flatten().shape[0]
    assert (batch_len // 320) == len(xb_id_list)
    # input_data = input_data.flatten()
    # assert data_len <= 0x20000
    
    # esram中给输入数据分配的地址空间最大值 （esram 容量限制）
    input_data_limit = 0x180000
    
    # 有效输出
    assert isinstance(output_column, list)
    assert len(output_column) == 2
    # output_eff_start = output_column[0] 
    output_eff_num = output_column[1] - output_column[0]

    # 指定tile
    tile_id = tile_id

    # 非bypass的情况下，支持的xb_id_list情况：[0], [2], [0, 1], [2, 3], [0, 1, 2, 3], [4], [6], [4,5], [6,7], [4, 5, 6, 7]
    if not bypass:
        assert xb_id_list in [[0], [2], [0, 1], [2, 3], [0, 1, 2, 3], [4], [6], [4,5], [6,7], [4, 5, 6, 7]]
        single_layer_list = [[0], [2], [4], [6]]
    else:
        single_layer_list = [[0], [1], [2], [3], [4], [5], [6], [7]]
        assert xb_id_list in single_layer_list
        
    double_layer_list = [[0,1],[2,3],[4,5],[6,7]]
    quadra_layer_list = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # 指定tile的模式
    tile_mode = 3 # tile分成四组，每组两个xb
    if xb_id_list in quadra_layer_list:
        tile_mode = 1
    
    # 指定bypass模式
    bypass_mode = 0
    adc_bypass_id = None
    bypass_sel = 0
    if bypass:
        bypass_mode = 1
        adc_bypass_id = xb_id_list
        bypass_sel = xb_id_list[0] % 2
    
    # 选择xb
    xb_arr_sel = 0
    if xb_id_list[0] >=4:
        xb_arr_sel = 3
    
    # 指定xbg mode
    xbg_mode_list = [0, 0, 0, 0] # xb_id_list in [[0], [2], [4], [6]]:
    xbg_calc_mode_list = [0,0,0,0]
    xbg_toggle_bit0_list = [0,0,0,0]
    xbg_tile_buf_en0_list = [0,0,0,0]
    xbg_tile_cal_en0_list = [0,0,0,0]
    xbg_fcn_en0_list = [0,0,0,0]
    xbg_relu_en_list = [0,0,0,0]
    xbg_in_pix_type_list = [3,3,3,3]
    xbg_out_pix_type_list = [3,3,3,3]

    index = xb_id_list[0] // 2
    # xbg_calc_mode_list[index] = 3
    assert calc_mode in [0, 1, 3]
    xbg_calc_mode_list[index] = calc_mode
    xbg_toggle_bit0_list[index] = 1
    xbg_tile_buf_en0_list[index] = 1
    xbg_tile_cal_en0_list[index] = 1
    xbg_fcn_en0_list[index] = 1
    if not relu:
        xbg_relu_en_list[index] = 1
    
    if xb_id_list in double_layer_list :
        index = double_layer_list.index(xb_id_list)
        xbg_mode_list[index] = 1

    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        xbg_mode_list[index * 2] = 2
    xbg_para_type_list = [0, 0, 0, 0]
    xbg_op_mode_list = [0, 0, 0, 0]
    
    # 设置xbg in/out pix type 
    if output_column[1] == 32:
        xbg_out_pix_type_list[index] = 2
    elif output_column[1] == 64:
        xbg_out_pix_type_list[index] = 3
    elif output_column[1] == 128:
        xbg_out_pix_type_list[index] = 4

    # 指定xb计算的起始列地址， 起始列为0, 32, 64, 96
    xb_start_column_list = [0] * len(xb_id_list)

    # 指定xb计算的列数 0:32列；1:64列；>=2:128列。
    xb_column_num_list = [3] * len(xb_id_list)

    # 指定输入地址列表
    input_addr_list = [0x0, 0x0, 0x0, 0x0]
     
    # 指定输入数据长度
    input_len_list = [0x0,0x0,0x0,0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        input_len_list[index] = batch_len
    if xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        input_len_list[index] = batch_len
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        input_len_list[index * 2] = batch_len 
    # 指定输入图片大小
    in_img_size_list = [[1,1],[1,1],[1,1],[1,1]]

    # 初始化输出 以及 输出地址
    out_addr_esram = input_data_limit
    
    # 指定输出地址
    output_addr_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        output_addr_list[index] = 0x68000000 + out_addr_esram
    elif xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        output_addr_list[index] = 0x68000000 + out_addr_esram
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        output_addr_list[index * 2] = 0x68000000 + out_addr_esram
     
    # 指定输出图片大小
    out_img_size_list = [[1,1],[1,1],[1,1],[1,1]]
    # 指定输出 axi_cnt
    # 固定输出 128 列
    out_len = 128
    num_type = '8bit'
    # 结果是否relu
    if (not relu) or bypass:
        out_len = 256
        num_type = '9bit'

    xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        xbg_axi_cnt_list[index] = out_len
    elif xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        xbg_axi_cnt_list[index] = out_len
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        xbg_axi_cnt_list[index * 2] = out_len 

    # 指定linebuffer的偏移地址 以及 数据量
    linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
    linebuf_width_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        linebuf_width_list[index] = batch_len
    elif xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        linebuf_width_list[index] = batch_len
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        linebuf_width_list[index * 2] = batch_len
    
    # 指定sfu参数
    relu_th_list = [0x0, 0x0, 0x0, 0x0]
    act_mode_list = [0x0, 0x0, 0x0, 0x0]
    shift_list = shift_list

    # 指定输入数据esram地址
    in_addr_esram = 0x0

    # 加载dummy 数据
    dummy_input = np.zeros((batch_len,),dtype=np.uint8)
    # 指定dummy 数据esram地址
    dummy_input_addr_esram = 0x200000
    # 传输dummy数据到esram
    set_input_to_esram(dummy_input, dummy_input_addr_esram)

    # 初始化输出array
    output = np.zeros((batch_num,output_eff_num))
    
    # res_9bit_en
    res_9bit_en = 0
    res_in_sel = 0
    res_out_sel = 0
    if (not relu) or bypass:
        res_in_sel = xb_id_list[0] // 2
        res_out_sel = xb_id_list[0] // 2
        res_9bit_en = 1

    # 配置运行寄存器
    TileOp( tile_id,  xb_id_list, 
            tile_mode = tile_mode, bypass_mode= bypass_mode, bypass_sel = bypass_sel, xb_arr_sel = xb_arr_sel, # tile mode
            xbg_mode_list = xbg_mode_list, xbg_para_type_list = xbg_para_type_list, xbg_op_mode_list = xbg_op_mode_list, # xbg mode 
            xbg_calc_mode_list = xbg_calc_mode_list, xbg_in_pix_type_list = xbg_in_pix_type_list, xbg_out_pix_type_list= xbg_out_pix_type_list, #xbg mode
            xbg_toggle_bit0_list = xbg_toggle_bit0_list, xbg_tile_buf_en0_list = xbg_tile_buf_en0_list, # xbg mode 
            xbg_tile_cal_en0_list = xbg_tile_cal_en0_list, xbg_fcn_en0_list = xbg_fcn_en0_list, xbg_relu_en_list = xbg_relu_en_list, # xbg mode 
            xb_start_column_list = xb_start_column_list, xb_column_num_list = xb_column_num_list, # xb column
            input_addr_list = input_addr_list, input_len_list = input_len_list, in_img_size_list = in_img_size_list, # input 
            output_addr_list = output_addr_list, out_img_size_list = out_img_size_list, xbg_axi_cnt_list = xbg_axi_cnt_list, # output 
            linebuf_addr_offset_list = linebuf_addr_offset_list, linebuf_width_list = linebuf_width_list, # linebuffer
            relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list, # sfu
            adc_range_list=adc_range_list,  adc_bypass_id=adc_bypass_id, # xb adc range
            res_in_sel = res_in_sel, res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, # res_9bit_en
            )

    # tile 输入的基地址
    rd_addr_base = 0x68

    # esram 写入tile buffer 地址
    tile_buffer_input_addr = 0x78000000

    # time1 = time.time()
    
    # 根据 input_data_limit 拆分 batch number
    
    # 1. 根据 input_data_limit 与 batch_len 计算最大的单次batch数 batch_num_split (向下取整)
    batch_num_split = input_data_limit // batch_len 
    
    # 用于计数 已经计算的输入数量
    # count = 0
    
    try:
        
        # 2. 每次传入 batch_num_split 的输入数据，直到当前所有的输入数据计算完成 （batch_num）
        for batch in range(0, batch_num, batch_num_split):
            
            # 指定输入数据esram地址
            in_addr_esram = 0x0
            
            # 
            batch_end = min(batch_num_split+batch, batch_num)
            # 拆分输入
            input_data_ = input_data[batch:batch_end,:]
            
            # 当前输入的batch 数量
            batch_num_mini = input_data_.shape[0]
            # 
            input_data_ = input_data_.flatten()
            
            # 3. 传输输入数据到esram
            set_input_to_esram(input_data_, in_addr_esram)
        

            for row in range(batch_num_mini):
                # print(f'==================================  第 {row} 次计算开始 ==================================>>>>>')
                # #校验数据是否是输入当前batch
                # # print('读数据地址 ： %#x' % in_addr_esram)
                # read_data_ = get_value_from_esram(in_addr_esram, batch_len)
                # assert (read_data_ == input_data[(row * batch_len) : (row + 1) * batch_len]).all()

                # # print('输入数据 esram_addr: %#x' % (in_addr_esram))
                # dummy 计算
                set_mcu_send_data_to_tile(tile_id, rd_addr_offset = dummy_input_addr_esram, rd_length=batch_len,
                                        rd_addr_base=rd_addr_base, rd_cycle=1, write_addr=tile_buffer_input_addr,
                                        write_length=batch_len)
                
                set_tile_run(tile_id, tile_mode)
                
                if IsCalcDone(tile_id, output_xb_id = xb_id_list[0], log_file=f'fc/reg_log/test_fc_one_layer_{time_record}.txt'):
                    # print('dummy计算结束！！！')
                    pass

                # 校准
                set_tile_cali(tile_id, xb_id_list)

                # 输入数据计算
                set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len,
                                        rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = tile_buffer_input_addr,
                                        write_length = batch_len)
                set_tile_run(tile_id, tile_mode)

                if IsCalcDone(tile_id, output_xb_id = xb_id_list[0], log_file=f'fc/reg_log/test_fc_one_layer_{time_record}.txt'):
                    # print('输入数据计算结束！！！')
                    pass

                # 获取输出
                # print(f'output_row_index : {output_row_index}')
                # output[row + batch,:] = get_tileop_results(out_addr_esram, out_len, num_type = num_type, out_len_effective=output_column)
                output[row + batch,:] = get_tileop_results_FPGA(out_addr_esram, out_len, num_type = num_type, out_len_effective=output_column)
                
                # esram地址递增
                in_addr_esram = in_addr_esram + batch_len
    
    except:
        
        # dump reg
        if dump_reg:
            file_path = dump_reg.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_TileReg(tile_id = tile_id, file=dump_reg)

        # dump serial reg
        if dump_serial_script:
            file_path = dump_serial_script.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_serial(tile_id = tile_id, file=dump_serial_script)

        # time2 = time.time()
        
        # print(f"计算 {batch_num} 输入时间为： {time2-time1} s")

        raise ValueError(f'Error !!!')
        
    # reset hardware 
    a111_hw_reset()
    
    return output

def FC_one_layer_bias(tile_id, xb_id_list, input_data, output_column, adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], 
                 shift_list = [0x3, 0x3, 0x3, 0x3], relu = True, 
                 bias = False, bias_num = [0], bias_input_value_list = [[0]],
                 dump_reg = False,  dump_serial_script = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        xb_id_list : 计算一层所需要的所有xb, 单次计算的数目只支持1,2,4
        input_data : numpy数组, 输入为320 (单个XB的行数) 的倍数, 用几个xb输入扩展为几倍, 支持batch
        output_column: 指定输出的列地址，[起始列，列数], 必须在 0 ~ 128 范围内
        adc_range_list: 设置每个XB的adc的range，和xb_id_list 一一对应
        shift_list: 设置每个XB group的移位数值
        relu: 输出是否relu
        bias: 是否有bias， 
        bias_num: 阵列前多少行作为bias，与xb_id_list 一一对应，0: 前0行作为bias； 1： 前4行作为bias，2：前8行，3：前32行
        bias_input_value: bias 输入的大小，
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
    输出：
        输出结果, numpy数组
    '''
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')
    
    # 判断tile_id是否合理
    assert isinstance(tile_id, int)
    assert tile_id >= 0 and tile_id <=5
    
    # 判断xb_id_list是否合理
    if len(xb_id_list) not in [1,2,4]:
        raise ValueError(f'输入xb_id：{xb_id_list}不满足 1层计算 要求!!!')
    
    # 判断输入数据是否合理
    assert 0 < len(input_data.shape) <= 2
    if len(input_data.shape) == 1:
        # batch_num = 1
        # data_len = input_data.shape[0]
        batch_len = input_data.shape[0]
        input_data = np.expand_dims(input_data, axis=0)
    else:
        batch_num = input_data.shape[0]
        batch_len = input_data.shape[1]
        # data_len = input_data.flatten().shape[0]
    
    # 校验输入长度
    if bias:
        bias_row_num = 0
        if bias_num[0] < 3 and bias_num[0] > 0:
            bias_row_num += 2 ** (bias_num[0] + 1)
        elif bias_num[0] == 3:
            bias_row_num += 32
        assert (batch_len + len(xb_id_list) * bias_row_num) // 320  == len(xb_id_list)
    else:
        assert (batch_len // 320) == len(xb_id_list)
        
    # input_data = input_data.flatten()
    # assert data_len <= 0x100000
    
    # esram中给输入数据分配的地址空间最大值 （esram 容量限制）
    input_data_limit = 0x100000
    
    # 有效输出
    assert isinstance(output_column, list)
    assert len(output_column) == 2
    # output_eff_start = output_column[0] 
    output_eff_num = output_column[1] - output_column[0]

    # 指定tile
    tile_id = tile_id

    # 支持的xb_id_list情况：[0], [2], [0, 1], [2, 3], [0, 1, 2, 3], [4], [6], [4,5], [6,7], [4, 5, 6, 7]
    assert xb_id_list in [[0], [2], [0, 1], [2, 3], [0, 1, 2, 3], [4], [6], [4,5], [6,7], [4, 5, 6, 7]]
    single_layer_list = [[0], [2], [4], [6]]
    double_layer_list = [[0,1],[2,3],[4,5],[6,7]]
    quadra_layer_list = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # 指定tile的模式
    tile_mode = 3 # tile分成四组，每组两个xb
    if xb_id_list in quadra_layer_list:
        tile_mode = 1
    xb_arr_sel = 0
    if xb_id_list[0] >=4:
        xb_arr_sel = 3
    
    # 指定xbg mode
    xbg_mode_list = [0, 0, 0, 0] # xb_id_list in [[0], [2], [4], [6]]:
    xbg_calc_mode_list = [0,0,0,0]
    xbg_toggle_bit0_list = [0,0,0,0]
    xbg_tile_buf_en0_list = [0,0,0,0]
    xbg_tile_cal_en0_list = [0,0,0,0]
    xbg_fcn_en0_list = [0,0,0,0]
    xbg_relu_en_list = [0,0,0,0]
    xbg_in_pix_type_list = [3,3,3,3]
    xbg_out_pix_type_list = [3,3,3,3]
    xbg_bias_en_list = [0,0,0,0]

    xb_start_row_list = [0] * len(xb_id_list)

    index = xb_id_list[0] // 2
    xbg_calc_mode_list[index] = 3
    xbg_toggle_bit0_list[index] = 1
    xbg_tile_buf_en0_list[index] = 1
    xbg_tile_cal_en0_list[index] = 1
    xbg_fcn_en0_list[index] = 1

    xb_bias_input_value_list = [[0]]

    # bias
    if bias:
        xbg_bias_en_list[index] = 1
        assert len(bias_num) == len(xb_id_list)
        xb_start_row_list = bias_num
        xb_bias_input_value_list = bias_input_value_list

    if not relu:
        xbg_relu_en_list[index] = 1
    
    if xb_id_list in double_layer_list :
        index = double_layer_list.index(xb_id_list)
        xbg_mode_list[index] = 1

    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        xbg_mode_list[index * 2] = 2
    xbg_para_type_list = [0, 0, 0, 0]
    xbg_op_mode_list = [0, 0, 0, 0]
    
    # 设置xbg in/out pix type 
    if output_column[1] == 32:
        xbg_out_pix_type_list[index] = 2
    elif output_column[1] == 64:
        xbg_out_pix_type_list[index] = 3
    elif output_column[1] == 128:
        xbg_out_pix_type_list[index] = 4

    # 指定xb计算的起始列地址， 起始列为0, 32, 64, 96
    xb_start_column_list = [0] * len(xb_id_list)

    # 指定xb计算的列数 0:32列；1:64列；>=2:128列。
    xb_column_num_list = [3] * len(xb_id_list)

    # 指定输入地址列表
    input_addr_list = [0x0, 0x0, 0x0, 0x0]
     
    # 指定输入数据长度
    input_len_list = [0x0,0x0,0x0,0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        input_len_list[index] = batch_len
    if xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        input_len_list[index] = batch_len
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        input_len_list[index * 2] = batch_len 
    # 指定输入图片大小
    in_img_size_list = [[1,1],[1,1],[1,1],[1,1]]

    
    # 初始化输出 以及 输出地址
    out_addr_esram = input_data_limit
    
    # 指定输出地址
    output_addr_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        output_addr_list[index] = 0x68000000 + out_addr_esram
    elif xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        output_addr_list[index] = 0x68000000 + out_addr_esram
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        output_addr_list[index * 2] = 0x68000000 + out_addr_esram
     
    # 指定输出图片大小
    out_img_size_list = [[1,1],[1,1],[1,1],[1,1]]
    # 指定输出 axi_cnt
    # 固定输出 128 列
    out_len = 128
    num_type = '8bit'
    # 结果是否relu
    if not relu:
        out_len = 256
        num_type = '9bit'

    xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        xbg_axi_cnt_list[index] = out_len
    elif xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        xbg_axi_cnt_list[index] = out_len
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        xbg_axi_cnt_list[index * 2] = out_len 

    # 指定linebuffer的偏移地址 以及 数据量
    linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
    linebuf_width_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list in single_layer_list:
        index = xb_id_list[0] // 2 
        linebuf_width_list[index] = batch_len
    elif xb_id_list in double_layer_list:
        index = double_layer_list.index(xb_id_list)
        linebuf_width_list[index] = batch_len
    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        linebuf_width_list[index * 2] = batch_len
    
    # 指定sfu参数
    relu_th_list = [0x0, 0x0, 0x0, 0x0]
    act_mode_list = [0x0, 0x0, 0x0, 0x0]
    shift_list = shift_list

    # 指定输入数据esram地址
    in_addr_esram = 0x0

    # 加载dummy 数据
    dummy_input = np.zeros((batch_len,),dtype=np.uint8)
    # 指定dummy 数据esram地址
    dummy_input_addr_esram = 0x200000
    # 传输dummy数据到esram
    set_input_to_esram(dummy_input, dummy_input_addr_esram)

    # 初始化输出array
    output = np.zeros((batch_num,output_eff_num))
    
    # res_9bit_en
    res_9bit_en = 0
    res_in_sel = 0
    res_out_sel = 0
    if not relu:
        res_in_sel = xb_id_list[0] // 2
        res_out_sel = xb_id_list[0] // 2
        res_9bit_en = 1

    # 配置运行寄存器
    TileOp( tile_id,  xb_id_list, 
            tile_mode = tile_mode, xb_arr_sel = xb_arr_sel, # tile mode
            xbg_mode_list = xbg_mode_list, xbg_para_type_list = xbg_para_type_list, xbg_op_mode_list = xbg_op_mode_list, # xbg mode 
            xbg_calc_mode_list = xbg_calc_mode_list, xbg_in_pix_type_list = xbg_in_pix_type_list, xbg_out_pix_type_list= xbg_out_pix_type_list, #xbg mode
            xbg_toggle_bit0_list = xbg_toggle_bit0_list, xbg_tile_buf_en0_list = xbg_tile_buf_en0_list, # xbg mode 
            xbg_tile_cal_en0_list = xbg_tile_cal_en0_list, xbg_fcn_en0_list = xbg_fcn_en0_list,  xbg_bias_en_list = xbg_bias_en_list, xbg_relu_en_list = xbg_relu_en_list, # xbg mode 
            xb_start_column_list = xb_start_column_list, xb_column_num_list = xb_column_num_list, # xb column
            xb_start_row_list = xb_start_row_list, # xb row
            input_addr_list = input_addr_list, input_len_list = input_len_list, in_img_size_list = in_img_size_list, # input 
            output_addr_list = output_addr_list, out_img_size_list = out_img_size_list, xbg_axi_cnt_list = xbg_axi_cnt_list, # output 
            linebuf_addr_offset_list = linebuf_addr_offset_list, linebuf_width_list = linebuf_width_list, # linebuffer
            relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list, # sfu
            adc_range_list=adc_range_list, # xb adc range
            res_in_sel = res_in_sel, res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, # res_9bit_en
            xb_bias_input_value_list = xb_bias_input_value_list, # bias input
            )

    # tile 输入的基地址
    rd_addr_base = 0x68

    # esram 写入tile buffer 地址
    tile_buffer_input_addr = 0x78000000

    # 根据 input_data_limit 拆分 batch number
    
    # 1. 根据 input_data_limit 与 batch_len 计算最大的单次batch数 batch_num_split (向下取整)
    batch_num_split = input_data_limit // batch_len 
    
    # 用于计数 已经计算的输入数量
    # count = 0
    
    try:
        
        # 2. 每次传入 batch_num_split 的输入数据，直到当前所有的输入数据计算完成 （batch_num）
        for batch in range(0, batch_num, batch_num_split):
            
            # 指定输入数据esram地址
            in_addr_esram = 0x0
            
            # 
            batch_end = min(batch_num_split+batch, batch_num)
            # 拆分输入
            input_data_ = input_data[batch:batch_end,:]
            
            # 当前输入的batch 数量
            batch_num_mini = input_data_.shape[0]
            # 
            input_data_ = input_data_.flatten()
            
            # 3. 传输输入数据到esram
            set_input_to_esram(input_data_, in_addr_esram)
        

            for row in range(batch_num_mini):
                # print(f'==================================  第 {row} 次计算开始 ==================================>>>>>')
                # #校验数据是否是输入当前batch
                # # print('读数据地址 ： %#x' % in_addr_esram)
                # read_data_ = get_value_from_esram(in_addr_esram, batch_len)
                # assert (read_data_ == input_data[(row * batch_len) : (row + 1) * batch_len]).all()

                # # print('输入数据 esram_addr: %#x' % (in_addr_esram))
                
                # dummy 计算
                set_mcu_send_data_to_tile(tile_id, rd_addr_offset = dummy_input_addr_esram, rd_length=batch_len,
                                        rd_addr_base=rd_addr_base, rd_cycle=1, write_addr=tile_buffer_input_addr,
                                        write_length=batch_len)
                
                set_tile_run(tile_id, tile_mode)
                
                if IsCalcDone(tile_id, output_xb_id = xb_id_list[0], log_file=f'fc/reg_log/test_fc_one_layer_{time_record}.txt'):
                    # print('dummy计算结束！！！')
                    pass

                # 校准
                set_tile_cali(tile_id, xb_id_list)

                # 输入数据计算
                set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len,
                                        rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = tile_buffer_input_addr,
                                        write_length = batch_len)
                set_tile_run(tile_id, tile_mode)

                if IsCalcDone(tile_id, output_xb_id = xb_id_list[0], log_file=f'fc/reg_log/test_fc_one_layer_{time_record}.txt'):
                    # print('输入数据计算结束！！！')
                    pass

                # 获取输出
                # print(f'output_row_index : {output_row_index}')
                # output[row + batch,:] = get_tileop_results(out_addr_esram, out_len, num_type = num_type, out_len_effective=output_column)
                output[row + batch,:] = get_tileop_results_FPGA(out_addr_esram, out_len, num_type = num_type, out_len_effective=output_column)
                
                # esram地址递增
                in_addr_esram = in_addr_esram + batch_len
    
    except:
        
        # dump reg
        if dump_reg:
            file_path = dump_reg.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_TileReg(tile_id = tile_id, file=dump_reg)

        # dump serial reg
        if dump_serial_script:
            file_path = dump_serial_script.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_serial(tile_id = tile_id, file=dump_serial_script)

        # time2 = time.time()
        
        # print(f"计算 {batch_num} 输入时间为： {time2-time1} s")

        raise ValueError(f'Error !!!')
        
    # reset hardware 
    a111_hw_reset()
    
    return output

def FC_two_layer(tile_id, xb_id_list, input_data, output_column1, output_column2, adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7],
                 shift_list = [0x3, 0x3, 0x3, 0x3], second_relu = False, dump_reg = False, dump_serial_script = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        xb_id_list : 计算一层所需要的所有xb, 单次计算的数目只支持1, 2, 4
        input_data : numpy数组，输入为320（单个XB的行数）的倍数，用几个xb输入扩展为几倍, 支持batch
        output_column1 : 第一层输出的列地址 [列起始，列数] 必须 32 对齐， 需要在 0 ~ 128 范围内
        output_column2 : 第二层输出的列地址 [列起始，列数]
        adc_range_list: 设置每个XB的adc的range，和xb_id_list 一一对应
        shift_list: 设置每个XB group的移位数值
        second_relu: 第二层输出是否relu
        bias: 是否有bias， 
        bias_num: 阵列前多少行作为bias，与xb_id_list 一一对应，0: 前0行作为bias； 1： 前4行作为bias，2：前8行，3：前32行
        bias_input_value: bias 输入的大小，
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
    输出：
        输出结果, numpy数组
    '''
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')

    # 判断tile_id是否合理, 
    assert isinstance(tile_id, int)
    assert tile_id >= 0 and tile_id <=5
    
    # 判断xb_id_list是否合理, 支持的xb_id_list情况：[0, 2], [0, 1, 2], [4, 6], [4, 5, 6]
    if len(xb_id_list) not in [2,3]:
        raise ValueError(f'输入xb_id：{xb_id_list}不满足 2层计算 要求!!!')
    
    # 判断输入数据是否合理
    assert 0 < len(input_data.shape) <= 2
    if len(input_data.shape) == 1:
        batch_num = 1
        data_len = input_data.shape[0]
        batch_len = data_len
    else:
        batch_num = input_data.shape[0]
        batch_len = input_data.shape[1]
        data_len = input_data.flatten().shape[0]

    assert (batch_len // 320) in [1, 2]
    input_data = input_data.flatten()
    assert data_len <= 0x60000 # dummy_input esram addr
    
    # 有效输出
    assert isinstance(output_column1, list)
    assert isinstance(output_column2, list)
    assert len(output_column1) == 2
    assert output_column1[0] in [0, 32, 64, 96]
    assert output_column1[1] in [32, 64, 128]
    assert (output_column1[0] + output_column1[1]) <= 128
    assert len(output_column2) == 2

    # 指定tile
    tile_id = tile_id

    # 支持的xb_id_list情况：[0, 2], [0, 1, 2], [4, 6], [4, 5, 6]
    assert xb_id_list in [[0, 2], [0, 1, 2], [4, 6], [4, 5, 6]]
    double_layer_list = [[0,2], [4,6]]
    triple_layer_list = [[0, 1, 2], [4, 5, 6]]

    # 指定tile的模式
    tile_mode = 3 # tile分成四组，每组两个xb
    # 选择输出的 xb 开关
    xb_arr_sel = 0
    if xb_id_list[0] >=4:
        xb_arr_sel = 3
    # 指定需要计算的xb
    xb_id_list = xb_id_list
    # 指定xbg mode
    xbg_mode_list = [0, 0, 0, 0] # xb_id_list in [[0], [2], [4], [6]]:
    if xb_id_list in triple_layer_list:
        index = triple_layer_list.index(xb_id_list)
        xbg_mode_list[index * 2] = 1
    xbg_para_type_list = [0, 0, 0, 0]
    xbg_op_mode_list = [0, 0, 0, 0]
    
    xbg_calc_mode_list = [0,0,0,0]
    xbg_toggle_bit0_list = [0,0,0,0]
    xbg_tile_buf_en0_list = [0,0,0,0]
    xbg_tile_cal_en0_list = [0,0,0,0]
    xbg_fcn_en0_list = [0,0,0,0]
    xbg_relu_en_list = [0,0,0,0]
    xbg_in_pix_type_list = [3,0,3,3]
    xbg_out_pix_type_list = [3,2,3,3]

    if xb_id_list in double_layer_list:
        index_list = [xb_id_list[0] // 2,  xb_id_list[1] // 2]
    elif xb_id_list in triple_layer_list:
        index_list = [xb_id_list[0] // 2,  xb_id_list[2] // 2]
    
    for index in index_list:
        xbg_calc_mode_list[index] = 3
        xbg_toggle_bit0_list[index] = 1
        xbg_tile_buf_en0_list[index] = 1
        xbg_tile_cal_en0_list[index] = 1
        xbg_fcn_en0_list[index] = 1

    # 设置xbg in/out pix type 
    if output_column1[1] == 32:
        xbg_out_pix_type_list[index_list[0]] = 2
        xbg_in_pix_type_list[index_list[1]] = 3
    elif output_column1[1] == 64:
        xbg_out_pix_type_list[index_list[0]] = 3
        xbg_in_pix_type_list[index_list[1]] = 4
    elif output_column1[1] == 128:
        xbg_out_pix_type_list[index_list[0]] = 4
        xbg_in_pix_type_list[index_list[1]] = 5
    


    if not second_relu:
        xbg_relu_en_list[index_list[-1]] = 1
    
    # 指定输入地址列表 & 指定输入数据长度
    if xb_id_list[0] == 0:
        input_addr_list = [0x0, 0x4000, 0x0, 0x0]
        input_len_list = [batch_len,output_column1[1],0x0,0x0]
    else:
        input_addr_list = [0x0, 0x0, 0x0, 0x4000]
        input_len_list = [0x0,0x0,batch_len,output_column1[1]]

    # 指定输入图片大小
    in_img_size_list = [[1,1],[1,1],[1,1],[1,1]]

    # 指定输出地址 输出大小, 第二层固定输出128
    # 固定输出 32 列
    assert output_column2[0] <= 32
    assert (output_column2[0] + output_column2[1]) <= 32
    out_len = 32
    num_type = '8bit'
    # 结果是否relu
    if not second_relu:
        out_len = 64
        num_type = '9bit'

    # 指定xb计算的起始列地址， 起始列为0, 32, 64, 96
    # 指定xb计算的列数 0:32列；1:64列；>=2:128列。
    if xb_id_list in double_layer_list:
        xb_start_column_list = [output_column1[0] // 32,  0]
        if output_column1[1] <= 32:
            xb_column_num_list = [0, 0]
        elif output_column1[1] <= 64:
            xb_column_num_list = [1, 0]
        else:
            xb_column_num_list = [3, 0]

    elif xb_id_list in triple_layer_list:
        xb_start_column_list = [output_column1[0] // 32 , output_column1[0] // 32, 0]
        if output_column1[1] <= 32:
            xb_column_num_list = [0, 0, 0]
        elif output_column1[1] <= 64:
            xb_column_num_list = [1, 1, 0]
        else:
            xb_column_num_list = [3, 3, 0]

    output_addr_list = [0x0, 0x0, 0x0, 0x0]
    xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list[0] == 0:
        output_addr_list = [0x78004000, 0x68080000, 0x0, 0x0]
        xbg_axi_cnt_list = [output_column1[1],out_len,0x0,0x0]
    else:
        output_addr_list = [0x0, 0x0, 0x78004000, 0x68080000]
        xbg_axi_cnt_list = [0x0, 0x0, output_column1[1],out_len]
    
    # 指定输出图片大小
    out_img_size_list = [[1,1],[1,1],[1,1],[1,1]]

    # 指定linebuffer的偏移地址 以及 数据量
    linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
    linebuf_width_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list[0] == 0:
        linebuf_addr_offset_list = [0x0, 0x4000, 0x0, 0x0]
        linebuf_width_list = [batch_len,out_len,0x0,0x0]
    else:
        linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x4000]
        linebuf_width_list = [0x0, 0x0, batch_len,out_len]

    # 指定sfu参数
    relu_th_list = [0x0, 0x0, 0x0, 0x0]
    act_mode_list = [0x0, 0x0, 0x0, 0x0]
    shift_list = shift_list

    # 指定输入数据esram地址
    in_addr_esram = 0x0

    # 加载dummy 数据
    dummy_input = np.zeros((batch_len,),dtype=np.uint8)
    # 指定dummy 数据esram地址
    dummy_input_addr_esram = 0x60000
    # 传输dummy数据到esram
    set_input_to_esram(dummy_input, dummy_input_addr_esram)

    # 初始化输出 以及 输出地址
    out_addr_esram = 0x80000
    output = np.zeros((batch_num,output_column2[1]))
    
    # res_9bit_en
    res_9bit_en = 0
    res_in_sel = 0
    res_out_sel = 0
    if not second_relu:
        res_in_sel = xb_id_list[-1] // 2
        res_out_sel = xb_id_list[-1] // 2
        res_9bit_en = 1

   # 配置运行寄存器
    TileOp( tile_id,  xb_id_list, 
            tile_mode = tile_mode, xb_arr_sel = xb_arr_sel, # tile mode
            xbg_mode_list = xbg_mode_list, xbg_para_type_list = xbg_para_type_list, xbg_op_mode_list = xbg_op_mode_list, # xbg mode 
            xbg_calc_mode_list = xbg_calc_mode_list, xbg_in_pix_type_list = xbg_in_pix_type_list, xbg_out_pix_type_list= xbg_out_pix_type_list, # xbg mode 
            xbg_toggle_bit0_list = xbg_toggle_bit0_list, xbg_tile_buf_en0_list = xbg_tile_buf_en0_list, # xbg mode 
            xbg_tile_cal_en0_list = xbg_tile_cal_en0_list, xbg_fcn_en0_list = xbg_fcn_en0_list, xbg_relu_en_list= xbg_relu_en_list, # xbg mode 
            xb_start_column_list = xb_start_column_list, xb_column_num_list = xb_column_num_list, # xb column
            input_addr_list = input_addr_list, input_len_list = input_len_list, in_img_size_list = in_img_size_list, # input 
            output_addr_list = output_addr_list, out_img_size_list = out_img_size_list, xbg_axi_cnt_list = xbg_axi_cnt_list, # output 
            linebuf_addr_offset_list = linebuf_addr_offset_list, linebuf_width_list = linebuf_width_list, # linebuffer
            relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list, # sfu
            adc_range_list=adc_range_list, # xb adc range
            res_in_sel = res_in_sel, res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, # res_en
            )

    # tile 输入的基地址
    rd_addr_base = 0x68

    # esram 写入tile buffer 地址
    tile_buffer_input_addr = 0x78000000

    time1 = time.time()
    
    # 传输输入数据到esram
    set_input_to_esram(input_data, in_addr_esram)
    
    for row in range(batch_num):
        print(f'==================================  第 {row} 次计算开始 ==================================>>>>>')
        #校验数据是否是输入当前batch
        # print('读数据地址 ： %#x' % in_addr_esram)
        read_data_ = get_value_from_esram(in_addr_esram, batch_len)
        assert (read_data_ == input_data[(row * batch_len) : (row + 1) * batch_len]).all()
                  
        print('输入数据 esram_addr: %#x' % (in_addr_esram))
        # dummy 计算
        set_mcu_send_data_to_tile(tile_id, rd_addr_offset = dummy_input_addr_esram, rd_length=batch_len,
                                rd_addr_base=rd_addr_base, rd_cycle=1, write_addr=tile_buffer_input_addr,
                                write_length=batch_len)
        
        set_tile_run(tile_id, tile_mode)
        
        if IsCalcDone(tile_id, output_xb_id = xb_id_list[-1], log_file=f'fc/reg_log/test_fc_two_layer_{time_record}.txt'):
            print('dummy计算结束！！！')
            pass
        
        # 校准
        set_tile_cali(tile_id, xb_id_list)
        
        # 输入数据计算
        set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len,
                                rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = tile_buffer_input_addr,
                                write_length = batch_len)
        
        set_tile_run(tile_id, tile_mode)
        
        if IsCalcDone(tile_id, output_xb_id = xb_id_list[-1], log_file=f'fc/reg_log/test_fc_two_layer_{time_record}.txt'):
            print('输入数据计算结束！！！')
            pass
        
        # 获取输出
        output_row_index = row 
        print(f'output_row_index : {output_row_index}')
        output[output_row_index,:] = get_tileop_results(out_addr_esram, out_len, num_type=num_type, out_len_effective=output_column2)
        
        # esram地址递增
        in_addr_esram = in_addr_esram + batch_len
    
    # dump tile reg
    if dump_reg:
        file_path = dump_reg.split('/')
        file_path = '/'.join(file_path[0:-1])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dump_TileReg(tile_id = tile_id, file=dump_reg)
    
    # dump serial reg
    if dump_serial_script:
        file_path = dump_serial_script.split('/')
        file_path = '/'.join(file_path[0:-1])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dump_serial(tile_id = tile_id, file=dump_serial_script)

    time2 = time.time()
    
    print(f"计算 {batch_num} 输入时间为： {time2-time1} s")

    # reset hardware 
    a111_hw_reset()

    return output

def FC_two_layer_bias(tile_id, xb_id_list, input_data, output_column1, output_column2, adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7],
                 shift_list = [0x3, 0x3, 0x3, 0x3], second_relu = False, 
                 bias = [False, False], bias_num = [[0,], [0,]], bias_input_value_list = [[[0]],[[0]]],
                 dump_reg = False, dump_serial_script = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        xb_id_list : 计算一层所需要的所有xb, 单次计算的数目只支持1, 2, 4
        input_data : numpy数组，输入为320（单个XB的行数）的倍数，用几个xb输入扩展为几倍, 支持batch
        output_column1 : 第一层输出的列地址 [列起始，列数] 必须 32 对齐， 需要在 0 ~ 128 范围内
        output_column2 : 第二层输出的列地址 [列起始，列数]
        adc_range_list: 设置每个XB的adc的range，和xb_id_list 一一对应
        shift_list: 设置每个XB group的移位数值
        second_relu: 第二层输出是否relu
        bias: 是否有bias.
        bias_num: 阵列前多少行作为bias, 双层列表， 外层与layer一一对应, 内层与xb_id_list一一对应, 0: 前0行作为bias; 1: 前4行作为bias, 2: 前8行, 3: 前32行
        bias_input_value: bias 输入的大小，三层列表, 外层与layer一一对应, 中间层与xb_id_list一一对应, 最内层与xb的行一一对应 
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
    输出：
        输出结果, numpy数组
    '''
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = lib.a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')

    # 判断tile_id是否合理, 
    assert isinstance(tile_id, int)
    assert tile_id >= 0 and tile_id <=5
    
    # 判断xb_id_list是否合理, 支持的xb_id_list情况：[0, 2], [0, 1, 2], [4, 6], [4, 5, 6]
    if len(xb_id_list) not in [2,3]:
        raise ValueError(f'输入xb_id：{xb_id_list}不满足 2层计算 要求!!!')
    
    # 判断输入数据是否合理
    assert 0 < len(input_data.shape) <= 2
    if len(input_data.shape) == 1:
        batch_num = 1
        data_len = input_data.shape[0]
        batch_len = data_len
    else:
        batch_num = input_data.shape[0]
        batch_len = input_data.shape[1]
        data_len = input_data.flatten().shape[0]
    
    # 校验输入长度
    if bias:
        bias_row_num = 0
        if bias_num[0] < 3 and bias_num[0] > 0:
            bias_row_num += 2 ** (bias_num[0] + 1)
        elif bias_num[0] == 3:
            bias_row_num += 32
        assert (batch_len + len(xb_id_list) * bias_row_num) // 320  in [1, 2]
    else:
        assert (batch_len // 320) in [1, 2]

    input_data = input_data.flatten()
    assert data_len <= 0x20000
    
    # 有效输出
    assert isinstance(output_column1, list)
    assert isinstance(output_column2, list)
    assert len(output_column1) == 2
    assert output_column1[0] in [0, 32, 64, 96]
    assert output_column1[1] in [32, 64, 128]
    assert (output_column1[0] + output_column1[1]) <= 128
    assert len(output_column2) == 2

    # 指定tile
    tile_id = tile_id

    # 支持的xb_id_list情况：[0, 2], [0, 1, 2], [4, 6], [4, 5, 6]
    assert xb_id_list in [[0, 2], [0, 1, 2], [4, 6], [4, 5, 6]]
    double_layer_list = [[0,2], [4,6]]
    triple_layer_list = [[0, 1, 2], [4, 5, 6]]

    # 指定tile的模式
    tile_mode = 3 # tile分成四组，每组两个xb
    # 选择输出的 xb 开关
    xb_arr_sel = 0
    if xb_id_list[0] >=4:
        xb_arr_sel = 3
    # 指定需要计算的xb
    xb_id_list = xb_id_list
    # 指定xbg mode
    xbg_mode_list = [0, 0, 0, 0] # xb_id_list in [[0], [2], [4], [6]]:
    if xb_id_list in triple_layer_list:
        index = triple_layer_list.index(xb_id_list)
        xbg_mode_list[index * 2] = 1
    xbg_para_type_list = [0, 0, 0, 0]
    xbg_op_mode_list = [0, 0, 0, 0]
    
    xbg_calc_mode_list = [0,0,0,0]
    xbg_toggle_bit0_list = [0,0,0,0]
    xbg_tile_buf_en0_list = [0,0,0,0]
    xbg_tile_cal_en0_list = [0,0,0,0]
    xbg_fcn_en0_list = [0,0,0,0]
    xbg_relu_en_list = [0,0,0,0]
    xbg_in_pix_type_list = [3,0,0,0]
    xbg_out_pix_type_list = [3,2,0,0]
    xbg_bias_en_list = [0,0,0,0]

    if xb_id_list in double_layer_list:
        index_list = [xb_id_list[0] // 2,  xb_id_list[1] // 2]
    elif xb_id_list in triple_layer_list:
        index_list = [xb_id_list[0] // 2,  xb_id_list[2] // 2]
    
    for index in index_list:
        xbg_calc_mode_list[index] = 3
        xbg_toggle_bit0_list[index] = 1
        xbg_tile_buf_en0_list[index] = 1
        xbg_tile_cal_en0_list[index] = 1
        xbg_fcn_en0_list[index] = 1

    # bias
    xb_start_row_list = []
    xb_bias_input_value_list = []
    if bias[0]:
        xbg_bias_en_list[index_list[0]] = 1
        xb_start_row_list += bias_num[0]
        xb_bias_input_value_list += bias_input_value_list[0]
    if bias[1]:
        xbg_bias_en_list[index_list[1]] = 1
        xb_start_row_list += bias_num[1]
        xb_bias_input_value_list += bias_input_value_list[1]

    # 设置xbg in/out pix type 
    if output_column1[1] == 32:
        xbg_out_pix_type_list[index_list[0]] = 2
        xbg_in_pix_type_list[index_list[1]] = 3
    elif output_column1[1] == 64:
        xbg_out_pix_type_list[index_list[0]] = 3
        xbg_in_pix_type_list[index_list[1]] = 4
    elif output_column1[1] == 128:
        xbg_out_pix_type_list[index_list[0]] = 4
        xbg_in_pix_type_list[index_list[1]] = 5
    

    if not second_relu:
        xbg_relu_en_list[index_list[-1]] = 1
    
    # 指定输入地址列表 & 指定输入数据长度
    if xb_id_list[0] == 0:
        input_addr_list = [0x0, 0x4000, 0x0, 0x0]
        input_len_list = [batch_len,output_column1[1],0x0,0x0]
    else:
        input_addr_list = [0x0, 0x0, 0x0, 0x4000]
        input_len_list = [0x0,0x0,batch_len,output_column1[1]]

    # 指定输入图片大小
    in_img_size_list = [[1,1],[1,1],[1,1],[1,1]]

    # 指定输出地址 输出大小, 第二层固定输出128
    # 固定输出 32 列
    assert output_column2[0] <= 32
    assert (output_column2[0] + output_column2[1]) <= 32
    out_len = 32
    num_type = '8bit'
    # 结果是否relu
    if not second_relu:
        out_len = 64
        num_type = '9bit'

    # 指定xb计算的起始列地址， 起始列为0, 32, 64, 96
    # 指定xb计算的列数 0:32列；1:64列；>=2:128列。
    if xb_id_list in double_layer_list:
        xb_start_column_list = [output_column1[0] // 32,  0]
        if output_column1[1] <= 32:
            xb_column_num_list = [0, 0]
        elif output_column1[1] <= 64:
            xb_column_num_list = [1, 0]
        else:
            xb_column_num_list = [3, 0]

    elif xb_id_list in triple_layer_list:
        xb_start_column_list = [output_column1[0] // 32 , output_column1[0] // 32, 0]
        if output_column1[1] <= 32:
            xb_column_num_list = [0, 0, 0]
        elif output_column1[1] <= 64:
            xb_column_num_list = [1, 1, 0]
        else:
            xb_column_num_list = [3, 3, 0]

    output_addr_list = [0x0, 0x0, 0x0, 0x0]
    xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list[0] == 0:
        output_addr_list = [0x78004000, 0x68080000, 0x0, 0x0]
        xbg_axi_cnt_list = [output_column1[1],out_len,0x0,0x0]
    else:
        output_addr_list = [0x0, 0x0, 0x78004000, 0x68080000]
        xbg_axi_cnt_list = [0x0, 0x0, output_column1[1],out_len]
    
    # 指定输出图片大小
    out_img_size_list = [[1,1],[1,1],[1,1],[1,1]]

    # 指定linebuffer的偏移地址 以及 数据量
    linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
    linebuf_width_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list[0] == 0:
        linebuf_addr_offset_list = [0x0, 0x4000, 0x0, 0x0]
        linebuf_width_list = [batch_len,out_len,0x0,0x0]
    else:
        linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x4000]
        linebuf_width_list = [0x0, 0x0, batch_len,out_len]

    # 指定sfu参数
    relu_th_list = [0x0, 0x0, 0x0, 0x0]
    act_mode_list = [0x0, 0x0, 0x0, 0x0]
    shift_list = shift_list

    # 指定输入数据esram地址
    in_addr_esram = 0x0

    # 加载dummy 数据
    dummy_input = np.zeros((batch_len,),dtype=np.uint8)
    # 指定dummy 数据esram地址
    dummy_input_addr_esram = 0x20000
    # 传输dummy数据到esram
    set_input_to_esram(dummy_input, dummy_input_addr_esram)

    # 初始化输出 以及 输出地址
    out_addr_esram = 0x80000
    output = np.zeros((batch_num,output_column2[1]))
    
    # res_9bit_en
    res_9bit_en = 0
    res_in_sel = 0
    res_out_sel = 0
    if not second_relu:
        res_in_sel = xb_id_list[-1] // 2
        res_out_sel = xb_id_list[-1] // 2
        res_9bit_en = 1

   # 配置运行寄存器
    TileOp( tile_id,  xb_id_list, 
            tile_mode = tile_mode, xb_arr_sel = xb_arr_sel, # tile mode
            xbg_mode_list = xbg_mode_list, xbg_para_type_list = xbg_para_type_list, xbg_op_mode_list = xbg_op_mode_list, # xbg mode 
            xbg_calc_mode_list = xbg_calc_mode_list, xbg_in_pix_type_list = xbg_in_pix_type_list, xbg_out_pix_type_list= xbg_out_pix_type_list, # xbg mode 
            xbg_toggle_bit0_list = xbg_toggle_bit0_list, xbg_tile_buf_en0_list = xbg_tile_buf_en0_list, # xbg mode 
            xbg_tile_cal_en0_list = xbg_tile_cal_en0_list, xbg_fcn_en0_list = xbg_fcn_en0_list, xbg_bias_en_list = xbg_bias_en_list, xbg_relu_en_list= xbg_relu_en_list, # xbg mode 
            xb_start_column_list = xb_start_column_list, xb_column_num_list = xb_column_num_list, # xb column
            xb_start_row_list = xb_start_row_list, # xb row
            input_addr_list = input_addr_list, input_len_list = input_len_list, in_img_size_list = in_img_size_list, # input 
            output_addr_list = output_addr_list, out_img_size_list = out_img_size_list, xbg_axi_cnt_list = xbg_axi_cnt_list, # output 
            linebuf_addr_offset_list = linebuf_addr_offset_list, linebuf_width_list = linebuf_width_list, # linebuffer
            relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list, # sfu
            adc_range_list=adc_range_list, # xb adc range
            res_in_sel = res_in_sel, res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, # res_en
            xb_bias_input_value_list = xb_bias_input_value_list, # bias input
            )

    # tile 输入的基地址
    rd_addr_base = 0x68

    # esram 写入tile buffer 地址
    tile_buffer_input_addr = 0x78000000

    time1 = time.time()
    
    # 传输输入数据到esram
    set_input_to_esram(input_data, in_addr_esram)

    for row in range(batch_num):
        print(f'==================================  第 {row} 次计算开始 ==================================>>>>>')
        #校验数据是否是输入当前batch
        # print('读数据地址 ： %#x' % in_addr_esram)
        read_data_ = get_value_from_esram(in_addr_esram, batch_len)
        assert (read_data_ == input_data[(row * batch_len) : (row + 1) * batch_len]).all()
                  
        print('输入数据 esram_addr: %#x' % (in_addr_esram))
        # dummy 计算
        set_mcu_send_data_to_tile(tile_id, rd_addr_offset = dummy_input_addr_esram, rd_length=batch_len,
                                rd_addr_base=rd_addr_base, rd_cycle=1, write_addr=tile_buffer_input_addr,
                                write_length=batch_len)
        
        set_tile_run(tile_id, tile_mode)
        
        if IsCalcDone(tile_id, output_xb_id = xb_id_list[-1], log_file=f'fc/reg_log/test_fc_two_layer_{time_record}.txt'):
            print('dummy计算结束！！！')
            pass

        # 校准
        set_tile_cali(tile_id, xb_id_list)

        # 输入数据计算
        set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len,
                                rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = tile_buffer_input_addr,
                                write_length = batch_len)
        set_tile_run(tile_id, tile_mode)

        if IsCalcDone(tile_id, output_xb_id = xb_id_list[-1], log_file=f'fc/reg_log/test_fc_two_layer_{time_record}.txt'):
            print('输入数据计算结束！！！')
            pass

        # 获取输出
        output_row_index = row 
        print(f'output_row_index : {output_row_index}')
        output[output_row_index,:] = get_tileop_results(out_addr_esram, out_len, num_type=num_type, out_len_effective=output_column2)
        
        # esram地址递增
        in_addr_esram = in_addr_esram + batch_len
    
    # dump tile reg
    if dump_reg:
        file_path = dump_reg.split('/')
        file_path = '/'.join(file_path[0:-1])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dump_TileReg(tile_id = tile_id, file=dump_reg)
    
    # dump serial reg
    if dump_serial_script:
        file_path = dump_serial_script.split('/')
        file_path = '/'.join(file_path[0:-1])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dump_serial(tile_id = tile_id, file=dump_serial_script)

    time2 = time.time()
    
    print(f"计算 {batch_num} 输入时间为： {time2-time1} s")

   # reset hardware 
    a111_hw_reset()

    return output

def split(ic, split_num):
    
    '''
    input:
        ic: 一个整数
        split_num: 将ic拆分的份数，使得每一份尽可能相同
    return:
        max_num:拆分之后，最大的份数
    '''
    t = math.ceil(ic / split_num)
    w = []
    rest = ic
    for i in range(split_num):
        temp = rest - t
        if temp > 0: 
            w.append(t)
            rest = temp
        else:
            w.append(rest)
    return np.array(w).max(), w

def Conv_one_layer(tile_id, xb_id_list, input_data,  output_column, calc_mode = 3, kernel_size = 3, stride = 1, padding = 0,
                   adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], shift_list = [0x3, 0x3, 0x3, 0x3],
                   relu = True, pool = False, bias = False, bias_num = [0], bias_input_value_list = [[0]],
                   dump_reg = False, dump_serial_script = False, bypass = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        xb_id_list : 计算一层所需要的所有xb, 单次计算的数目只支持1,2,4
        input_data : numpy数组, (b, c, h , w), 支持batch
        output_column: 指定输出的列地址，[起始列，列数], 必须在 0 ~ 128 范围内
        calc_mode: 选择计算模式，0：高4bit，1：低4bit，3：8bit。
        kernel_size: 卷积权重的 kernel size, 仅支持 1,  3, 7
        stride: 卷积滑窗的步长， 支持 1, 2
        padding: 输入是否需要padding
        adc_range_list: 设置每个XB的adc的range, 和xb_id_list 一一对应
        shift_list: 设置每个XB group的移位数值
        relu: 输出是否relu
        pool: 输出是否需要pool
        bias: 是否有bias
        bias_num: 阵列前多少行作为bias, 与xb_id_list 一一对应, 0: 前0行作为bias; 1: 前4行作为bias, 2: 前8行, 3: 前32行
        bias_input_value: bias 输入的大小， 
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
        bypass: 输出是否直接使用ADC的结果, 不经过移位, relu等数字单元 
    输出：
        输出结果, numpy数组
    '''
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = lib.a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')
    
    # 输入拆分
    if padding:
        input_data = pad(input_data, stride=padding)
        padding = 0
        
    # 1.判断是否需要拆分
    if len(input_data.shape) == 3:
        c, h, w = input_data.shape
        input_data = np.expand_dims(input_data,axis=0)
        b = 1
    else:
        c, h, w = input_data.shape[1:]
        b = input_data.shape[0]
        
    batch_len = c * h * w

    # tile buffer 的容量限制
    data_limit = 0x4000
    
    if batch_len > data_limit:
        # 2.需要拆分
        # pass
        out_img_h = int(math.floor((h - kernel_size) // stride) + 1 )
        out_img_w = int(math.floor((w - kernel_size) // stride) + 1 )
        # 3.判断是否需要对列方向进行拆分
        if c > data_limit:
            raise ValueError(f'通道数据量{c} 大于 缓存容量{data_limit}, 暂不支持通道数的拆分!!!')
        elif c * w * kernel_size >= data_limit:
            # 拆分列方向
            split_num = 1
            while True:
                # 4.计算被拆分的计算次数
                max_col_split_num, col_split_num = split(out_img_w, split_num)
                # 5. 根据拆分的计算次数 算出输入的数据量
                col_number = kernel_size + stride * (max_col_split_num - 1)
                # 
                if stride == 2:
                    # 如果是stride =2, 保证输入图像的宽度(col_number)和高度(kernel_size)都为偶数，
                    # 满足阵列计算的要求, 而实际阵列计算不会计算该添加的行
                    if col_number % 2 == 1:
                        col_number += 1
                    if kernel_size % 2 == 1:
                        data_num = c * (kernel_size + 1) * col_number
                    else:
                        data_num = c * kernel_size * col_number
                else:
                    # 拆分列时，行数固定为kernel_size
                    data_num = c * kernel_size * col_number     
                # 6.判断拆分之后的计算量是否大于容量
                if data_num <= data_limit:
                    break
                else:
                    split_num += 1

                if split_num > out_img_w:
                    raise ValueError(f'拆分次数{split_num} 超过图像宽度 {out_img_w} !!!')
            
            # 7. 根据拆分的结果获取原始输入的数据, 并组合最后计算结果
            output_h = []
            for h_ in range(0, h, stride):
                if (h - h_) < kernel_size:
                    continue
                output_w = []
                index = 0
                for w_ in col_split_num:
                    if index == 0:
                        col_start = 0
                        col_end = kernel_size + stride * (w_ - 1)
                    else:
                        col_start = stride * (np.array(col_split_num[:index]).sum())
                        col_end = col_start + kernel_size + stride * (col_split_num[index] - 1)
                        
                    input_data_ = input_data[:,:,h_:h_+kernel_size,col_start:col_end]
                    # 
                    if stride == 2:
                        if kernel_size % 2 == 1:
                            # 如果是stride =2 并且kernel size 是奇数的话，需要添加一行全0，
                            # 使得输入的图像的尺寸为偶数，满足阵列计算的要求, 而实际阵列计算不会计算该添加的行
                            insert_data = np.zeros((b,c,1,input_data_.shape[3]), dtype=input_data_.dtype)
                            input_data_ = np.concatenate([input_data_, insert_data], axis=2)
                        if (col_end - col_start) % 2 == 1:
                            # 如果是stride =2 并且列数是奇数的话，需要添加一列全0， 
                            # 使得输入的图像的尺寸为偶数，满足阵列计算的要求, 而实际阵列计算不会计算该添加的列
                            insert_data = np.zeros((b,c,input_data_.shape[2],1), dtype=input_data_.dtype)
                            input_data_ = np.concatenate([input_data_, insert_data], axis=3)

                    output_ = Conv_one_layer_wo_split_image(tile_id, xb_id_list, input_data_, output_column, calc_mode=calc_mode, kernel_size=kernel_size,
                                stride=stride,  padding = padding, adc_range_list = adc_range_list, shift_list = shift_list,
                                relu= relu, pool=pool, bias=bias, bias_num=bias_num, bias_input_value_list=bias_input_value_list,
                                dump_reg=dump_reg, dump_serial_script=dump_serial_script, bypass=bypass)
                    index += 1    
                    output_w.append(output_)
                output_w = np.concatenate(output_w, axis=3)
                output_h.append(output_w)
            # 8. 输出还原
            output = np.concatenate(output_h, axis=2)
        else:
            # 拆分行方向
            split_num = 2
            while True:
                # 4.计算被拆分的计算次数
                max_row_split_num, row_split_num = split(out_img_h, split_num)
                # 5. 根据拆分的计算次数 算出输入的数据量
                row_number = kernel_size + stride * (max_row_split_num - 1)
                # 
                if row_number == kernel_size:
                    break
                if stride == 2 and kernel_size % 2 == 1:
                    # 如果是stride =2 并且kernel size 是奇数的话，需要添加一行，
                    # 使得输入的图像的尺寸为偶数，满足阵列计算的要求, 而实际阵列计算不会计算该添加的行
                    data_num = c * (row_number + 1) * w
                else:
                    data_num = c * row_number * w
                # 6.判断拆分之后的计算量是否大于容量
                
                if data_num <= data_limit:
                    if stride == 2 :
                        # 判断输入是否满足计算条件
                        if kernel_size % 2 == 1 and (row_number + 1) % 2 == 0:
                            break
                        elif kernel_size % 2 == 0 and (row_number) % 2 == 0:
                            break    
                        else:
                            split_num += 1
                            
                    elif stride == 1:
                        break
                    else:
                        split_num += 1
                else:
                    split_num += 1

                if split_num > out_img_h:
                    raise ValueError(f'拆分次数{split_num} 超过图像高度 {out_img_h} !!!')
            
            # 7. 根据拆分的结果获取原始输入的数据, 并组合最后计算结果
            output = []
            index = 0
            for h_ in row_split_num:
                if index == 0:
                    row_start = 0
                    row_end = kernel_size + stride * (h_ - 1)
                else:
                    row_start = stride * (np.array(row_split_num[:index]).sum())
                    row_end = row_start + kernel_size + stride * (row_split_num[index] - 1)
                if stride == 2 and kernel_size % 2 == 1:
                    # 如果是stride =2 并且kernel size 是奇数的话，需要添加一行全0，
                    # 使得输入的图像的尺寸为偶数，满足阵列计算的要求, 而实际阵列计算不会计算该添加的行
                    input_data_ = input_data[:,:,row_start:row_end,:]
                    insert_data = np.zeros((b,c,1,w), dtype=input_data_.dtype)
                    input_data_ = np.concatenate([input_data_, insert_data], axis=2)
                else:
                    input_data_ = input_data[:,:,row_start:row_end,:]
                output_ = Conv_one_layer_wo_split_image(tile_id, xb_id_list, input_data_, output_column, calc_mode=calc_mode, kernel_size=kernel_size,
                              stride=stride,  padding = padding, adc_range_list = adc_range_list, shift_list = shift_list,
                              relu= relu, pool=pool, bias=bias, bias_num=bias_num, bias_input_value_list=bias_input_value_list,
                              dump_reg=dump_reg, dump_serial_script=dump_serial_script, bypass=bypass)
                index += 1    
                output.append(output_)
            # 8. 输出还原
            output = np.concatenate(output, axis=2)
        
    else:
        # 不需要拆分
        output = Conv_one_layer_wo_split_image(tile_id, xb_id_list, input_data, output_column, calc_mode=calc_mode, kernel_size=kernel_size,
                              stride=stride,  padding = padding, adc_range_list = adc_range_list, shift_list = shift_list,
                              relu= relu, pool=pool, bias=bias, bias_num=bias_num, bias_input_value_list=bias_input_value_list,
                              dump_reg=dump_reg, dump_serial_script=dump_serial_script, bypass=bypass)
    
    # reset hardware 
    a111_hw_reset()
    
    return output
    
def Conv_one_layer_wo_split_image(tile_id, xb_id_list, input_data,  output_column, calc_mode = 3, kernel_size = 3, stride = 1, padding = 0,
                   adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], shift_list = [0x3, 0x3, 0x3, 0x3],
                   relu = True, pool = False, bias = False, bias_num = [0], bias_input_value_list = [[0]],
                   dump_reg = False, dump_serial_script = False, bypass = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        xb_id_list : 计算一层所需要的所有xb, 单次计算的数目只支持1,2,4
        input_data : numpy数组, (b, c, h , w), 支持batch
        output_column: 指定输出的列地址，[起始列，列数], 必须在 0 ~ 128 范围内
        calc_mode: 选择计算模式，0：高4bit，1：低4bit，3：8bit。
        kernel_size: 卷积权重的 kernel size, 仅支持 1,  3, 7
        stride: 卷积滑窗的步长， 支持 1, 2
        padding: 输入是否需要padding
        adc_range_list: 设置每个XB的adc的range, 和xb_id_list 一一对应
        shift_list: 设置每个XB group的移位数值
        relu: 输出是否relu
        pool: 输出是否需要pool
        bias: 是否有bias
        bias_num: 阵列前多少行作为bias, 与xb_id_list 一一对应, 0: 前0行作为bias; 1: 前4行作为bias, 2: 前8行, 3: 前32行
        bias_input_value: bias 输入的大小， 
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
        bypass: 输出是否直接使用ADC的结果, 不经过移位, relu等数字单元 
    输出：
        输出结果, numpy数组
    '''
    
    
    # 判断 stride是否合理
    assert stride in [1,2]
    
    # 判断tile_id是否合理
    assert isinstance(tile_id, int)
    assert tile_id >= 0 and tile_id <=5
    
    # 判断xb_id_list是否合理
    if len(xb_id_list) not in [1,2,4]:
        raise ValueError(f'输入xb_id：{xb_id_list}不满足 1层计算 要求!!!')
    
    if padding:
        input_data = pad(input_data, stride=padding)

    # 判断输入数据是否合理
    assert len(input_data.shape) in [3, 4]
    
    if len(input_data.shape) == 3:
        batch_num = 1
        c, h, w = input_data.shape
        data_len = c * h * w
        batch_len = data_len
        # 转换输入layout （c,h,w) 为 （h,w,c)
        input_data = input_data.transpose(1,2,0).copy()
    else:
        batch_num = input_data.shape[0]
        c, h, w = input_data.shape[1:]
        batch_len = c * h * w
        data_len = batch_num * c * h * w
        # 转换输入layout （b,c,h,w) 为 （b,h,w,c)
        input_data = input_data.transpose(0,2,3,1).copy()
    
    # stride == 2时，输入size只能是偶数
    if stride == 2:
        assert h % 2 == 0
    
    # 指定输出图片大小
    out_img_size_list = [[0,0],[0,0],[0,0],[0,0]]
    out_img_h = int(math.floor((h - kernel_size) // stride) + 1 )
    out_img_w = int(math.floor((w - kernel_size) // stride) + 1 )
    
    # 指定输出 axi_cnt
    # 固定输出 128 列
    out_len = out_img_w * out_img_h * output_column[1]
    num_type = '8bit'
    
    # 初始化总的输出
    output = np.zeros((batch_num, out_len))
    
    # ==== 自适应输入最大地址 ====
    
    # 根据 input_data_limit 拆分 batch number
    # 1. 根据 input_data_limit 与 batch_len 计算最大的单次batch数 batch_num_split (向下取整)
    #    需要根据输入与输出的结果长度共同判断
    # total_esram_capacity = 0x300000 # 总共的片上缓存容量
    # # 通过单个batch的输入与输出之和判定当前输入的batch 数量    
    # batch_num_split = total_esram_capacity // (batch_len + out_len)
    
    # # 2. esram中给输入数据分配的地址空间最大值 （esram 容量限制）
    # # input_data_limit = 0x100000
    # input_data_limit = batch_num_split * batch_len
    
    # while input_data_limit % 256 != 0:
    #     input_data_limit += 1
    #     if (out_len * batch_num_split) > (total_esram_capacity - input_data_limit):
    #         batch_num_split = batch_num_split - 1
    #         input_data_limit = batch_num_split * batch_len
    
    # ==== 确定性输入最大地址 ====
    input_data_limit = 0x180000
    batch_num_split = input_data_limit // batch_len
    
    assert input_data_limit % 256 == 0
    # assert data_len <= input_data_limit
    # assert batch_len <= 0x4000 # 计算单层的数据量小于tile buf的容量
    assert c % 32 == 0 # 输入channel需要是32的倍数
    # input_data = input_data.flatten()
    # 有效输出
    assert isinstance(output_column, list)
    assert len(output_column) == 2
    # output_eff_start = output_column[0] 
    output_eff_num = output_column[1]

    # 指定tile
    tile_id = tile_id

    # 非bypass的情况下，支持的xb_id_list情况：[0], [2], [0, 1], [2, 3], [0, 1, 2, 3], [4], [6], [4,5], [6,7], [4, 5, 6, 7]
    if not bypass:
        assert xb_id_list in [[0], [2], [0, 1], [2, 3], [0, 1, 2, 3], [4], [6], [4,5], [6,7], [4, 5, 6, 7]]
        single_layer_list = [[0], [2], [4], [6]]
    else:
        single_layer_list = [[0], [1], [2], [3], [4], [5], [6], [7]]
        assert xb_id_list in single_layer_list
        
    # single_layer_list = [[0], [2], [4], [6]]
    double_layer_list = [[0,1],[2,3],[4,5],[6,7]]
    quadra_layer_list = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # 指定tile的模式
    tile_mode = 3 # tile分成四组，每组两个xb
    if xb_id_list in quadra_layer_list:
        tile_mode = 1
    
    # 指定bypass模式
    bypass_mode = 0
    adc_bypass_id = None
    bypass_sel = 0
    if bypass:
        bypass_mode = 1
        adc_bypass_id = xb_id_list
        bypass_sel = xb_id_list[0] % 2
        
    xb_arr_sel = 0
    if xb_id_list[0] >=4:
        xb_arr_sel = 3
    
    # 指定xbg mode
    xbg_mode_list = [0, 0, 0, 0] # xb_id_list in [[0], [2], [4], [6]]:
    xbg_calc_mode_list = [0,0,0,0]
    xbg_toggle_bit0_list = [0,0,0,0]
    xbg_tile_buf_en0_list = [0,0,0,0]
    xbg_tile_cal_en0_list = [0,0,0,0]
    xbg_relu_en_list = [0,0,0,0]
    xbg_in_pix_type_list = [3,0,0,0]
    xbg_out_pix_type_list = [3,0,0,0]
    xbg_kernel_type_list = [0,0,0,0]
    xbg_out_kernel_type_list=[1,1,1,1] # TODO
    xbg_bias_en_list = [0,0,0,0]
    
    xb_start_row_list = [0] * len(xb_id_list)

    index = xb_id_list[0] // 2
    # xbg_calc_mode_list[index] = 3
    assert calc_mode in [0, 1, 3] 
    xbg_calc_mode_list[index] = calc_mode # 指定计算模式
    xbg_toggle_bit0_list[index] = 1
    xbg_tile_buf_en0_list[index] = 1
    xbg_tile_cal_en0_list[index] = 1
    
    xb_bias_input_value_list = [[0]]

    # bias
    if bias:
        xbg_bias_en_list[index] = 1
        assert len(bias_num) == len(xb_id_list)
        xb_start_row_list = bias_num
        xb_bias_input_value_list = bias_input_value_list
    # kernel type
    if kernel_size == 1:
        xbg_kernel_type_list[index] = 0
    elif kernel_size == 3:
        xbg_kernel_type_list[index] = 1
    elif kernel_size == 7:
        xbg_kernel_type_list[index] = 2
    else:
        raise ValueError(f'不支持kernel size {kernel_size} !!!')

    if not relu:
        xbg_relu_en_list[index] = 1
    
    if xb_id_list in double_layer_list :
        index = double_layer_list.index(xb_id_list)
        xbg_mode_list[index] = 1

    elif xb_id_list in quadra_layer_list:
        index = quadra_layer_list.index(xb_id_list)
        xbg_mode_list[index * 2] = 2
    xbg_para_type_list = [0, 0, 0, 0]
    xbg_op_mode_list = [0, 0, 0, 0]
    
    # 设置xbg in pix type 
    xbg_in_pix_type_list[index] = int(math.log(c, 2) - 2)

    # 设置xbg out pix type 
    if output_column[1] == 32:
        xbg_out_pix_type_list[index] = 2
    elif output_column[1] == 64:
        xbg_out_pix_type_list[index] = 3
    elif output_column[1] == 128:
        xbg_out_pix_type_list[index] = 4
    elif output_column[1] == 256:
        xbg_out_pix_type_list[index] = 5
    elif output_column[1] == 512:
        xbg_out_pix_type_list[index] = 6

    # 指定xb计算的起始列地址， 起始列为0, 32, 64, 96
    xb_start_column_list = [0] * len(xb_id_list)

    # 指定xb计算的列数 0:32列；1:64列；>=2:128列。
    xb_column_num_list = [0] * len(xb_id_list)
    if output_column[1] == 64:
        xb_column_num_list = [1] * len(xb_id_list)
    elif output_column[1] == 128:
        xb_column_num_list = [3] * len(xb_id_list)
    
    # 指定输入地址列表
    input_addr_list = [0x0, 0x0, 0x0, 0x0]
    
    # 指定输入图片大小
    in_img_size_list = [[0,0],[0,0],[0,0],[0,0]]
    in_img_size_list[index] = [h , w]

    # 指定输入数据长度
    input_len_list = [0x0,0x0,0x0,0x0]
    # 指定输出地址
    output_addr_list = [0x0, 0x0, 0x0, 0x0]
    # 指定输出axi cnt
    xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
    
    # 指定linebuffer的偏移地址 以及 数据量
    linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
    linebuf_width_list = [0x0, 0x0, 0x0, 0x0]
    # line_buf size
    linebuffer_size = w * c # 只算图像的宽和channel
    
    # 指定输出图片大小
    out_img_size_list[index] = [out_img_h, out_img_w]
    
    # 结果是否relu
    if (not relu) or bypass:
        out_len = 2 * out_len
        num_type = '9bit'
    
    if xb_id_list in single_layer_list:
        # 指定输入数据长度
        index = xb_id_list[0] // 2 
        input_len_list[index] = batch_len
        # 指定输出地址
        output_addr_list[index] = 0x68000000 + input_data_limit
        # 指定输出axi cnt
        xbg_axi_cnt_list[index] = out_len
        # 指定linebuffer的偏移地址 以及 数据量
        linebuf_width_list[index] = linebuffer_size 
    if xb_id_list in double_layer_list:
        # 指定输入数据长度
        index = double_layer_list.index(xb_id_list)
        input_len_list[index] = batch_len
        # 指定输出地址
        output_addr_list[index] = 0x68000000 + input_data_limit
        # 指定输出axi cnt
        xbg_axi_cnt_list[index] = out_len
        # 指定linebuffer的偏移地址 以及 数据量
        linebuf_width_list[index] = linebuffer_size 
    elif xb_id_list in quadra_layer_list:
        # 指定输入数据长度
        index = quadra_layer_list.index(xb_id_list) * 2
        input_len_list[index ] = batch_len 
        # 指定输出地址
        output_addr_list[index ] = 0x68000000 + input_data_limit
        # 指定输出axi cnt
        xbg_axi_cnt_list[index ] = out_len 
        # 指定linebuffer的偏移地址 以及 数据量
        linebuf_width_list[index ] = linebuffer_size
    
    # if xb_id_list in single_layer_list:
    #     index = xb_id_list[0] // 2 
    #     output_addr_list[index] = 0x68000000 + input_data_limit
    # elif xb_id_list in double_layer_list:
    #     index = double_layer_list.index(xb_id_list)
    #     output_addr_list[index] = 0x68000000 + input_data_limit
    # elif xb_id_list in quadra_layer_list:
    #     index = quadra_layer_list.index(xb_id_list)
    #     output_addr_list[index * 2] = 0x68000000 + input_data_limit
     
    # xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
    # if xb_id_list in single_layer_list:
    #     index = xb_id_list[0] // 2 
    #     xbg_axi_cnt_list[index] = out_len
    # elif xb_id_list in double_layer_list:
    #     index = double_layer_list.index(xb_id_list)
    #     xbg_axi_cnt_list[index] = out_len
    # elif xb_id_list in quadra_layer_list:
    #     index = quadra_layer_list.index(xb_id_list)
    #     xbg_axi_cnt_list[index * 2] = out_len 

    # # 指定linebuffer的偏移地址 以及 数据量
    # linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
    # linebuf_width_list = [0x0, 0x0, 0x0, 0x0]
    # # line_buf size
    # linebuffer_size = w * c # 只算图像的宽和channel
    
    # if xb_id_list in single_layer_list:
    #     index = xb_id_list[0] // 2 
    #     linebuf_width_list[index] = linebuffer_size 
    # elif xb_id_list in double_layer_list:
    #     index = double_layer_list.index(xb_id_list)
    #     linebuf_width_list[index] = linebuffer_size
    # elif xb_id_list in quadra_layer_list:
    #     index = quadra_layer_list.index(xb_id_list)
    #     linebuf_width_list[index * 2] = linebuffer_size
    
    # 指定sfu参数
    relu_th_list = [0x0, 0x0, 0x0, 0x0]
    act_mode_list = [0x0, 0x0, 0x0, 0x0]
    shift_list = shift_list
    
    # 初始化输出地址
    out_addr_esram = input_data_limit
    
    # res_9bit_en
    res_9bit_en = 0
    res_in_sel = 0
    res_out_sel = 0
    if (not relu) or bypass:
        res_in_sel = xb_id_list[0] // 2
        res_out_sel = xb_id_list[0] // 2
        res_9bit_en = 1

    # 设置stride
    tile_skip = 0
    if stride == 2:
        # tile_skip = 0001 --> xbg 1
        # tile_skip = 0010 --> xbg 2 
        # tile_skip = 0100 --> xbg 3
        # tile_skip = 1000 --> xbg 4
        tile_skip = 2**(xb_id_list[0] // 2)
        
    # 配置运行寄存器
    TileOp( tile_id,  xb_id_list, 
            tile_mode = tile_mode, bypass_mode=bypass_mode, bypass_sel=bypass_sel, xb_arr_sel = xb_arr_sel,  # tile mode
            xbg_mode_list = xbg_mode_list, xbg_para_type_list = xbg_para_type_list, xbg_op_mode_list = xbg_op_mode_list, # xbg mode 
            xbg_calc_mode_list = xbg_calc_mode_list, xbg_in_pix_type_list = xbg_in_pix_type_list, xbg_out_pix_type_list= xbg_out_pix_type_list, #xbg mode
            xbg_kernel_type_list = xbg_kernel_type_list, xbg_toggle_bit0_list = xbg_toggle_bit0_list, #xbg mode
            xbg_tile_buf_en0_list = xbg_tile_buf_en0_list, xbg_tile_cal_en0_list = xbg_tile_cal_en0_list, # xbg mode 
            xbg_out_kernel_type_list = xbg_out_kernel_type_list, xbg_bias_en_list = xbg_bias_en_list, xbg_relu_en_list = xbg_relu_en_list, # xbg mode 
            xb_start_column_list = xb_start_column_list, xb_column_num_list = xb_column_num_list, # xb column
            xb_start_row_list = xb_start_row_list, # xb row
            input_addr_list = input_addr_list, input_len_list = input_len_list, in_img_size_list = in_img_size_list, # input 
            output_addr_list = output_addr_list, out_img_size_list = out_img_size_list, xbg_axi_cnt_list = xbg_axi_cnt_list, # output 
            linebuf_addr_offset_list = linebuf_addr_offset_list, linebuf_width_list = linebuf_width_list, # linebuffer
            relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list, # sfu
            adc_range_list=adc_range_list, adc_bypass_id=adc_bypass_id, # xb adc range
            res_in_sel = res_in_sel, res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, # res_9bit_en
            xb_bias_input_value_list = xb_bias_input_value_list, # bias input
            tile_skip = tile_skip # stride
            )

    # tile 输入的基地址
    rd_addr_base = 0x68

    # esram 写入tile buffer 地址
    tile_buffer_input_addr = 0x78000000

    time1_ = time.time()
    
    # 用于计数 已经计算的输入数量
    count = 0
    
    try : 
        # 2. 每次传入 batch_num_split 的输入数据，直到当前所有的输入数据计算完成 （batch_num）
        for batch in range(0, batch_num, batch_num_split):
            
            # 指定输入数据esram地址
            in_addr_esram = 0x0
            
            # 
            batch_end = min(batch_num_split+batch, batch_num)
            # 拆分输入
            input_data_ = input_data[batch:batch_end,:,:,:]
            
            # 当前输入的batch 数量
            batch_num_mini = input_data_.shape[0]
            # 
            input_data_ = input_data_.flatten()
            
            # 3. 传输输入数据到esram
            set_input_to_esram(input_data_, in_addr_esram)
            
            for row in range(batch_num_mini):
                
                # 4. 校准
                set_tile_cali(tile_id, xb_id_list)

                # 5. 输入数据计算
                set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len,
                                        rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = tile_buffer_input_addr,
                                        write_length = batch_len)
                set_tile_run(tile_id, tile_mode)

                # 6. 判断计算是否完成
                if IsCalcDone(tile_id, output_xb_id = xb_id_list[0], log_file=f'conv/reg_log/test_conv_one_layer_{time_record}.txt'):
                    pass
                
                # 7. 获取输出
                output[batch + row, :] = get_tileop_results_FPGA(out_addr_esram, out_len, num_type = num_type, out_len_effective=output_column, op_type='CONV', bypass=bypass)
                
                # 8. esram 输入地址递增
                in_addr_esram = in_addr_esram + batch_len
                        
                # 8. 如果 stride == 2，清零寄存器 24， 224
                if stride == 2:
                    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET0, 0x38)
                    a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET1, 0x30)

                count += 1
            
        # 9. 转换 输出output 维度为 (b, h, w, c)
        output = output.reshape(batch_num, out_img_h , out_img_w , output_column[1])
        # 转换 输出维度为（b, c, h, w)
        output = output.transpose(0,3,1,2)
        
    except:
        
        # dump reg
        if dump_reg:
            file_path = dump_reg.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_TileReg(tile_id = tile_id, file=dump_reg)

        # dump serial reg
        if dump_serial_script:
            file_path = dump_serial_script.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_serial(tile_id = tile_id, file=dump_serial_script)
        
        time2_ = time.time()
        print(f"计算 {count} 输入时间为： {time2_-time1_} s")
        
        # reset hardware 
        a111_hw_reset()
        
        raise ValueError(f'Error !!!')

    return output

def Conv_two_layer(tile_id, xb_id_list, input_data, img_size2, output_column1, output_column2, 
                   kernel_size1 = 3, kernel_size2 = 3, stride1 = 1, stride2 = 1, padding1 = 0,
                   padding2 = 0, adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], shift_list = [0x3, 0x3, 0x3, 0x3],
                   second_relu = True, pool = False, bias = [False, False], bias_num = [[0,], [0,]], 
                   bias_input_value_list = [[[0]],[[0]]], para_mode = [0, 0],
                   dump_reg = False, dump_serial_script = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        xb_id_list : 计算两层所需要的所有xb, 单次计算的数目只支持2,3,4
        input_data : numpy数组, 第一层输入图片的维度，(b, c, h , w), 支持batch
        img_size2: 第二层输入图片的维度 (c, h, w)
        output_column1: 第一层指定输出的列地址，[起始列，列数], 必须在 0 ~ 128 范围内
        output_column2: 第二层指定输出的列地址，[起始列，列数], 必须在 0 ~ 128 范围内
        kernel_size1: 第一层卷积权重的 kernel size, 仅支持 1,  3,  7
        kernel_size2: 第二层卷积权重的 kernel size, 支持1, 3, 7
        stride1: 第一层卷积滑窗的步长, 支持 1,2
        stride2: 第二层卷积滑窗的步长, 支持 1,2
        padding1: 第一层的输入padding
        padding2: padding 第一层输出结果, padding的宽度 = 1
        adc_range_list: 设置每个XB的adc的range, 和xb_id_list 一一对应
        shift_list: 设置每个XB group的移位数值
        second_relu: 第二层输出是否relu
        pool: 输出是否需要pool
        bias: 是否有bias.
        bias_num: 阵列前多少行作为bias, 双层列表， 外层与layer一一对应, 内层与xb_id_list一一对应, 0: 前0行作为bias; 1: 前4行作为bias, 2: 前8行, 3: 前32行
        bias_input_value: bias 输入的大小，三层列表, 外层与layer一一对应, 中间层与xb_id_list一一对应, 最内层与xb的行一一对应 
        para_mode: 并行计算, 支持同时计算1, 2个点; 0 --> 只计算1个点; 1 --> 同时计算2个点
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
    输出：
        输出结果, numpy数组
    '''
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')
    
    # 判断stride 是否合理, 目前只支持1,2
    assert stride1 in [1,2]
    assert stride2 in [1,2]

    # 判断tile_id是否合理
    assert isinstance(tile_id, int)
    assert tile_id >= 0 and tile_id <=5
    
    # 判断xb_id_list是否合理
    if len(xb_id_list) not in [2, 3, 4]:
        raise ValueError(f'输入xb_id：{xb_id_list}不满足 1层计算 要求!!!')
    
    if padding1:
        assert isinstance(padding1, int)
        input_data = pad(input_data, stride=padding1)

    # 判断输入数据是否合理
    assert len(input_data.shape) in [3, 4]
    # image1
    if len(input_data.shape) == 3:
        batch_num = 1
        c1, h1, w1 = input_data.shape
        data_len = c1 * h1 * w1
        batch_len1 = data_len
        # 转换输入layout （c,h,w) 为 （h,w,c)
        input_data = input_data.transpose(1,2,0).copy()
    else:
        batch_num = input_data.shape[0]
        c1, h1, w1 = input_data.shape[1:]
        batch_len1 = c1 * h1 * w1
        data_len = batch_num * c1 * h1 * w1
        # 转换输入layout （b,c,h,w) 为 （b,h,w,c)
        input_data = input_data.transpose(0,2,3,1).copy()    
        
    # image2
    c2, h2, w2 = img_size2
    if padding2:
        h2 += 2
        w2 += 2
    batch_len2 = c2 * h2 * w2
    assert data_len <= 0x80000
    assert batch_len1 <= 0x4000 # 计算单层的数据量小于tile buf的容量
    assert batch_len2 <= 0x4000
    assert c1 % 4 == 0
    # assert c2 == output_column1[1]
    assert c2 in [32, 64, 128, 256]
    input_data = input_data.flatten()

    # 有效输出
    assert isinstance(output_column1, list)
    assert isinstance(output_column2, list)
    assert len(output_column1) == 2
    assert output_column1[0] in [0, 32, 64, 96]
    assert output_column1[1] in [32, 64, 128]
    assert (output_column1[0] + output_column1[1]) <= 128
    assert len(output_column2) == 2
    assert output_column2[1] in [32, 64, 128]
    # 指定tile
    tile_id = tile_id

    # 支持的xb_id_list情况：[0, 2], [0, 1, 2], [4, 6], [4, 5, 6]
    assert xb_id_list in [[0, 2], [0, 1, 2], [0, 1, 2, 3], [4, 6], [4, 5, 6], [4, 5, 6, 7]]
    double_layer_list = [[0, 2], [4, 6]]
    triple_layer_list = [[0, 1, 2], [4, 5, 6]]
    quadra_layer_list = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # 获取index list
    if xb_id_list in double_layer_list:
        index_list = [xb_id_list[0] // 2,  xb_id_list[1] // 2]
    elif xb_id_list in triple_layer_list or xb_id_list in quadra_layer_list:
        index_list = [xb_id_list[0] // 2,  xb_id_list[2] // 2]

    # 指定tile的模式
    tile_mode = 3 # tile分成四组，每组两个xb
    # 选择输出的 xb 开关
    xb_arr_sel = 0
    if xb_id_list[0] >=4:
        xb_arr_sel = 3

    # 指定xbg mode
    xbg_mode_list = [0, 0, 0, 0] # xb_id_list in [[0], [2], [4], [6]]:

    # 考虑两个xb合作计算一个点 # TODO
    weight_len1 = kernel_size1 * kernel_size1 * c1
    weight_len2 = kernel_size2 * kernel_size2 * c2
    if weight_len1 > 320 :
        xbg_mode_list[index_list[0]] = 1
    if weight_len2 > 320 :
        xbg_mode_list[index_list[1]] = 1

    # 考虑xb并行计算
    xbg_para_type_list = [0, 0, 0, 0]
    if xb_id_list[0] == 0:
        xbg_para_type_list[0] = para_mode[0]
        xbg_para_type_list[1] = para_mode[1]
    else:
        xbg_para_type_list[2] = para_mode[0]
        xbg_para_type_list[3] = para_mode[1]

    # 考虑 xb concat
    xbg_op_mode_list = [0, 0, 0, 0]
    xbg_calc_mode_list = [0,0,0,0]
    xbg_toggle_bit0_list = [0,0,0,0]
    xbg_tile_buf_en0_list = [0,0,0,0]
    xbg_tile_cal_en0_list = [0,0,0,0]
    xbg_relu_en_list = [0,0,0,0]
    xbg_in_pix_type_list = [3,0,0,0]
    xbg_out_pix_type_list = [3,2,0,0]
    xbg_bias_en_list = [0,0,0,0]
    xbg_kernel_type_list = [0,0,0,0]
    xbg_out_kernel_type_list=[0,0,1,1] # TODO
    xbg_pad_en_list = [0,0,0,0]
    
    for index in index_list:
        xbg_calc_mode_list[index] = 3
        xbg_toggle_bit0_list[index] = 1
        xbg_tile_buf_en0_list[index] = 1
        xbg_tile_cal_en0_list[index] = 1
    # padding
    if padding2:
        xbg_pad_en_list[index_list[0]] = 1

    # bias
    xb_start_row_list = []
    xb_bias_input_value_list = []
    if bias[0]:
        xbg_bias_en_list[index_list[0]] = 1
        xb_start_row_list += bias_num[0]
        xb_bias_input_value_list += bias_input_value_list[0]
    if bias[1]:
        xbg_bias_en_list[index_list[1]] = 1
        xb_start_row_list += bias_num[1]
        xb_bias_input_value_list += bias_input_value_list[1]
    
    # kernel type
    assert kernel_size1 in [1,3,7]
    assert kernel_size2 in [1,3,7]
    kernel_value_list = [1, 3, 7]
    xbg_kernel_type_list[index_list[0]] = kernel_value_list.index(kernel_size1)
    xbg_kernel_type_list[index_list[1]] = kernel_value_list.index(kernel_size2)
    
    # 设置xbg in/out pix type
    if c1 == 32:
        xbg_in_pix_type_list[index_list[0]] = 3
    elif c1 == 64:
        xbg_in_pix_type_list[index_list[0]] = 4
    elif c1 == 128:
        xbg_in_pix_type_list[index_list[0]] = 5
    elif c1 == 256:
        xbg_in_pix_type_list[index_list[0]] = 6

    if c2 == 32:
        xbg_out_pix_type_list[index_list[0]] = 2
        xbg_in_pix_type_list[index_list[1]] = 3
    elif c2 == 64:
        xbg_out_pix_type_list[index_list[0]] = 3
        xbg_in_pix_type_list[index_list[1]] = 4
    elif c2 == 128:
        xbg_out_pix_type_list[index_list[0]] = 4
        xbg_in_pix_type_list[index_list[1]] = 5
    elif c2 == 256:
        xbg_out_pix_type_list[index_list[0]] = 5
        xbg_in_pix_type_list[index_list[1]] = 6
    
    if output_column2[1] == 32:
        xbg_out_pix_type_list[index_list[1]] = 2
    elif output_column2[1] == 64:
        xbg_out_pix_type_list[index_list[2]] = 3
    elif output_column2[1] == 128:
        xbg_out_pix_type_list[index_list[1]] = 4
    elif output_column2[1] == 256:
        xbg_out_pix_type_list[index_list[1]] = 5

    
    if not second_relu:
        xbg_relu_en_list[index_list[-1]] = 1
    
    # 指定输入地址列表 & 指定输入数据长度
    second_layer_input_addr = 0x4000
    # while second_layer_input_addr % 32 != 0:
    #     second_layer_input_addr += 1
    #     if (second_layer_input_addr + batch_len2) >= 0x8000:
    #         raise ValueError(f'第二层的输入地址 : {second_layer_input_addr}, 数据量 : {batch_len2}, 超过了 Tile buffer 大小 !!!')
    #     if second_layer_input_addr % 32 == 0:
    #         break
    if xb_id_list[0] == 0:
        input_addr_list = [0x0, second_layer_input_addr, 0x0, 0x0]
        input_len_list = [batch_len1,batch_len2,0x0,0x0]
    else:
        input_addr_list = [0x0, 0x0, 0x0, second_layer_input_addr]
        input_len_list = [0x0,0x0,batch_len1,batch_len2]

    # 指定输入图片大小
    in_img_size_list = [[1,1],[1,1],[1,1],[1,1]]
    if xb_id_list[0] == 0:
        in_img_size_list = [[h1,w1],[h2,w2],[1,1],[1,1]]
    else:
        in_img_size_list = [[1,1],[1,1],[h1,w1],[h2,w2]]

    # 指定xb计算的起始列地址， 起始列为0, 32, 64, 96
    # 指定xb计算的列数 0:32列；1:64列；>=2:128列。
    if xb_id_list in double_layer_list:
        xb_start_column_list = [output_column1[0] // 32,  output_column2[0] // 32]
        xb_column_num_list = [0, 0]
        # LAYER1
        if output_column1[1] <= 32:
            xb_column_num_list[0] = 0
        elif output_column1[1] <= 64:
            xb_column_num_list[0] = 1
        else:
            xb_column_num_list[0] = 3
        # LAYER2
        if output_column2[1] <= 32:
            xb_column_num_list[1] = 0
        elif output_column2[1] <= 64:
            xb_column_num_list[1] = 1
        else:
            xb_column_num_list[1] = 3

    elif xb_id_list in triple_layer_list:
        xb_start_column_list = [output_column1[0] // 32 , output_column1[0] // 32,  output_column2[0] // 32]
        xb_column_num_list = [0, 0, 0]
        # LAYER1
        if output_column1[1] <= 32:
            xb_column_num_list[0:2] = [0,0]
        elif output_column1[1] <= 64:
            xb_column_num_list[0:2] = [1,1]
        else:
            xb_column_num_list[0:2] = [3,3]
        # LAYER2
        if output_column2[1] <= 32:
            xb_column_num_list[2] = 0
        elif output_column2[1] <= 64:
            xb_column_num_list[2] = 1
        else:
            xb_column_num_list[2] = 3
    
    elif xb_id_list in quadra_layer_list:
        xb_start_column_list = [output_column1[0] // 32 , output_column1[0] // 32, output_column2[0] // 32, output_column2[0] // 32]
        xb_column_num_list = [0, 0, 0]
        # LAYER1
        if output_column1[1] <= 32:
            xb_column_num_list[0:2] = [0,0]
        elif output_column1[1] <= 64:
            xb_column_num_list[0:2] = [1,1]
        else:
            xb_column_num_list[0:2] = [3,3]
        # LAYER2
        if output_column2[1] <= 32:
            xb_column_num_list[2:4] = [0,0]
        elif output_column2[1] <= 64:
            xb_column_num_list[2:4] = [1,1]
        else:
            xb_column_num_list[2:4] = [3,3]
    

    # 指定输出图片大小
    # out image1
    out_img_size_list = [[0,0],[0,0],[0,0],[0,0]]
    out_img_h1 = int(math.floor((h1 - kernel_size1) // stride1) + 1 )
    out_img_w1 = int(math.floor((w1 - kernel_size1) // stride1) + 1 )
    out_img_size_list[index_list[0]] = [out_img_h1, out_img_w1]
    # out image2
    out_img_h2 = int(math.floor((h2 - kernel_size2) // stride2) + 1 )
    out_img_w2 = int(math.floor((w2 - kernel_size2) // stride2) + 1 )
    out_img_size_list[index_list[1]] = [out_img_h2, out_img_w2]

    # 指定输出大小
    assert output_column2[0] <= 128
    assert (output_column2[0] + output_column2[1]) <= 128
    out_len = out_img_h2 * out_img_w2 * output_column2[1]
    num_type = '8bit'
    # 结果是否relu
    if not second_relu:
        out_len = 2 * out_len
        num_type = '9bit'

    # 初始化输出数组 以及 输出地址
    out_addr_esram = 0x80000 # esram 地址
    output = np.zeros((batch_num, out_img_h2 * out_img_w2 * output_column2[1]))
    
    output_addr_list = [0x0, 0x0, 0x0, 0x0] # tile buffer 地址
    xbg_axi_cnt_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list[0] == 0:
        output_addr_list = [0x78000000 + second_layer_input_addr, 0x68080000, 0x0, 0x0]
        xbg_axi_cnt_list = [batch_len2,out_len,0x0,0x0]
    else:
        output_addr_list = [0x0, 0x0, 0x78000000 + second_layer_input_addr, 0x68080000]
        xbg_axi_cnt_list = [0x0, 0x0, batch_len2,out_len]

    # 指定linebuffer的偏移地址 以及 数据量
    linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0]
    linebuf_width_list = [0x0, 0x0, 0x0, 0x0]
    if xb_id_list[0] == 0:
        linebuf_addr_offset_list = [0x0, second_layer_input_addr, 0x0, 0x0]
        linebuf_width_list = [w1 * c1, w2 * c2, 0x0, 0x0]
    else:
        linebuf_addr_offset_list = [0x0, 0x0, 0x0, second_layer_input_addr]
        linebuf_width_list = [0x0, 0x0, w1 * c1, w2 * c2]

    # 指定sfu参数
    relu_th_list = [0x0, 0x0, 0x0, 0x0]
    act_mode_list = [0x0, 0x0, 0x0, 0x0]
    shift_list = shift_list

    # 指定输入数据esram地址
    in_addr_esram = 0x0

    # res_9bit_en
    res_9bit_en = 0
    res_in_sel = 0
    res_out_sel = 0
    if not second_relu:
        res_in_sel = xb_id_list[-1] // 2
        res_out_sel = xb_id_list[-1] // 2
        res_9bit_en = 1

    # 设置stride
    tile_skip = 0
    if stride1 == 2:
        # tile_skip = 0001 --> xbg 1
        # tile_skip = 0100 --> xbg 3
        assert xb_id_list[0] // 2 in [0, 2]
        assert h1 % 2 == 0 # 输入图像大小需要是整数
        
        tile_skip += 2**(xb_id_list[0] // 2)
        
        
    if stride2 == 2:
        # tile_skip = 0010 --> xbg 2 
        # tile_skip = 1000 --> xbg 4
        assert xb_id_list[0] // 2 in [1, 3]
        assert h2 % 2 == 0 # 输入图像大小需要是整数
        
        tile_skip += 2**(xb_id_list[0] // 2)
    
    # 配置运行寄存器
    TileOp( tile_id,  xb_id_list, 
            tile_mode = tile_mode, xb_arr_sel = xb_arr_sel, # tile mode
            xbg_mode_list = xbg_mode_list, xbg_para_type_list = xbg_para_type_list, xbg_op_mode_list = xbg_op_mode_list, # xbg mode 
            xbg_calc_mode_list = xbg_calc_mode_list, xbg_in_pix_type_list = xbg_in_pix_type_list, xbg_out_pix_type_list= xbg_out_pix_type_list, #xbg mode
            xbg_kernel_type_list = xbg_kernel_type_list, xbg_toggle_bit0_list = xbg_toggle_bit0_list, #xbg mode
            xbg_tile_buf_en0_list = xbg_tile_buf_en0_list, xbg_tile_cal_en0_list = xbg_tile_cal_en0_list, # xbg mode 
            xbg_out_kernel_type_list = xbg_out_kernel_type_list, xbg_bias_en_list = xbg_bias_en_list, xbg_relu_en_list = xbg_relu_en_list, # xbg mode 
            xb_start_column_list = xb_start_column_list, xb_column_num_list = xb_column_num_list, # xb column
            xb_start_row_list = xb_start_row_list, # xb row
            input_addr_list = input_addr_list, input_len_list = input_len_list, in_img_size_list = in_img_size_list, # input 
            output_addr_list = output_addr_list, out_img_size_list = out_img_size_list, xbg_axi_cnt_list = xbg_axi_cnt_list, # output 
            linebuf_addr_offset_list = linebuf_addr_offset_list, linebuf_width_list = linebuf_width_list, # linebuffer
            relu_th_list = relu_th_list, act_mode_list = act_mode_list, shift_list = shift_list, # sfu
            adc_range_list=adc_range_list, # xb adc range
            res_in_sel = res_in_sel, res_out_sel = res_out_sel, res_9bit_en = res_9bit_en, # res_9bit_en
            xb_bias_input_value_list = xb_bias_input_value_list, # bias input
            pad_en_list = xbg_pad_en_list, # padding
            tile_skip = tile_skip, # stride
            )

    # tile 输入的基地址
    rd_addr_base = 0x68

    # esram 写入tile buffer 地址
    tile_buffer_input_addr = 0x78000000

    time1 = time.time()
    
    # 传输输入数据到esram
    set_input_to_esram(input_data, in_addr_esram)

    for row in range(batch_num):
        print(f'==================================  第 {row} 次计算开始 ==================================>>>>>')
        #校验数据是否是输入当前batch
        # print('读数据地址 ： %#x' % in_addr_esram)
        read_data_ = get_value_from_esram(in_addr_esram, batch_len1)
        assert (read_data_ == input_data[(row * batch_len1) : (row + 1) * batch_len1]).all()

        print('输入数据 esram_addr: %#x' % (in_addr_esram))

        # 校准
        set_tile_cali(tile_id, xb_id_list)

        # 输入数据计算
        set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len1,
                                rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = tile_buffer_input_addr,
                                write_length = batch_len1)
        set_tile_run(tile_id, tile_mode)

        if IsCalcDone(tile_id, output_xb_id = xb_id_list[-1], log_file=f'conv/reg_log/test_conv_two_layer_{time_record}.txt'):
            print('输入数据计算结束！！！')
            pass

        # 获取输出
        output_row_index = row 
        print(f'output_row_index : {output_row_index}')
        output[output_row_index,:] = get_tileop_results(out_addr_esram, out_len, num_type = num_type, out_len_effective=output_column2, op_type='CONV')
        
        # esram地址递增
        in_addr_esram = in_addr_esram + batch_len1

        # # 如果 stride ==2，清零寄存器 24， 224
        if stride1 == 2 or stride2 == 2:
            a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET0, 0x38)
            a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET1, 0x30)
            
    # 转换 输出output 维度为 (b, c, h, w)
    output = output.reshape(batch_num, out_img_h2, out_img_w2, output_column2[1])
    output = output.transpose(0,3,1,2)

    # dump reg
    if dump_reg:
        file_path = dump_reg.split('/')
        file_path = '/'.join(file_path[0:-1])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dump_TileReg(tile_id = tile_id, file=dump_reg)

    # dump serial reg
    if dump_serial_script:
        file_path = dump_serial_script.split('/')
        file_path = '/'.join(file_path[0:-1])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        dump_serial(tile_id = tile_id, file=dump_serial_script)

    time2 = time.time()
    
    print(f"计算 {batch_num} 输入时间为： {time2-time1} s")

    # reset hardware 
    a111_hw_reset()
    
    return output

def GlobalAvgPool(tile_id, input_data, dump_reg = False, dump_serial_script = False):
    '''
    输入：
        tile_id : tile_id number, int, 0~5
        input_data : numpy数组, 第一层输入图片的维度，(b, c, h , w), 支持 batch
        dump_reg : 文件路径, 寄存器状态保存路径, default=False
        dump_serial_script : 文件路径, 生成串口脚本保存路径, default=False
    输出：
        output: 全局池化之后的数据
    '''
    
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = lib.a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')
    
    # 数据类型转换
    input_data = input_data.astype(np.uint8)
    
    # 判断输入数据是否合理
    assert len(input_data.shape) in [3, 4]
    
    if len(input_data.shape) == 3:
        batch_num = 1
        c, h, w = input_data.shape
        data_len = c * h * w
        batch_len = data_len
        # 转换输入layout （c,h,w) 为 （h,w,c)
        input_data = input_data.transpose(1,2,0).copy()
        input_data = np.expand_dims(input_data, axis=0)
        
    else:
        batch_num = input_data.shape[0]
        c, h, w = input_data.shape[1:]
        batch_len = c * h * w
        data_len = batch_num * c * h * w
        # 转换输入layout （b,c,h,w) 为 （b,h,w,c)
        input_data = input_data.transpose(0,2,3,1).copy()
    # 判断输入数据的通道数是否为[32, 64, 128, 256]
    assert c in [32, 64, 128, 256]
    
    pool_kernel_size = h - 1
    # 指定输出图片大小
    out_img_h = h // pool_kernel_size
    out_img_w = w // pool_kernel_size
    
    # 指定输出 axi_cnt
    # 固定输出 128 列
    out_len = out_img_w * out_img_h * c
    
    # 初始化总的输出
    output = []
    
    # 根据 input_data_limit 拆分 batch number
    
    # 1. 根据 input_data_limit 与 batch_len 计算最大的单次batch数 batch_num_split (向下取整)
    #    需要根据输入与输出的结果长度共同判断
    total_esram_capacity = 0x300000 # 总共的片上缓存容量
    # 通过单个batch的输入与输出之和判定当前输入的batch 数量    
    # batch_num_split = total_esram_capacity // (batch_len + out_len)
    batch_num_split = 156
    
    # 2. esram中给输入数据分配的地址空间最大值 （esram 容量限制）
    # input_data_limit = 0x100000
    input_data_limit = batch_num_split * batch_len
    # print(f' input_data_limit: {input_data_limit}')
    # exit()
    while input_data_limit % 256 != 0:
        input_data_limit += 1
        if (out_len * batch_num_split) > (total_esram_capacity - input_data_limit):
            batch_num_split = batch_num_split - 1
            input_data_limit = batch_num_split * batch_len
    
    assert input_data_limit % 256 == 0

    # 3. 配置tile mode
    set_tile_mode(tile_id, mcu_en2 = 0, mcu_en3 = 0)
    
    # 4. 配置pool模式 (2C寄存器)
    pool_eng_size = int(math.log2(c // 32))
    # 平均池化通过移位实现
    pool_eng_shift = int(round(math.log2(h * w)))
    
    set_tile_resnet_ctl(tile_id= tile_id,  res_in_sel = 0, res_out_sel = 0, 
                        pool_eng_ker=pool_kernel_size, pool_eng_shift=pool_eng_shift,
                        pool_eng_size=pool_eng_size, pool_eng_en=1)
    
    # 5.配置输入的基地址
    rd_addr_base = 0x68
    
    # 6. 每次传入 batch_num_split 的输入数据，直到当前所有的输入数据计算完成 （batch_num）
    try:
        for batch in range(0, batch_num, batch_num_split):
            
            # 指定输入数据esram地址
            in_addr_esram = 0x0
            
            # 
            batch_end = min(batch_num_split+batch, batch_num)
            # 拆分输入
            input_data_ = input_data[batch:batch_end,:,:,:]
            
            # 当前输入的batch 数量
            batch_num_mini = input_data_.shape[0]
            # 
            input_data_ = input_data_.flatten()
            
            # 7. 传输输入数据到esram
            set_input_to_esram(input_data_, in_addr_esram)
            
            # 8. 指定输出地址
            out_addr_esram_keep = 0x68000000 + input_data_limit
            out_addr_esram = out_addr_esram_keep
            
            for i in range(batch_num_mini):
                
                # 9. 设置 MCU 输入输出地址
                set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len,
                                        rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = out_addr_esram,
                                        write_length = out_len)
                
                # 10. 开始计算 (24寄存器：0x3900)
                
                reset_ctl = reg_reset()
                reset_ctl.esram_start = 1
                reset_ctl.pool_eng_start = 1
                reset_ctl.buf_trans0 = 1
                reset_ctl.buf_trans2 = 1
                
                a111_ffi.lib.a111_write_tile_reg(tile_id, a111_ffi.lib.TILE_RESET0, reset_ctl.bits) 
                
                # 接受中断
                if IsCalcDone(tile_id, pool_en=True):
                    # print(f'第{batch + i + 1}个batch 输入数据 池化计算结束！！！')
                    pass
                
                # if batch > 156:
                #     print(batch)
                #     print(f'in_addr_esram: %#x' % in_addr_esram)
                #     print(f'out_addr_esram: %#x' % out_addr_esram)
                #     input()
                
                # 11. 输入，输出地址累加
                in_addr_esram = in_addr_esram + batch_len
                out_addr_esram = out_addr_esram + out_len

                
            # 12. 从esram 中读出输出数据
            rd_buf = a111_ffi.ffi.new(f"uint8_t[{out_len * batch_num_mini}]")
            # print(f'输出数据长度：{out_len}')
            
            out_addr_esram_offset = out_addr_esram_keep - 0x68000000
            a111_ffi.lib.a111_read_data_from_eSRAM( out_addr_esram_offset, rd_buf, out_len * batch_num_mini)
            
            for j in range(out_len * batch_num_mini):
                output.append(rd_buf[j])
        
        assert len(output) == out_len * batch_num
        output = np.array(output)
        # 转换为输出b, c, h, w 格式
        
        output = output.reshape(batch_num, out_img_h, out_img_w, c)
        output = output.transpose(0, 3, 1, 2)

    except:
        
        # dump reg
        if dump_reg:
            file_path = dump_reg.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_TileReg(tile_id = tile_id, file=dump_reg)

        # dump serial reg
        if dump_serial_script:
            file_path = dump_serial_script.split('/')
            file_path = '/'.join(file_path[0:-1])
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            dump_serial(tile_id = tile_id, file=dump_serial_script)

    # reset hardware 
    a111_hw_reset()
    
    return output
    
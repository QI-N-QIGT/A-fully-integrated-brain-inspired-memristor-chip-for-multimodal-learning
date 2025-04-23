from .libA111SDK import *
import time
from .a111_helper import TileOpObj, dump_serial
import math
time_record =  time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime(time.time()))

def A111_config(tileopobj):
    # 硬件初始化
    open_a111()

    # 初始化时钟 CALC
    re = a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')

    if isinstance(tileopobj, TileOpObj):

        TileOp(tile_id = tileopobj.tile_id, xb_id_list = tileopobj.xb_id_list,
            
                tile_mode = tileopobj.tile_mode, bypass_mode = tileopobj.bypass_mode, bypass_sel = tileopobj.bypass_sel, rsv0 = tileopobj.rsv0, 
                pool0_en = tileopobj.pool_en_list[0], pool1_en = tileopobj.pool_en_list[1], pool2_en = tileopobj.pool_en_list[2], pool3_en = tileopobj.pool_en_list[3],
                xb_arr_sel = tileopobj.xb_arr_sel, mcu_en0 = tileopobj.mcu_en_list[0], mcu_en1 = tileopobj.mcu_en_list[1], mcu_en2 = tileopobj.mcu_en_list[2], mcu_en3 = tileopobj.mcu_en_list[3], 
                rsv1 = tileopobj.rsv1, slave_mode = tileopobj.slave_mode, mcu_mode = tileopobj.mcu_mode, res_load = tileopobj.res_load, res_en = tileopobj.res_en ,bp_mode = tileopobj.bp_mode, # tile mode 
                
                xbg_mode_list = tileopobj.xbg_mode_list, xbg_para_type_list = tileopobj.xbg_para_type_list, xbg_op_mode_list = tileopobj.xbg_op_mode_list,
                xbg_calc_mode_list=tileopobj.xbg_calc_mode_list, xbg_in_pix_type_list=tileopobj.xbg_in_pix_type_list, xbg_out_pix_type_list = tileopobj.xbg_out_pix_type_list, 
                xbg_kernel_type_list=tileopobj.xbg_kernel_type_list, xbg_pool_mode_list=tileopobj.xbg_pool_mode_list, xbg_toggle_en0_list = tileopobj.xbg_toggle_en0_list, xbg_toggle_bit0_list=tileopobj.xbg_toggle_bit0_list,
                xbg_tile_buf_en0_list=tileopobj.xbg_tile_buf_en0_list, xbg_tile_cal_en0_list=tileopobj.xbg_tile_cal_en0_list, xbg_fcn_en0_list=tileopobj.xbg_fcn_en0_list, xbg_out_kernel_type_list=tileopobj.xbg_out_kernel_type_list,
                xbg_bias_en_list=tileopobj.xbg_bias_en_list, xbg_relu_en_list=tileopobj.xbg_relu_en_list, xbg_bit_mode_list=tileopobj.xbg_bit_mode_list, # xbg mode

                xb_start_column_list = tileopobj.xb_start_column_list, xb_column_num_list = tileopobj.xb_column_num_list, # xb column
                xb_start_row_list = tileopobj.xb_start_row_list, # xb row
                input_addr_list = tileopobj.input_addr_list, input_len_list = tileopobj.input_len_list, in_img_size_list = tileopobj.in_img_size_list , # input 
                output_addr_list = tileopobj.output_addr_list, out_img_size_list = tileopobj.out_img_size_list, xbg_axi_cnt_list = tileopobj.xbg_axi_cnt_list, # output 
                in_buf_type_list = tileopobj.in_buf_type_list, out_buf_type_list = tileopobj.out_buf_type_list, # buf type
                linebuf_addr_offset_list = tileopobj.linebuf_addr_offset_list, linebuf_width_list = tileopobj.linebuf_width_list, # linebuf
                relu_th_list =tileopobj.relu_th_list, act_mode_list = tileopobj.act_mode_list, shift_list = tileopobj.shift_list, # sfu
                adc_range_list = tileopobj.adc_range_list, # xb adc range
                res_in_sel = tileopobj.res_in_sel, res_out_sel = tileopobj.res_out_sel, res_9bit_en = tileopobj.res_9bit_en, # res_9bit , when relu == false, res_9bit_en = 1
                xb_bias_input_value_list =tileopobj.xb_bias_input_value_list, # xb bias input value
                pad_en_list = tileopobj.pad_en_list, # padding
                tile_skip = tileopobj.tile_skip, # stride
                )
    else:
        raise ValueError(f'暂不支持 tileopobj类型 : {type(tileopobj)} !!!')
    
    

def A111_run(tile_id, xb_id_list, tile_mode, input_data, output_feature_size, in_addr_esram=0x0,
             out_addr_esram = 0x80000, first_op_type = 'FC', last_op_type = 'FC', last_relu = False, dump_reg = False, 
             dump_serial_script = False, verbose = True, log_file = None):
    
    # 初始化硬件
    open_a111()
    
    # 初始化时钟 CALC
    re = a111_pll_clk_init(CLK_MODE_CALC)
    if re:
        raise ValueError('时钟初始化失败！！！')
    else:
        print('时钟初始化成功！！！')

    # tile 输入的基地址
    rd_addr_base = 0x68

    # esram 写入tile buffer 地址
    tile_buffer_input_addr = 0x78000000

    if last_op_type == 'FC':
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
        assert data_len <= 0x80000

        # 有效输出
        assert isinstance(output_feature_size, list)
        assert len(output_feature_size) == 2

        # 加载dummy 数据
        dummy_input = np.zeros((batch_len,),dtype=np.uint8)
        # 指定dummy 数据esram地址
        dummy_input_addr_esram = 0x80000
        # 传输dummy数据到esram
        set_input_to_esram(dummy_input, dummy_input_addr_esram)

        # 指定输出地址 输出大小, 第二层固定输出128
        # 固定输出 128 列
        assert output_feature_size[1] <= 128
        if output_feature_size[1] <= 32:
            out_len = 32
        elif output_feature_size[1] <= 64:
            out_len = 64
        else:
            out_len = 128
        num_type = '8bit'
        # 结果是否relu
        if not last_relu:
            out_len = 2 * out_len
            num_type = '9bit'

        # 初始化输出 以及 输出地址
        output = np.zeros((batch_num, output_feature_size[1]))

    elif last_op_type == 'CONV':

        # 判断输入数据是否合理
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
        assert data_len <= 0x80000
        assert batch_len <= 0x8000 # 计算单层的数据量小于tile buf的容量
        assert c % 4 == 0
        input_data = input_data.flatten()
        # 有效输出
        assert isinstance(output_feature_size, list)
        assert len(output_feature_size) == 3
        out_img_channel, out_img_h, out_img_w = output_feature_size
        out_len = out_img_w * out_img_h * out_img_channel
        num_type = '8bit'
        # 初始化输出 
        output = np.zeros((batch_num,out_len))
        # 结果是否relu
        if not last_relu:
            out_len = 2 * out_len
            num_type = '9bit'

    else:
        raise ValueError(f"暂不支持 tileop type {last_op_type} !!! 仅支持 ['FC','CONV']")
    
    time1 = time.time()

    # 传输输入数据到esram
    set_input_to_esram(input_data, in_addr_esram)

    for row in range(batch_num):
        if verbose:
            print(f'==================================  第 {row} 次计算开始 ==================================>>>>>')
        # 校验数据是否是输入当前batch
        read_data_ = get_value_from_esram(in_addr_esram, batch_len, verbose=verbose)
        assert (read_data_ == input_data[(row * batch_len) : (row + 1) * batch_len]).all()

        if verbose:
            print('输入数据 esram_addr: %#x' % (in_addr_esram))

        # 如果第一层是全连接，则可以进行dummy计算
        if first_op_type == 'FC':

            # dummy 计算
            set_mcu_send_data_to_tile(tile_id, rd_addr_offset = dummy_input_addr_esram, rd_length=batch_len,
                                    rd_addr_base=rd_addr_base, rd_cycle=1, write_addr=tile_buffer_input_addr,
                                    write_length=batch_len)
            
            set_tile_run(tile_id, tile_mode)
            
            if IsCalcDone(tile_id, output_xb_id = xb_id_list[-1], log_file=log_file, verbose=verbose):
                if verbose:
                    print('dummy计算结束！！！')
                pass

        # 校准
        set_tile_cali(tile_id, xb_id_list)

        # 输入数据计算
        set_mcu_send_data_to_tile(tile_id, rd_addr_offset = in_addr_esram, rd_length = batch_len,
                                rd_addr_base = rd_addr_base, rd_cycle = 1, write_addr = tile_buffer_input_addr,
                                write_length = batch_len)
        set_tile_run(tile_id, tile_mode)

        if IsCalcDone(tile_id, output_xb_id = xb_id_list[0], log_file=log_file, verbose=verbose):
            if verbose:
                print('输入数据计算结束！！！')
            pass

        # 获取输出
        output_row_index = row 
        if verbose:
            print(f'output_row_index : {output_row_index}')
        output[output_row_index,:] = get_tileop_results(out_addr_esram, out_len, num_type = num_type,
                                                         out_len_effective=[0, output_feature_size[1]], 
                                                         op_type=last_op_type, verbose = verbose)
        
        # esram地址递增
        in_addr_esram = in_addr_esram + batch_len
            
    if last_op_type == 'CONV':
        # 转换 输出output 维度为 (b, c, h, w)
        output = output.reshape(batch_num, out_img_h , out_img_w , out_img_channel)
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
    if verbose:
        print(f"计算 {batch_num} 输入时间为： {time2-time1} s")

    # reset hardware 
    a111_hw_reset()
    
    return output

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
                in_buf_type_list = [0x06, 0x06, 0x06, 0x06], out_buf_type_list = [0x6, 0x6, 0x6, 0x6], # buf type 
                linebuf_addr_offset_list = [0x0, 0x0, 0x0, 0x0], linebuf_width_list = [0x140, 0x0, 0x0, 0x0], # linebuf
                relu_th_list = [0x0, 0x0, 0x0, 0x0], act_mode_list = [0x0, 0x0, 0x0, 0x0], shift_list = [0x0, 0x0, 0x0, 0x0], # sfu
                adc_range_list = [7, 7, 7, 7, 7, 7, 7, 7], # xb adc range
                res_in_sel = 0, res_out_sel = 0, res_9bit_en = 0, # res_9bit , when relu == false, res_9bit_en = 1
                xb_bias_input_value_list = [[0]], # xb bias input value
                pad_en_list = [0,0,0,0] # padding
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
        # buf type
        self.in_buf_type_list = in_buf_type_list
        self.out_buf_type_list = out_buf_type_list  
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
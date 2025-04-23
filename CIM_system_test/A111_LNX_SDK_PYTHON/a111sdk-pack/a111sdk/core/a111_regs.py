from .a111_reg_type import Reg32


class reg_slave_addr(Reg32):
    FIELDS = (
        ('addr', 32),
    )


class reg_tile_ctl0(Reg32):
    FIELDS = (
        ('tile_mode', 2),
        ('para_type', 2),
        ('op_mode', 2),
        ('calc_mode', 2),
        ('pixel_type_in', 4),
        ('pixel_type_out', 4),
        ('ker_type_in', 3),
        ('pool_mode', 1),
        ('toggle_en', 1),
        ('toggle_bit', 1),
        ('buf_en', 1),
        ('cal_en', 1),
        ('fcn_en', 1),
        ('ker_type_out', 3),
        ('bias_en', 1),
        ('rev_en', 1),
        ('bit_mode', 2),
    )


class reg_tile_ctl2(Reg32):
    FIELDS = (
        ('tile_mode', 2),
        ('bypass_mode', 1),
        ('bypass_sel', 1),
        ('rsv0', 2),
        ('pool0_en', 1),
        ('pool1_en', 1),
        ('pool2_en', 1),
        ('pool3_en', 1),
        ('xb_arr_sel', 2),
        ('mcu_en0', 1),
        ('mcu_en1', 1),
        ('mcu_en2', 1),
        ('mcu_en3', 1),
        ('rsv1', 6),
        ('slave_mode', 3),
        ('mcu_mode', 1),
        ('res_load', 1),
        ('res_en', 1),
        ('bp_mode', 4),
    )


class reg_padding_ctl(Reg32):
    FIELDS = (
        ('pad_en0', 1),
        ('pad_en1', 1),
        ('rsv0', 2),
        ('pad_en2', 1),
        ('pad_en3', 1),
        ('rsv1', 2),
        ('rsv2', 8),
        ('pad0_top', 1),
        ('pad0_bot', 1),
        ('pad0_left', 1),
        ('pad0_right', 1),
        ('pad1_top', 1),
        ('pad1_bot', 1),
        ('pad1_left', 1),
        ('pad1_right', 1),
        ('pad2_top', 1),
        ('pad2_bot', 1),
        ('pad2_left', 1),
        ('pad2_right', 1),
        ('pad3_top', 1),
        ('pad3_bot', 1),
        ('pad3_left', 1),
        ('pad3_right', 1),
    )


class reg_gate_signal(Reg32):
    FIELDS = (
        ('gate_xb', 8),
        ('gate_tile', 1),
    )


class reg_reset(Reg32):
    FIELDS = (
        ('pulse', 1),
        ('mcu_fetch_start', 1),
        ('fsm_clr', 1),
        ('mcu_fsm_clr', 1),
        ('tile_buf_clr0', 1),
        ('tile_buf_clr1', 1),
        ('buf_reset0', 1),
        ('buf_reset1', 1),
        ('buf_trans0', 1),
        ('buf_trans1', 1),
        ('buf_reset2', 1),
        ('buf_trans2', 1),
        ('pool_eng_start', 1),
        ('esram_start', 1),
    )


class reg_res_ctl(Reg32):
    FIELDS = (
        ('res_in_sel', 2),
        ('res_out_sel', 2),
        ('res_9bit_en', 1),
        ('res_shift', 3),
        ('tile_skip', 4),
        ('pool_eng_ker', 4),
        ('pool_eng_shift', 4),
        ('pool_eng_size', 2),
        ('pool_eng_en', 1),
        ('res_out_mode', 1),
    )


class reg_img(Reg32):
    FIELDS = (
        ('width', 9),
        ('rsv0', 7),
        ('height', 9),
        ('rsv1', 7),
    )


class reg_in_axi_cnt(Reg32):
    FIELDS = (
        ('cnt', 32),
    )


class reg_sfu_ctl(Reg32):
    FIELDS = (
        ('relu_th', 9),
        ('act_mode', 1),
        ('rsv0', 2),
        ('shift', 4),
    )


class reg_bl_start(Reg32):
    FIELDS = (
        ('bl0_start', 2),
        ('bl1_start', 2),
        ('bl2_start', 2),
        ('bl3_start', 2),
        ('bl4_start', 2),
        ('bl5_start', 2),
        ('bl6_start', 2),
        ('bl7_start', 2),
    )


class reg_tile_buf_ctl(Reg32):
    FIELDS = (
        ('addr_start', 16),
        ('addr_end', 16),
    )


class reg_tile_buf_ctl_size(Reg32):
    FIELDS = (
        ('size', 24),
    )


class reg_fifo_thresh(Reg32):
    FIELDS = (
        ('thresh', 32),
    )


class reg_fcn_ctl(Reg32):
    FIELDS = (
        ('len0', 16),
        ('len1', 16),
    )


class reg_buf_ctl(Reg32):
    FIELDS = (
        ('offset', 16),
        ('length', 16),
    )


class reg_buf_ctl_type(Reg32):
    FIELDS = (
        ('type0', 3),
        ('type1', 3),
        ('rsv', 26),
    )


class reg_buf_ctl3_en(Reg32):
    FIELDS = (
        ('wrap_en0', 1),
        ('tog_en0', 1),
        ('type0', 3),
        ('rsv0', 3),
        ('wrap_en1', 1),
        ('tog_en1', 1),
        ('type1', 3),
        ('rsv1', 3),
        ('wrap_en2', 1),
        ('tog_en2', 1),
        ('type2', 3),
        ('rsv2', 3),
    )


class reg_buf_ctl7_en(Reg32):
    FIELDS = (
        ('wrap_en3', 1),
        ('tog_en3', 1),
        ('type3', 3),
        ('rsv3', 3),
        ('wrap_en4', 1),
        ('tog_en4', 1),
        ('type4', 3),
        ('rsv4', 3),
    )


class reg_write_addr(Reg32):
    FIELDS = (
        ('addr', 32),
    )


class reg_write_len(Reg32):
    FIELDS = (
        ('len', 24),
    )


class reg_read_ctl(Reg32):
    FIELDS = (
        ('base_addr', 8),
        ('read_cycle', 4),
    )


class reg_int_req(Reg32):
    FIELDS = (
        ('req', 10),
    )


class reg_ana_ctl0(Reg32):
    FIELDS = (
        ('pd_all_xb0', 1),
        ('bl_cal_dac_pd_xb0', 1),
        ('adc_pd_xb0', 1),
        ('bl_dac_n_pd_xb0_xb1', 1),
        ('bl_dac_p_pd_xb0_xb1', 1),
        ('sl_dac_pd_xb0_xb1', 1),
        ('wl_dac_pd_xb0', 1),
        ('sa1_pd_xb0_xb1', 1),
        ('clr_mode_en_xb0', 1),
        ('set_mode_xb0', 1),
        ('reset_mode_xb0', 1),
        ('calc_mode_en_xb0', 1),
        ('calc_mode_sel_xb0', 1),
        ('rsv0', 2),
        ('prg_mode_en_xb0', 1),
        ('pd_all_xb1', 1),
        ('bl_cal_dac_pd_xb1', 1),
        ('adc_pd_xb1', 1),
        ('rsv1', 3),
        ('wl_dac_pd_xb1', 1),
        ('rsv2', 1),
        ('clr_mode_en_xb1', 1),
        ('set_mode_xb1', 1),
        ('reset_mode_xb1', 1),
        ('calc_mode_en_xb1', 1),
        ('calc_mode_sel_xb1', 1),
        ('rsv3', 2),
        ('prg_mode_en_xb1', 1),
    )


class reg_ana_ctl1(Reg32):
    FIELDS = (
        ('rsv0', 8),
        ('clr_mode_en_xb2', 1),
        ('set_mode_xb2', 1),
        ('reset_mode_xb2', 1),
        ('calc_mode_en_xb2', 1),
        ('calc_mode_sel_xb2', 1),
        ('rsv1', 2),
        ('prg_mode_en_xb1', 1),
        ('pd_all_xb2', 1),
        ('bl_cal_dac_pd_xb2', 1),
        ('adc_pd_xb2', 1),
        ('bl_dac_n_pd_xb2_xb3', 1),
        ('bl_dac_p_pd_xb2_xb3', 1),
        ('sl_dac_pd_xb2_xb3', 1),
        ('wl_dac_pd_xb2', 1),
        ('sa1_pd_xb2_xb3', 1),
        ('clr_mode_en_xb3', 1),
        ('set_mode_xb3', 1),
        ('reset_mode_xb3', 1),
        ('calc_mode_en_xb3', 1),
        ('calc_mode_sel_xb3', 1),
        ('rsv2', 2),
        ('prg_mode_en_xb3', 1),
    )


class reg_ana_ctl2(Reg32):
    FIELDS = (
        ('pd_all_xb3_xb4', 1),
        ('bl_cal_dac_pd_xb3_xb4', 1),
        ('adc_pd_xb3_xb4', 1),
        ('bl_dac_n_pd_xb4_xb5', 1),
        ('bl_dac_p_pd_xb4_xb5', 1),
        ('sl_dac_pd_xb4_xb5', 1),
        ('wl_dac_pd_xb3_xb4', 1),
        ('sa1_pd_xb4_xb5', 1),
        ('clr_mode_en_xb4', 1),
        ('set_mode_xb4', 1),
        ('reset_mode_xb4', 1),
        ('calc_mode_en_xb4', 1),
        ('calc_mode_sel_xb4', 1),
        ('rsv1', 2),
        ('prg_mode_en_xb4', 1),
        ('pd_all_xb5', 1),
        ('bl_cal_dac_pd_xb5', 1),
        ('adc_pd_xb5', 1),
        ('rsv2', 3),
        ('wl_dac_pd_xb5', 1),
        ('rsv3', 1),
        ('clr_mode_en_xb5', 1),
        ('set_mode_xb5', 1),
        ('reset_mode_xb5', 1),
        ('calc_mode_en_xb5', 1),
        ('calc_mode_sel_xb5', 1),
        ('rsv4', 2),
        ('prg_mode_en_xb5', 1),
    )


class reg_ana_ctl3(Reg32):
    FIELDS = (
        ('pd_all_xb6', 1),
        ('bl_cal_dac_pd_xb6', 1),
        ('adc_pd_xb6', 1),
        ('bl_dac_n_pd_xb6_xb7', 1),
        ('bl_dac_p_pd_xb6_xb7', 1),
        ('sl_dac_pd_xb6_xb7', 1),
        ('wl_dac_pd_xb6', 1),
        ('sa1_pd_xb6_xb7', 1),
        ('clr_mode_en_xb6', 1),
        ('set_mode_xb6', 1),
        ('reset_mode_xb6', 1),
        ('calc_mode_en_xb6', 1),
        ('calc_mode_sel_xb6', 1),
        ('rsv1', 2),
        ('prg_mode_en_xb6', 1),
        ('pd_all_xb7', 1),
        ('bl_cal_dac_pd_xb7', 1),
        ('adc_pd_xb7', 1),
        ('rsv2', 3),
        ('wl_dac_pd_xb7', 1),
        ('rsv3', 1),
        ('clr_mode_en_xb7', 1),
        ('set_mode_xb7', 1),
        ('reset_mode_xb7', 1),
        ('calc_mode_en_xb7', 1),
        ('calc_mode_sel_xb7', 1),
        ('rsv4', 2),
        ('prg_mode_en_xb7', 1),
    )


class reg_addr_ctl(Reg32):
    FIELDS = (
        ('xb0_start', 2),
        ('xb0_size', 2),
        ('xb1_start', 2),
        ('xb1_size', 2),
        ('xb2_start', 2),
        ('xb2_size', 2),
        ('xb3_start', 2),
        ('xb3_size', 2),
        ('xb4_start', 2),
        ('xb4_size', 2),
        ('xb5_start', 2),
        ('xb5_size', 2),
        ('xb6_start', 2),
        ('xb6_size', 2),
        ('xb7_start', 2),
        ('xb7_size', 2),
    )


class reg_adc_rg_sel(Reg32):
    FIELDS = (
        ('rg_xb0_xb4', 3),
        ('bp_xb0_xb4', 1),
        ('rg_xb1_xb5', 3),
        ('bp_xb1_xb5', 1),
        ('rg_xb2_xb6', 3),
        ('bp_xb2_xb6', 1),
        ('rg_xb3_xb7', 3),
        ('bp_xb3_xb7', 1),
    )


class reg_az_ctl0(Reg32):
    FIELDS = (
        ('az_cyc0', 8),
        ('az_cyc1', 4),
        ('rsv1', 4),
        ('az_cyc2', 8),
        ('az_cyc3', 4),
        ('rsv2', 4),
    )


class reg_az_ctl1(Reg32):
    FIELDS = (
        ('az_cyc4', 8),
        ('az_cyc5', 4),
        ('az_cyc6', 4),
        ('az_cyc7', 4),
        ('az_cyc8', 2),
    )


class reg_cyc_ctl0(Reg32):
    FIELDS = (
        ('adc_fs_h', 3),
        ('adc_fs_l', 3),
        ('bl_wl_cycle', 2),
        ('read_cyc', 8),
        ('read_cyc_fall', 4),
    )


class reg_cali_tri(Reg32):
    FIELDS = (
        ('fcalc_cali', 1),
        ('bcalc_cali', 1),
        ('dac_test_trigger', 1),
        ('adc_lc_test_trigger', 1),
        ('adc_offset_test_trigger', 1),
        ('rsv', 3),
        ('xb0_tri', 1),
        ('xb1_tri', 1),
        ('xb2_tri', 1),
        ('xb3_tri', 1),
        ('xb4_tri', 1),
        ('xb5_tri', 1),
        ('xb6_tri', 1),
        ('xb7_tri', 1),
    )


class reg_fcali_cyc(Reg32):
    FIELDS = (
        ('cycle', 20),
    )


class reg_bcali_cyc(Reg32):
    FIELDS = (
        ('cycle', 20),
    )


class reg_cali_ctl(Reg32):
    FIELDS = (
        ('sa1_fs_h', 4),
        ('sa2_fs_h', 4),
        ('read_cali_mode', 1),
        ('read_cali', 1),
        ('adc_dout_read_mode', 1),
        ('adc_dout_read_cycle', 3),
        ('rsv0', 2),
        ('sa2_sel_a', 4),
        ('sa2_out_add_en', 2),
    )

class reg_pll_ctl(Reg32):
    FIELDS = (
        ('pll_n',       6),
        ('pll_m',       6),
        ('rsv0',        2),
        ('pll_od',      2),
        ('rsv1',        2),
        ('pll_bypass',  1),
        ('rsv2',        10),
        ('pll_asleep',  1),
        ('pll_reset',   1),
        ('rsv3',        1),
    )
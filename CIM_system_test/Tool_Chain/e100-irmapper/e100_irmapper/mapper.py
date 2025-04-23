from .parser import IrParser
from .search import *
from .placement import *
from .helper import *

class mapper(IrParser):
    
    def __init__(self, * , ir=None, device=None, cpu_layer=None,
                 search_method=Base, weight_format='CHW',
                 average_copy=None, specify_para_num=None, specify_split_num = None,
                 place_strategy=OneOnOne, window_copy=False,
                 calc_info=None, relu_fuse = False, pool_fuse = False, split_fuse = False,
                 silu_fuse = False, conv_mul_add_fuse = False, runtime = 'simulation', specify_device_id_list = None,
                 masked_device_id_list = None, adaptive_split_ir = False, type_conversion_list = None,
                 operator_replace = False, target_device = '144k', layer_data_type_dict = None, BN_adaptive_split = False):
        
        ir_ = copy.deepcopy(ir)
        
        if type_conversion_list != None:
            ir_ = insert_type_conversion_op(ir_, type_conversion_list)
        
        # 算子替换
        if operator_replace:
            ir_ = replace_op(ir_)
            # ir_.dump_json(file = f'Fused_ir.yaml')
        
        # 算子融合
        if relu_fuse or pool_fuse or split_fuse or silu_fuse or conv_mul_add_fuse:
            ir_, self.fused_op_dict = fuse_op(ir_, relu_fuse = relu_fuse, pool_fuse=pool_fuse, split_fuse = split_fuse, silu_fuse = silu_fuse,
                                              conv_mul_add_fuse = conv_mul_add_fuse)
            # ir_.dump_json(file = f'Fused_ir_0509.yaml')
            # exit()
        if ir_.devices == None:  
            self.device_ir = make_device_ir(ir_, device)
        else:
            self.device_ir = ir_
        super().__init__(self.device_ir, cpu_layer, specify_device_id_list, masked_device_id_list)
        self.device = device
        self.search_method = search_method
        self.weight_format = weight_format
        self.average_copy = average_copy
        self.specify_split_num = specify_split_num
        self.specify_para_num = specify_para_num
        self.place_strategy = place_strategy
        self.window_copy = window_copy
        self.calc_info = calc_info
        self.runtime = runtime
        self.adaptive_split_ir = adaptive_split_ir
        self.target_device = target_device
        self.layer_data_type_dict = layer_data_type_dict 
        # self.run()
        self.BN_adaptive_split = BN_adaptive_split
        
    def run(self, CIMA_alpha=0, CIMA_method = 'random_serach', CIMA_datawidth = 8, CIMA_dmac_layer = None, CIMA_insert_mul_add_op = None):
        
        # device field
        device_field = ['144k', 'a111', 'cima']
        
        self.place = self.search_method(self.node_info, self.node_weight, self.hardware_config,
                                   weight_format=self.weight_format,
                                   average_copy=self.average_copy,
                                   specify_para_num=self.specify_para_num,
                                   specify_split_num = self.specify_split_num,
                                   place_strategy=self.place_strategy,
                                   window_copy=self.window_copy, ir= self.device_ir, 
                                   adaptive_split_ir = self.adaptive_split_ir,
                                   dmac_layer = CIMA_dmac_layer,
                                   insert_mul_add_op = CIMA_insert_mul_add_op,
                                   BN_adaptive_split = self.BN_adaptive_split
                                   )
        
        if '144k' == self.target_device:
            self.place.run()
            
            place_info = self.place.node_mapping_info
            split_info = self.place.split_num
            copy_info = self.place.average_copy
            self.mapped_ir = make_mapped_ir(self.device_ir,split_info,place_info,copy_info,
                                        cpu_layer=self.cpu_layer,calc_info=self.calc_info,
                                        runtime=self.runtime)
            
        elif 'a111' == self.target_device:
            self.place.run()
            
            place_info = self.place.node_mapping_info
            split_info = self.place.split_num
            copy_info = self.place.average_copy
            
            self.mapped_ir = make_mapped_ir(self.device_ir,split_info,place_info,copy_info,
                                        cpu_layer=self.cpu_layer,calc_info=self.calc_info,
                                        runtime = self.runtime,
                                        device=self.place.device_field,
                                        input_buffer_addr = self.place.input_buffer_addr,
                                        output_buffer_addr = self.place.output_buffer_addr,
                                        in_buf_type = self.place.in_buf_type,
                                        out_buf_type = self.place.out_buf_type, tile_all = self.place.tile_all,
                                        tile_occupied_xb = self.place.tile_occupied_xb)
            
        elif 'cima' == self.target_device:
            self.place.run(CIMA_alpha = CIMA_alpha, CIMA_method=CIMA_method, CIMA_datawidth = CIMA_datawidth)
            
            place_info = self.place.node_mapping_info
            split_info = self.place.split_num
            copy_info = self.place.average_copy
            
            self.mapped_ir = make_mapped_ir(self.device_ir,split_info,place_info,copy_info,
                                            cpu_layer=self.cpu_layer,calc_info=self.calc_info,
                                            device=self.place.device_field,in_line_buffer_addr = self.place.in_line_buffer_addr,
                                            credit_len = self.place.credit_len,
                                            dmac_layer = CIMA_dmac_layer, layer_data_type_dict = self.layer_data_type_dict)
        else:
            raise ValueError (f'暂不支持device {self.target_device}, 目前仅支持目标平台 {device_field} !!! ')
     
    
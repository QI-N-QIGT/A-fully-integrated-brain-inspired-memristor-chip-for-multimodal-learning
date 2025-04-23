from pathlib import Path
from .parser import OnnxParser
from .passop import *
from .shape_operation import *
import copy
from e100_irtool.tools import flatten_layers, layer_graph # noqa

class ConvertONNX(object):
    
    def __init__(self, onnx_file = None, ir_file=None, weight_half_level=None,
                weight_scale = None, fix_layer_name=False, data_range_specify=None, 
                data_clamp_std = 0, store_intermediate_model = False, 
                specify_input_layer = None, specify_output_layer = None):
        '''
        onnx_file: ONNX 模型文件
        ir_file: 需要保存的ir的文件名称
        '''
        self.onnx_file = onnx_file  
        self.ir_file = ir_file
        self.weight_half_level = weight_half_level
        # self.cpu_layer = cpu_layer
        self.weight_scale = weight_scale
        self.fix_layer_name = fix_layer_name
        self.store_intermediate_model = store_intermediate_model 
        self.data_range_specify = data_range_specify
        self.data_clamp_std = data_clamp_std
        self.specify_input_layer = specify_input_layer
        self.specify_output_layer = specify_output_layer
        self._convert()
        
    def _convert(self):
        
        if self.onnx_file == None:
            raise ValueError("缺少输入文件，请输入onnx模型文件！！！")
        
        # 模型解析
        model = onnx.load(self.onnx_file)
        model = load_onnx_model(model,fix_layer_name=self.fix_layer_name,store_intermediate_model=self.store_intermediate_model)
        self.model_parser = OnnxParser(model,self.weight_half_level,
                                       self.weight_scale,self.data_clamp_std, 
                                       self.data_range_specify)

        # IR生成
        self.ir = make_ir()
        
        # 冗余层列表，如果指定了输入、输出层，则之前和之后的层都属于冗余层
        self.all_reduntant_layers = []
        if self.specify_input_layer != None:
            self.model_parser.inputs.clear()
            for sil in self.specify_input_layer:
                self.model_parser.inputs.append(self.model_parser.nodes[sil].input[0])
            # 标记指定输入前面的所有层
            self.get_pre_layers(self.model_parser.inputs)

        # 输入层
        g_inputs = []
        
        for input_name in self.model_parser.inputs:
            input_value_info = self.model_parser.value_infos[input_name]
            input_shape = dim_to_list(input_value_info.type.tensor_type.shape.dim)
            if len(input_shape) == 4:
                in_channel, in_height, in_width = input_shape[1:]
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            elif len(input_shape) == 2:
                in_channel = input_shape[1]
                in_height, in_width = 1, 1
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            elif len(input_shape) == 3:
                in_channel = input_shape[2]
                in_height, in_width = 1, 1
                assert in_height == input_shape[0]
                assert in_width == input_shape[1]
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            else:
                raise ValueError(f"暂不支持该维度{input_shape}的数据格式！！！")
            g_inputs.append(temp_d)
        self.ir.add_layer('graph_input',type='input',inputs=g_inputs)
        
        # 如果指定了输出层，则输出层之后的所有节点不需要转换到 ir
        
        if self.specify_output_layer != None:
            out_node_names = []
            for ol in self.specify_output_layer:
                out_node_names.append(self.model_parser.nodes[ol].output[0])
            # 标记指定输出后面的所有层
            self.get_next_layers(out_node_names)
        
        # 判断中间层是否含有Loop Op
        HasLoopOp = False
        
        # 中间的层 
        MakeIR = MakeIROp()
        for node_name in self.model_parser.nodes.keys():
            if node_name in self.all_reduntant_layers:
                continue
            # print(node_name)
            node = self.model_parser.nodes[node_name]
            op_type = node.op_type
            if op_type in ['LSTM']:
                HasLoopOp = True
            func = getattr(MakeIR,op_type,None)
            if func == None:
                raise ValueError(f'未在 makeop 中实现算子 {op_type} !!!')
            func(self.ir, self.model_parser, node_name)
            
        
        output_layer_name = []
        if self.specify_output_layer != None:
            for sl in self.specify_output_layer:
                output_layer_name.append(self.model_parser.nodes[sl].output[0])
        else:
            output_layer_name = self.model_parser.graph_output
        
        # 输出层
        g_outputs = []
        for out_name in output_layer_name:
            # if self.specify_output_layer != None:
            #     out_node_name = self.model_parser.nodes[out_name].output[0]
            # else:
            #     out_node_name = out_name
            out_value_info = self.model_parser.value_infos[out_name]
            out_shape = dim_to_list(out_value_info.type.tensor_type.shape.dim)
            ref_name = self.model_parser.predecessors[out_name][0].name
            if len(out_shape) == 4:
                out_channel,out_height,out_width = out_shape[1:]
                temp_d = dict(ref=ref_name, channel=out_channel, height=out_height, width=out_width,channel_last=True)
            elif len(out_shape) == 2:
                out_channel = out_shape[1]
                temp_d = dict(ref=ref_name, channel=out_channel, height=1, width=1, channel_last=True)
            else:
                raise ValueError(f"暂不支持该维度{out_shape}的数据格式！！！")
            g_outputs.append(temp_d)
        self.ir.add_layer('graph_output',type='output',inputs=g_outputs)
        
        # Loop FuseOp
        if HasLoopOp:
            self.LoopFuseOp()
            
        # flatten IR
        self.ir.layers = self.ir.flatten_layers()

    def LoopFuseOp(self):
        
        layers_info = copy.deepcopy(self.ir.layers)
        next_layer_dict = self.get_ir_next_layer(layers_info)
        
        for name, layer in layers_info.items():    
            # 判断是否可以进行融合
            can_fuse_loop = False
            fused_layer_name = []    
            if layer.type == 'loop':
                next_layers = next_layer_dict[name]
                
                # 只支持下一层只有一个算子的情况 (第一层算子)
                if len(next_layers) == 1:
                    nl = next_layers[0]
                    if layers_info[nl].type == 'op' and layers_info[nl].op.op_id == 'squeeze':
                        fused_layer_name.append(nl)
                        # 再下一层算子 (第二层算子)
                        next_layers_2nd = next_layer_dict[nl]
                        if len(next_layers_2nd) == 1:
                            nl_2nd = next_layers_2nd[0]    
                            if layers_info[nl_2nd].type == 'op' and layers_info[nl_2nd].op.op_id == 'gather':
                                fused_layer_name.append(nl_2nd)
                                can_fuse_loop = True
                
                # 可以融合
                if can_fuse_loop:
                    # 更新 第三层算子的ref name
                    nl_3nd = next_layer_dict[nl_2nd]
                    for n3 in nl_3nd:
                        nl_3nd_layers_info = self.ir.layers[n3]
                        nl_3nd_layers_info.inputs[0].ref = f'{name}:0'
                    # 删除掉被融合的层
                    for fln in fused_layer_name:
                        self.ir.layers.pop(fln)
            
        # sorted layer, layer 层排序
        self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
        
    def get_next_layers(self, out_node_names):
        
        for on in out_node_names:
            for node in self.model_parser.successors[on]:
                if node.name not in self.all_reduntant_layers:
                    self.all_reduntant_layers.append(node.name)
                IsOutput = True
                inn_ = []
                for i in node.output:
                    if i not in self.model_parser.graph_output:
                        inn_.append(i)
                        IsOutput = False
                if not IsOutput and inn_ != []:
                    self.get_next_layers(inn_)
    
    def get_pre_layers(self, input_node_names):
        
        for inn in input_node_names:
            for node in self.model_parser.predecessors[inn]:
                if node.name not in self.all_reduntant_layers:
                    self.all_reduntant_layers.append(node.name)
                IsInput = True
                inn_ = []
                for i in node.input:
                    if i not in self.model_parser.constant.keys() and i not in self.model_parser.parameters.keys() and i not in self.model_parser.graph_input:
                        inn_.append(i)
                        IsInput = False
                if not IsInput and inn_ != []:
                    self.get_pre_layers(inn_)
                    
    def dump(self):
        if self.ir_file == None:
            # 如果没有输出文件路径，则默认输出到当前运行的文件路径下，文件名称与onnx模型文件名一致
            file_path = os.getcwd()
            ir_file = Path(file_path +'\\'+ self.onnx_file.split('\\')[-1].split('.')[0] + '.yaml')
        else:
            ir_file =  self.ir_file
        self.ir.dump(file=ir_file)
    
    def gen_calc_info(self, calc_info_obj = None):
        '''
        input : calc_info_obj
        '''
        from e100_irmapper.device.c200 import C200CalcInfo
        from e100_irmapper.device.a111 import A111CalcInfo
        calc_info = {}
        weight_quant_scale = self.model_parser.weight_quant_scale
        if weight_quant_scale != {}:
            for key in weight_quant_scale.keys():
                layer_name = key.split('.')[0]
                if isinstance(calc_info_obj, dict):
                    calc_ = calc_info_obj[layer_name]
                    calc_.weight_scale = float(weight_quant_scale[key])
                    calc_info[layer_name] = calc_
                elif isinstance(calc_info_obj, C200CalcInfo) or isinstance(calc_info_obj, A111CalcInfo):
                    copy_ = copy.deepcopy(calc_info_obj)
                    copy_.weight_scale = float(weight_quant_scale[key])
                    calc_info[layer_name] = copy_
                else:
                    raise ValueError(f"calc info {type(calc_info_obj)} 类型错误!!!")
        return calc_info

    def get_ops(self):
        
        # 获取模型的有效运算次数(ops), 乘法与加法各算一次
        MAC_count = 0
        Relu_count = 0
        
        MVM_parameters_count = 0
        for node_name, node in self.model_parser.nodes.items():
            if node_name in self.all_reduntant_layers:
                continue
            if node.op_type in ['Conv', 'MatMul', 'Gemm', 'ConvTranspose']:
                
                # 计算次数: 该层的输出图像的大小的乘积
                output_shape = dim_to_list(self.model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
                if len(output_shape) == 4:
                    calc_times = output_shape[2] * output_shape[3]
                elif len(output_shape) == 2:
                    calc_times = 1
                else:
                    raise ValueError(f'暂不支持输出维度：{output_shape}')
                
                # 单次的计算量: 权重维度的乘积 * 2 (乘加各算1次)
                weight_shape =  dim_to_list(self.model_parser.value_infos[node.input[1]].type.tensor_type.shape.dim)
                calc_numbers = np.prod(np.array(weight_shape)) 
                MVM_parameters_count += calc_numbers / 10 **(3)
                
                # 单层总的计算量: 单次的计算量 * 计算次数
                MAC_count += (calc_numbers / 10 **(3)) * (calc_times / 10 **(3)) * 2 

                print(f'{node_name} MAC operations: {(calc_numbers / 10 **(3)) * (calc_times / 10 **(3)) * 2 / 10**(3)} GOPs')
                print(f'{node_name} Weight Parameters: {round(calc_numbers / 10**(6), 5)} M')
                
            if node.op_type == 'Relu':
                input_shape = dim_to_list(self.model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
                # 计算量: 输入图像的大小
                Relu_count += np.prod(np.array(input_shape))
                
        print(f'Total Conv/FC MAC operations: {round(MAC_count / 10**(3), 4)} GOPs')
        print(f'Total Conv/FC MAC parameters: {round(MVM_parameters_count / 10**(3), 4)} M')
        
        return MAC_count / 10**(3), MVM_parameters_count / 10**(3)

    
    def get_ir_pre_layer(self, layers):
        ''''
        获取当前层名称与前一层的名称的对应字典
        input: {layer_name:layer_object}
        return: {current_layer_name: pre_layer_name}
        '''
        prefix_layer = {}
        for name,layer in layers.items():
            if layer.type not in ['input'] and not (layer.type == 'op' and layer.op.op_id in ['constant']):
                prefix_layer[name] =  []
                for i in layer.inputs:
                    if 'graph_input' not in i.ref:
                        #判断前一层是不是flatten，reshape等算子，如果是，则跳过
                        ref = i.ref
                        if ':' in ref:
                            ref = ref.split(':')[0]
                        prefix_layer[name].append(ref)
                    else:        
                        prefix_layer[name].append(i.ref)
        return prefix_layer   

    def get_ir_next_layer(self, layers):
        '''
        获取当前层名称与下一层的名称的对应字典
        input: {layer_name:layer_object}
        return: {current_layer_name: next_layer_name}
        '''
        next_layer = {}
        pre_layer = self.get_ir_pre_layer(layers)  
        
        for k,v in pre_layer.items():
            # if layers[k].type == 'op' and layers[k].op.op_id not in ['flatten']:
            #     for name in v:
            #         if name not in next_layer.keys():
            #             next_layer[name] = []
            #         next_layer[name].append(k)
            if layers[k].type == 'op' and layers[k].op.op_id in ['flatten']:
                continue
        
            for name in v:
                if name not in next_layer.keys():
                    next_layer[name] = []
                next_layer[name].append(k)
                    
        return next_layer
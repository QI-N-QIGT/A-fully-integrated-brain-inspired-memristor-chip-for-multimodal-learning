from ..helper import *
from ..placement import *
from ..esti_model import *
import warnings
from scipy.spatial.distance import cdist
from e100_irtool.core import make_op
from ..parser import IrParser
from ..self_defined_op.fused_op import *
import re

class Base(object):
    
    def __init__(self, node_info, node_weight, hardware_config, weight_format='CHW',
                 average_copy=None, specify_para_num=None, specify_split_num=None, 
                 place_strategy=OneOnOne, window_copy=False, ir=None,
                 adaptive_split_ir = False, dmac_layer = None, insert_mul_add_op=None, BN_adaptive_split = False):
        '''
        node_info: 字典形式，key为所有节点的信息，{'node_name':{'op_type':STR,'kernel_size':INT,'stride':INT,'calc_num':INT,'in_precision':INT,'shape':[w,h],
                                                              'in_channel':INT,'out_channel':INT,out_precision':INT,'copy_constraint':INT},...}
        node_weight: 字典形式，key为所有需要进行排布的节点名称，未根据硬件的大小进行拆分。
                    value为含有两个元素的列表[w,h]，第一个元素为宽，第二个元素为高，（卷积核需要按照片上部署的方式展开）。
        hardware_config: 字典形式，硬件信息，{'name':[STR],'xb_number':INT,'xb_shape':[w,h],'adc_num':INT,'dac_num':INT,
                                            'dac_precision':INT}
                         其中name 列表信息表征 硬件的层次，从左至右依次递进。
        weight_format: 字符串，默认为CHW，支持{'CHW','HWC'}
        specify_para_num: 字典形式，指定节点的并行方式，复制为并行，p:para_diff_array,r:para_same_array {'node_name':[p,r],...}，默认为None
                            当window_copy为False时，默认same_array为对角线的方式复制
        specify_para_num: 字典形式，指定节点的拆分方式（复制之后拆分），[行拆分的份数，列拆分的份数]
        average_copy: 字典形式，指定节点的复制方式，复制为求平均，默认为None，{'node_name':[r_h,r_w],...}，
        place_strategy: 排布策略，类形式，默认为 OneOnOne
        window_copy: 复制策略，以窗口的形式进行权重复制，类型为 Boolean，参考文献：
                    Zhang, Y., et al. (2021). "Efficient and Robust RRAM-Based Convolutional Weight Mapping With Shifted and Duplicated Kernel." 
                    IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 40(2): 287-300.
        '''
        self.node_info = node_info
        self.node_weight = node_weight
        self.weight_format = weight_format
        self.average_copy = average_copy
        self.specify_para_num = specify_para_num
        self.specify_split_num = specify_split_num
        self.hardware_config = hardware_config
        self.place_strategy = place_strategy
        self.window_copy = window_copy
        self.ir = ir
        self.adaptive_split_ir = adaptive_split_ir
        self.dmac_layer = dmac_layer
        self.insert_mul_add_op = insert_mul_add_op
        # self.run()
        self.BN_adaptive_split = BN_adaptive_split
        
    def get_hardware_info(self):
        '''
        解析参数，获取硬件参数
        '''
        self.XB_num = self.hardware_config['xb_number']
        self.XB_size = self.hardware_config['xb_shape']
        self.hd_name = self.hardware_config['name']
        self.dac_num = self.hardware_config['dac_num']
        self.adc_num = self.hardware_config['adc_num']
        self.dac_precision = self.hardware_config['dac_precision']
        
        # 判断冲突的参数
        if 'a111-tile' in self.hd_name[0] and self.window_copy != False:        
            raise ValueError(f'a111-tile 不支持 window copy的复制方法!!!') 

        # device type a111, 144k
        self.device_field = self.hd_name[0]
        
        
    def split_average(self, CIMA_datawidth = 8):
        
        '''
        自适应根据阵列大小平均切分权重，或者根据给定的切分份数切分权重 node_name -> [r,w,h]/ r为复制并行的值
        '''
        # 先复制权重，再根据阵列大小拆分权重
        # 根据复制情况更新权重的大小
        if self.average_copy != None:
            for i in self.average_copy.keys():
                if i in self.node_weight.keys():
                    w, h = self.node_weight[i]
                    w_ = w * self.average_copy[i][1]
                    h_ = h * self.average_copy[i][0]
                    self.node_weight[i] = [w_, h_]
                else:
                    warnings.warn(f'需要mapping到device的层不包括: {i} !!!')
        
        if self.weight_format == 'HWC':
            XB_size = self.XB_size
            DMAC_size = None
            if 'cima' in self.hd_name[0]:
                # packaged array size
                if CIMA_datawidth == 8:
                    XB_size = [self.XB_size[0] * 2, self.XB_size[1]]
                elif CIMA_datawidth == 4:
                    XB_size = [self.XB_size[0] * 4, self.XB_size[1]]
                
                DMAC_size =  self.hardware_config['dmac_shape']
                  
            self.split_node_weight, self.split_num = split_node_HWC(self.node_weight,self.node_info,self.specify_para_num,
                                                                    XB_size, DMAC_size, self.dmac_layer, device=self.device_field)
            
        elif self.weight_format == 'CHW':
            
            self.split_num = {}
            for i in self.node_weight.keys():
                
                if self.specify_para_num != None and i in self.specify_para_num.keys():
                    p_diff_array,p_same_array = self.specify_para_num[i]
                else:
                    p_diff_array,p_same_array = 1, 1
                
                if self.window_copy and self.node_info[i]['op_type'] in ['conv2d', 'conv_transpose2d']:
                    # 按照window_copy方式复制权重
                    # assert (i in self.specify_split_num.keys())
                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0] # 指定行拆份数
                        _w = self.specify_split_num[i][1] # 指定列拆份数
                        self.split_num[i] = [p_diff_array, p_same_array, _w, _h] #
                    else:
                        self.split_num[i] = [p_diff_array, p_same_array, 1, 1] #
                else:
                    
                    # 按照对角线的方式复制权重
                    self.node_weight[i][1] = self.node_weight[i][1] * p_same_array
                    self.node_weight[i][0] = self.node_weight[i][0] * p_same_array
                    
                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0] # 指定行拆份数
                        _w = self.specify_split_num[i][1] # 指定列拆份数
                    else:
                        _h = math.ceil(self.node_weight[i][1] /  self.XB_size[1])
                        _w = math.ceil(self.node_weight[i][0] /  self.XB_size[0])
                    self.split_num[i] = [p_diff_array,p_same_array, _w, _h]
                    
            if self.window_copy:
                self.split_node_weight, self.split_num = split_node_window_duplicate(self.node_info,self.XB_size,self.split_num)
            else:
                self.split_node_weight = split_node(self.node_weight,self.split_num)
        
        else:
            raise ValueError(f"暂不支持权重格式{self.weight_format}")
        
    def run(self, CIMA_alpha = 0, CIMA_method = 'random_search', CIMA_datawidth = 8):
        '''
        按照给定的策略进行排布。
        self.placed_nodes: 
            拆分之后的节点的具体的位置信息，双层列表形式，内层列表表示放在同一个XB上的层，外层列表不区分往上层级，如果硬件架构
            包括Tile，chip等层级，则根据外层列表的顺序，依次放入tile和chip中，例如[[1,2,3],[4,5],...]在一个只拥有XB层级的
            硬件架构下，表示[1,2,3]和[4,5]分别是两个不同的XB中；而如果是在一个拥有TILE层级的架构下，一个TILE拥有两个XB，则不
            仅表示[1,2,3]和[4,5]分别是两个不同的XB中，还表示它们是属于同一个TILE。
        CIMA_alpha: CIMA架构映射的超参数用于调节节点之间的通讯开销.
        CIMA_method: CIMA架构映射方法.
        '''
        # 获取硬件信息
        self.get_hardware_info()
        
        # 自适应切分权重 node_name -> [r,w,h]
        self.split_average(CIMA_datawidth = CIMA_datawidth)

        # 记录拆分之后的最后层 与之前的层 名称的对应关系 
        self.split_weight_layer_dict = {}
        
        # 记录拆分之后的BN层与之前层的名称的对应关系
        self.split_bn_layer_dict = {}
        
        # 自适应的转换 IR, 按照切分的权重进行数据链接
        if self.adaptive_split_ir:
            
            # 当前层信息
            layers_info = self.ir.layers
            
            next_layer_dict = get_next_layer(self.ir.layers)
            split_layer_name = []
            new_split_num = {}
            
            # # 特殊处理CIMA，resnet50网络结构中，给Conv_67和Conv_68之前插入一个Identity层
            # if 'Conv_67' in layers_info.keys() and 'Conv_68' in layers_info.keys():
            #     identity_name = 'Conv_67_68_Identity'
            #     op_ = make_op('identity')
            #     ref_layer_name = layers_info['Conv_67'].inputs[0].ref
            #     assert ref_layer_name == layers_info['Conv_68'].inputs[0].ref, f'{ref_layer_name}, {layers_info["Conv_68"].inputs[0].ref}'
            #     input_shape = layers_info[ref_layer_name].inputs[0]
            #     inputs_ = [dict(ref=ref_layer_name,channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
            #     outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
            #     self.ir.add_layer(identity_name,op=op_,inputs=inputs_,outputs=outputs_)
            #     # 替换Conv_67和Conv_68的输入
            #     self.ir.layers['Conv_67'].inputs[0].ref = identity_name
            #     self.ir.layers['Conv_68'].inputs[0].ref = identity_name
            
            # # 特殊处理CIMA，resnet50网络结构中，给Split_180复制一份Split_180_Conv_181_182,专门给Conv_181, Conv_182层
            # if 'Conv_181' in layers_info.keys() and 'Conv_182' in layers_info.keys():
            #     identity_name = 'Split_180_Conv_181_182'
            #     copied_layer = layers_info['Split_180'].clone()
            #     self.ir.layers[identity_name] = copied_layer
            #     # 替换Conv_67和Conv_68的输入
            #     self.ir.layers['Conv_181'].inputs[0].ref = f'{identity_name}:0'
            #     self.ir.layers['Conv_182'].inputs[0].ref = f'{identity_name}:1'
        
            
            for k, v in self.split_num.items():
                
                # 判断行列方向是否需要拆分 v[2] 表示列方向拆分次数，v[3]表示行方向拆分次数
                if v[2] * v[3] != 1:
                    current_layer = layers_info[k]
                    # ref_name = current_layer.inputs[0].ref
                    # if ':' in ref_name:
                    #     ref_name = ref_name.split(':')[0]
                    # former_layer = layers_info[ref_name]
                    split_layer_name.append(k)
                    if v[3] > 1:
                        # 判断行方向是否拆分，是的话，插入 split 算子
                        insert_split_node_name = f'{k}_Split'
                        assert current_layer.op.in_channel % v[3] == 0, f'{k}, {v}'
                        axis = 1
                        split = []
                        split_output = []
                        for i in range(v[3]):
                            # 默认输入通道是均匀拆分的
                            split.append(current_layer.op.in_channel // v[3])
                            split_output.append({'channel': int(current_layer.op.in_channel // v[3]),
                                                 'width': current_layer.inputs[0].width,
                                                 'height': current_layer.inputs[0].height})
                        op_ = make_op('split', axis=axis, split=split)
                        split_input = current_layer.inputs
                        self.ir.add_layer(insert_split_node_name, op=op_, inputs=split_input, outputs=split_output)
                        
                    split_in_channel = current_layer.op.in_channel // v[3]
                    # print(k)
                    # print(current_layer.op.out_channel)
                    # print(v)
                    if current_layer.op.out_channel % v[2] != 0:
                        warnings.warn(f"当前层 {k} 输出通道为 {current_layer.op.out_channel}, 拆分次数为 {v[2]}, 拆分为: {math.ceil(current_layer.op.out_channel // v[2])} !!!")
                        current_layer.op.out_channel += 1
                    split_out_channel = int(math.ceil(current_layer.op.out_channel // v[2]))
                    
                    # 获取 输入和输出 图像的高宽
                    in_width = current_layer.inputs[0].width
                    in_height = current_layer.inputs[0].height
                    
                    out_width = current_layer.outputs[0].width
                    out_height = current_layer.outputs[0].height
                    
                    # 判断Conv层之后是否是BN层，且BN层是否也需要自适应拆分
                    IsSplitBN = False
                    if self.BN_adaptive_split:
                        # 当卷积只有一个输出，且下一层是BN层，则认为是需要自适应拆分的
                        if len(next_layer_dict[k]) == 1 and layers_info[next_layer_dict[k][0]].op.op_id == 'batch_norm2d':
                            IsSplitBN = True
                            self.split_bn_layer_dict[next_layer_dict[k][0]] = []
                    # 拆分之后数据的组装顺序，先Concat后Add 或者是先Add 后Concat
                    ConcatFirst = False
                    
                    if ConcatFirst:
                        # 先列方向生成求 concat 的算子，然后再行方向生成求 add 的算子
                        for h_ in range(v[3]):
                            for w_ in range(v[2]):
                                # 重新生成当前算子层
                                new_insert_layer = current_layer.clone()
                                # 如果新插入了split层，则改变当前的输入ref为split之后的名称
                                if v[3] > 1:
                                    new_insert_layer.inputs[0].ref = insert_split_node_name + f':{h_}'
                                
                                new_node_name = k + f'_{h_}_{w_}' 
                                
                                new_insert_layer.inputs[0].channel = split_in_channel
                                new_insert_layer.outputs[0].channel = split_out_channel
                                
                                original_weight_shape = current_layer.weights['weight'].shape
                                if len(original_weight_shape) == 4:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel, original_weight_shape[2], original_weight_shape[3])
                                elif len(original_weight_shape) == 2:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel)
                                else:
                                    raise ValueError(f'暂不支持 权重维度 : {original_weight_shape} !!!')
                                new_insert_layer.op.in_channel = split_in_channel
                                new_insert_layer.op.out_channel = split_out_channel
                                
                                # 判断当前层是否融合了激活函数并且行方向需要拆分，是的话，则需要去掉激活，并再加法之后再融合激活
                                if new_insert_layer.op.op_id in ['fused_conv2d', 'fused_fc'] and v[3] > 1:
                                    if new_insert_layer.op.silu != None:
                                        new_insert_layer.op.silu = None
                                    
                                    if new_insert_layer.op.relu != None:
                                        new_insert_layer.op.relu = None
                                
                                if 'bias' in new_insert_layer.weights.keys():
                                    new_insert_layer.weights['bias'].shape = (split_out_channel)
                                
                                self.ir.layers[new_node_name] = new_insert_layer

                                # 更新split num
                                new_split_num[new_node_name] = [self.split_num[k][0], self.split_num[k][1], 1, 1]

                                # 判断Conv层之后是否是BN层，且BN层是否也需要自适应拆分
                                if IsSplitBN:
                                    # 拆分BN层的名称
                                    insert_bn_node_name = f'{k}_BN_{h_}_{w_}'
                                    # 
                                    BN_layer = layers_info[next_layer_dict[k][0]].clone()
                                    BN_layer.inputs[0].ref = k + f'_{h_}_{w_}'
                                    BN_layer.op.in_channel = split_out_channel
                                    BN_layer.op.out_channel = split_out_channel
                                    # 修改 inputs and outputs attr
                                    BN_layer.inputs[0].channel = split_out_channel
                                    BN_layer.outputs[0].channel = split_out_channel
                                    # 修改 bias weights mean var
                                    BN_layer.op.scale = layers_info[next_layer_dict[k][0]].op.scale[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.bias = layers_info[next_layer_dict[k][0]].op.bias[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_mean = layers_info[next_layer_dict[k][0]].op.input_mean[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_var = layers_info[next_layer_dict[k][0]].op.input_var[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    
                                    self.ir.layers[insert_bn_node_name] = BN_layer
                                    # 记录拆分前后的BN层名称关系
                                    self.split_bn_layer_dict[next_layer_dict[k][0]].append(insert_bn_node_name)
                                    
                            # 如果列方向拆分，则需要插入concat算子
                            if v[2] > 1:
                                 
                                # 插入concat算子            
                                insert_concat_node_name = f'{k}_Concat_{h_}'
                                # 默认在通道维度进行concat
                                op_ = make_op('concat', axis=1)
                                concat_input = []
                                for w_ in range(v[2]):
                                    if IsSplitBN:
                                        ref_name = f'{k}_BN_{h_}_{w_}'
                                    else:
                                        ref_name = k + f'_{h_}_{w_}'
                                    concat_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                                concat_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                                self.ir.add_layer(insert_concat_node_name, op=op_, inputs=concat_input, outputs=concat_output)
                                
                                
                        # 如果行方向拆分，输出端还需要插入add算子
                        if v[3] > 1:
                            insert_add_node_name = f'{k}_Add'
                            op_ = make_op('add')
                            if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                if current_layer.op.silu != None or current_layer.op.relu != None:
                                    op_ = make_op('fused_add')
                                    
                            add_input = []
                            for h_ in range(v[3]):
                                # 需要判断是否进行了concat，如果是，则使用concat的名称作为ref name， 否则可以直接使用拆分之后的卷积层的名称使用
                                ref_name = k + f'_{h_}_0'
                                if v[2] > 1:
                                    ref_name = f'{k}_Concat_{h_}'
                                add_input.append(dict(ref=ref_name, channel=current_layer.op.out_channel, width=out_width, height=out_height))
                            add_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                            
                            # 是否需要融合激活函数
                            if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                if current_layer.op.silu != None:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                    self.ir.layers[insert_add_node_name].op.silu = {'op_id': 'silu'}
                                elif current_layer.op.relu != None:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                    self.ir.layers[insert_add_node_name].op.relu = {'op_id': 'relu'}
                                else:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                            else:
                                self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                        
                        # 记录拆分之后的层最后一个名称
                        if v[3] > 1:
                            self.split_weight_layer_dict[k] = insert_add_node_name
                        else:
                            self.split_weight_layer_dict[k] = insert_concat_node_name
                                
                    else:
                        
                        # 先行方向生成求 add 的算子，然后再列方向生成求 concat 的算子
                        for w_ in range(v[2]):
                            for h_ in range(v[3]):
                                # 重新生成当前算子层
                                new_insert_layer = current_layer.clone()
                                # 如果新插入了split层，则改变当前的输入ref为split之后的名称
                                if v[3] > 1:
                                    new_insert_layer.inputs[0].ref = insert_split_node_name + f':{h_}'
                                
                                new_node_name = k + f'_{h_}_{w_}' 
                                
                                new_insert_layer.inputs[0].channel = split_in_channel
                                new_insert_layer.outputs[0].channel = split_out_channel
                                
                                # print
                                original_weight_shape = current_layer.weights['weight'].shape
                                if len(original_weight_shape) == 4:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel, original_weight_shape[2], original_weight_shape[3])
                                elif len(original_weight_shape) == 2:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel)
                                else:
                                    raise ValueError(f'暂不支持 权重维度 : {original_weight_shape} !!!')
                                new_insert_layer.op.in_channel = split_in_channel
                                new_insert_layer.op.out_channel = split_out_channel
                                
                                # 判断当前层是否融合了激活函数并且行方向需要拆分，是的话，则需要去掉激活，并再加法之后再融合激活
                                if new_insert_layer.op.op_id in ['fused_conv2d', 'fused_fc'] and v[3] > 1:
                                    if new_insert_layer.op.silu != None:
                                        new_insert_layer.op.silu = None
                                    
                                    if new_insert_layer.op.relu != None:
                                        new_insert_layer.op.relu = None
                                
                                if 'bias' in new_insert_layer.weights.keys():
                                    new_insert_layer.weights['bias'].shape = (split_out_channel)
                                
                                self.ir.layers[new_node_name] = new_insert_layer

                                # 更新split num
                                new_split_num[new_node_name] = [self.split_num[k][0], self.split_num[k][1], 1, 1]

                                # 判断Conv层之后是否是BN层，且BN层是否也需要自适应拆分
                                if IsSplitBN:
                                    # 拆分BN层的名称
                                    insert_bn_node_name = f'{k}_BN_{h_}_{w_}'
                                    # 
                                    BN_layer = layers_info[next_layer_dict[k][0]].clone()
                                    BN_layer.inputs[0].ref = k + f'_{h_}_{w_}'
                                    BN_layer.op.in_channel = split_out_channel
                                    BN_layer.op.out_channel = split_out_channel
                                    # 修改 inputs and outputs attr
                                    BN_layer.inputs[0].channel = split_out_channel
                                    BN_layer.outputs[0].channel = split_out_channel
                                    # 修改 bias weights mean var
                                    BN_layer.op.scale = layers_info[next_layer_dict[k][0]].op.scale[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.bias = layers_info[next_layer_dict[k][0]].op.bias[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_mean = layers_info[next_layer_dict[k][0]].op.input_mean[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_var = layers_info[next_layer_dict[k][0]].op.input_var[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    
                                    self.ir.layers[insert_bn_node_name] = BN_layer
                                    # 记录拆分前后的BN层名称关系
                                    self.split_bn_layer_dict[next_layer_dict[k][0]].append(insert_bn_node_name)
                            # 如果行方向拆分，输出端还需要插入add算子
                            if v[3] > 1:
                                insert_add_node_name = f'{k}_Add_{w_}'
                                op_ = make_op('add')
                                if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                    if current_layer.op.silu != None or current_layer.op.relu != None:
                                        op_ = make_op('fused_add')
                                add_input = []
                                for h_ in range(v[3]):
                                    if IsSplitBN:
                                        ref_name = f'{k}_BN_{h_}_{w_}'
                                    else:
                                        ref_name = k + f'_{h_}_{w_}' 
                                    add_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                                add_output = [dict(channel=split_out_channel, width=out_width, height=out_height)]
                                
                                # 是否需要融合激活函数
                                if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                    if current_layer.op.silu != None:
                                        self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                        self.ir.layers[insert_add_node_name].op.silu = {'op_id': 'silu'}
                                    elif current_layer.op.relu != None:
                                        self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                        self.ir.layers[insert_add_node_name].op.relu = {'op_id': 'relu'}
                                    else:
                                        self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                else:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                
                                
                        # 如果列方向拆分，则需要插入concat算子
                        if v[2] > 1:
                            insert_concat_node_name = f'{k}_Concat'
                            # 默认在通道维度进行concat
                            op_ = make_op('concat', axis=1)
                            concat_input = []
                            for w_ in range(v[2]):
                                # 需要判断是否进行了add，如果是，则使用add的名称作为ref name， 否则可以直接使用拆分之后的卷积层的名称使用
                                ref_name = k + f'_0_{w_}'
                                if v[3] > 1:
                                    ref_name = f'{k}_Add_{w_}'
                                concat_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                            concat_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                            self.ir.add_layer(insert_concat_node_name, op=op_, inputs=concat_input, outputs=concat_output)
                                
                        # 记录拆分之后的层最后一个名称
                        if v[2] > 1:
                            self.split_weight_layer_dict[k] = insert_concat_node_name
                        else:
                            self.split_weight_layer_dict[k] = insert_add_node_name
                    
                    # 如果BN层也自适应拆分的话，则需要修改BN蹭的额下一层的ref
                    next_layer_list = next_layer_dict[k]
                    replace_ref_name = k
                    if IsSplitBN:
                        next_layer_list = next_layer_dict[next_layer_dict[k][0]]
                        replace_ref_name = next_layer_dict[k][0]
                    # print(k)
                    # print(next_layer_list)
                    # input()
                    # 将下一层的ref name也进行相应的更改
                    for nl in next_layer_list:
                        c = 0
                        if ConcatFirst:
                            if v[3] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_add_node_name
                                    c += 1
                            elif v[2] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_concat_node_name
                                    c += 1
                        else:
                            if v[2] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_concat_node_name
                                    c += 1
                            elif v[3] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_add_node_name
                                    c += 1
                    
                    # 移除当前层
                    self.ir.layers.pop(k)
                    
                    # 如果BN层也需要自适应拆分，则移除BN层
                    if IsSplitBN:
                        bn_name = next_layer_dict[k][0]
                        # print(bn_name)
                        self.ir.layers.pop(bn_name)
                        
                else:
                    new_split_num[k] = v
            
            
            # print(self.ir.layers.keys())
            # 按照算子的连接关系进行排序           
            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
            
            # 如果是CIMA架构，可以进行算子插入以及算子融合
            if 'cima' in self.hd_name[0]:
                
                # 融合 Add + Split， 以及Concat + Split
                self.ir, _ = fuse_op(self.ir, split_fuse=True)
                
                # 当有算子的输出个数不是2**(n)的倍数时，拆入identity算子使得每层的输出个数为2**(n)
                self.ir = insert_identity_op(self.ir)
                
                # 分级插入算子，减少源的数量
                self.ir = insert_transition_op(self.ir)
                
            # self.ir.dump_json(file=f'Hardware_adaptive_ir_torch_0509.yaml')
            # exit()
            # 更新 split_node_weight
            new_split_node_weight = {}
            for k,v in self.split_node_weight.items():
                k_ = k.split('.')
                if k_[0] in split_layer_name:
                    new_split_node_weight[f'{k_[0]}_{k_[2]}_{k_[3]}.0.0.0'] = v
                else:
                    new_split_node_weight[k] = v
            self.split_node_weight = new_split_node_weight
            
            # 更新node info
            ir_parser = IrParser(ir = self.ir)
            self.node_info = ir_parser.node_info
            
            # 更新 split num
            self.split_num = new_split_num
        
        # print(split_layer_name)
        # print(self.split_node_weight)            
        # 按照给定的策略进行排布
        self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()
        
        if 'rram-144k' in self.hd_name[0]:
            sum_ = len(self.placed_nodes)
        else:
            assert isinstance(self.placed_nodes,dict)
            sum_ = 0
            for i in self.placed_nodes:
                v = self.placed_nodes[i]
                
                ori_lname = ('_').join(i.split('_')[:2])
                if self.dmac_layer != None and ori_lname in self.dmac_layer:
                    continue
                sum_ += len(v)
        rest_xb = self.XB_num - sum_
        print(f'当前需要 {sum_} 个XB !!!')
        if rest_xb < 0 :
            raise ValueError(f'按照当前策略 {self.place_strategy.__name__} 无法放下！至少需要 {sum_} 个XB !!! 当前拥有 {self.XB_num} 个XB !!!')
        # 给每一个阵列赋上device名称
        self.ref_to_device( CIMA_alpha = CIMA_alpha, CIMA_method = CIMA_method, CIMA_datawidth = CIMA_datawidth)
    
    def ref_to_device(self, CIMA_alpha = 0, CIMA_method ='random', CIMA_datawidth = 8):
        '''
        给每一个阵列赋上mapping info, mapping_info 是 DeviceMappinginfo object，
        return:
            返回各层的mapping_info，字典形式，{'node_name':[{'index':[r,h_num,w_num],'device_ref':STR,'device_addr':[h_start,w_start,h,w]},...],...}
            index 为三元素的列表形式用于描述复制和拆分之后的不同的块的位置，索引的顺序为[r][h][w]，例如[1,2,3]表示复制的第一份的权重中切分之后的行方向的第二和
            列方向第三 的权重块，device_ref 指向的硬件即为该权重块的运算设备，device_addr 即为位于该XB的物理位置。
        '''
        self.node_mapping_info = {}
        assert len(self.placed_nodes) <= len(self.hd_name)
        # 144k 资源分配
        if 'rram-144k' in self.hd_name[0]:
            for index in range(len(self.placed_nodes)):            
                device_ref = self.hd_name[index]
                for node_addr in self.placed_nodes[index]:
                    key = list(node_addr.keys())[0]
                    value = list(node_addr.values())[0]
                    for i in range(len(value)):
                        value[i] = int(value[i])
                    name_ = key.split('.')
                    node_name = name_[0]
                    if self.window_copy:
                        index_ = [int(name_[1]),int(name_[2]),int(name_[3].split('_')[0])]
                    else:
                        index_ = [int(name_[1]),int(name_[2]),int(name_[3])]
                    if node_name not in self.node_mapping_info.keys():
                        self.node_mapping_info[node_name] = []
                    mapping_info = C200DeviceMappingInfo(index = index_, device=device_ref, address=value)
                    self.node_mapping_info[node_name].append(mapping_info)
        
        # a111 资源分配
        elif 'a111-tile' in self.hd_name[0]:
            
            # self.in_esram_addr = {}
            self.input_buffer_addr = {}
            self.output_buffer_addr = {}
            # self.out_esram_addr = {}
            self.in_buf_type = {}
            self.out_buf_type = {}
            
            self.tile_all = []
            tile_op = []
            mapped_xb_id_count = 0
            mapped_tile_id_count = 0
            placed_nodes = copy.deepcopy(self.placed_nodes)
            count = 0
            self.layer_occupied_xb = {}
            
            while True:
                if placed_nodes == {}:
                    break
                
                node_name = list(placed_nodes.keys())[0]
                if len(placed_nodes[node_name]) > 4:
                    raise ValueError(f'{node_name} 超过4个xb')
                if len(tile_op) < 2 and (mapped_xb_id_count + len(placed_nodes[node_name])) <= 4:
                    tile_op.append(node_name)
                    # mapped_xb_id_count += len(placed_nodes[node_name])
                    count += 1
                else:
                    self.tile_all.append(tile_op)
                    tile_op = []
                    mapped_tile_id_count += 1
                    mapped_xb_id_count = 0
                    count = 0
                    continue
                
                # 依次给每个拆分的权重分配xb
                self.layer_occupied_xb[node_name] = (len(placed_nodes[node_name]) // 2 + 1) * 2
                count = 0
                for node_addr in placed_nodes[node_name]:
                    if mapped_xb_id_count % 2 == 1 and count == 0:
                        mapped_xb_id_count += 1
                    index = 4 * mapped_tile_id_count + mapped_xb_id_count
                    
                    if index > 48:
                        print(self.tile_all)
                        print(self.placed_nodes)
                        raise ValueError(f'需要的xb数量 超过 总和(48个XB)！！！')
                    
                    device_ref = self.hd_name[index]
                    
                    key = list(node_addr.keys())[0]
                    value = list(node_addr.values())[0]
                    name_ = key.split('.')
                    index_ = [int(name_[1]),int(name_[2]),int(name_[3])]
                    
                    if node_name not in self.node_mapping_info.keys():
                        self.node_mapping_info[node_name] = []
                    mapping_info = A111DeviceMappingInfo(index = index_, device=device_ref, address=value)
                    self.node_mapping_info[node_name].append(mapping_info)
                    # mapped xb id
                    mapped_xb_id_count += 1
                    count += 1
                    
                if len(list(placed_nodes.keys())) == 1:
                    self.tile_all.append(tile_op)
                placed_nodes.pop(node_name)
            
            # 分配内存地址
            esram_start_addr = 0
            self.tile_occupied_xb = {}
            
            # 地址type to list
            buf_size_type = [0x800, 0x1000, 0x1800, 0x2000, 0x2800, 0x3000, 0x4000, 0x8000]
            
            for tile in self.tile_all:
                first_node_name = tile[0]
                device_name = self.node_mapping_info[first_node_name][0].device
                tile_name = '.'.join(device_name.split('.')[0:3])
                self.tile_occupied_xb[tile_name] = []
                tile_start_addr = 0
                for node_name in tile:
                    self.tile_occupied_xb[tile_name].append(self.layer_occupied_xb[node_name])
                    node_info = self.node_info[node_name]
                    # 获取当前层输入的长度 (单个batch)
                    in_len_ = node_info['in_data_len']
                    out_len_ = node_info['out_data_len']
                    # tile in buffer 地址分配, 2k字节对齐
                    tile_in_len = ((in_len_ // 2048) + 1) * 2048
                    if tile_in_len > (32 * 1024):
                        # raise ValueError(f'{tile_in_len} 超过tile buffer 内存上限 32KB')
                        warnings.warn(f'{tile_in_len} 超过tile buffer 内存上限 32KB, 该层 {node_name} 需要拆分进行多次计算')
                        self.in_buf_type[node_name] = 6
                    if (tile_in_len // 2048) <= 6:
                        self.in_buf_type[node_name] = (tile_in_len // 2048) - 1
                    elif (tile_in_len // 2048) <= 8:
                        self.in_buf_type[node_name] = 6
                    else:
                        self.in_buf_type[node_name] = 6 # 分配的最大tile的内存空间为 16k，超过的部分使用wrapper进行处理
                    
                    index = tile.index(node_name)
                    
                    # tile的输入层需要分配tile buffer和esram 地址
                    if index == 0:
                        # esram 地址分配,256字节对齐
                        esram_len = ((in_len_ // 256) + 1) * 256
                        # esram地址分配
                        # self.in_esram_addr[tile_name] = [esram_start_addr, in_len_]
                        esram_start_addr += esram_len
                        # tile buffer 地址分配
                        self.input_buffer_addr[node_name] = {'base':hex(0x78000000), 
                                                             'start':hex(tile_start_addr),
                                                             'end':hex(tile_start_addr + buf_size_type[self.in_buf_type[node_name]])}

                    else: 
                        # self.input_buffer_addr[node_name] = [tile_start_addr-tile_out_len, in_len_]
                        self.input_buffer_addr[node_name] = {'base':hex(0x78000000), 
                                                             'start':hex(tile_start_addr),
                                                             'end':hex(tile_start_addr + buf_size_type[self.in_buf_type[node_name]])}
                    
                    # 对齐输入地址    
                    tile_start_addr += buf_size_type[self.in_buf_type[node_name]]    
                        
                    # tile out buffer 地址 2048字节 对齐
                    tile_out_len = ((out_len_ // 2048 ) + 1) * 2048
                    
                    if tile_out_len > (32 * 1024):
                        # raise ValueError(f'{tile_out_len} 超过tile buffer 内存上限 32KB')
                        warnings.warn(f'{tile_out_len} 超过tile buffer 内存上限 32KB, 该层{node_name} 需要拆分进行多次计算')
                        self.in_buf_type[node_name] = 6
                        
                    if (tile_out_len // 2048) <= 6:
                        self.out_buf_type[node_name] = (tile_in_len // 2048) - 1
                    elif (tile_out_len // 2048) <= 8:
                        self.out_buf_type[node_name] = 6
                    else:
                        self.out_buf_type[node_name] = 6 # 分配的最大tile的内存空间为 16k，超过的部分使用wrapper进行处理
                    
                        
                    # tile的输出层不需要分配tile buffer地址，只需要分配esram 地址
                    if index != (len(tile) - 1):
                        
                        # self.output_buffer_addr[node_name] = [tile_start_addr + tile_in_len, out_len_]
                        self.output_buffer_addr[node_name] = {'base':hex(0x78000000), # TODO
                                                             'start':hex(tile_start_addr),
                                                             'end':hex(tile_start_addr + buf_size_type[self.out_buf_type[node_name]])}
                        
                    else:
                        
                        # self.output_buffer_addr[node_name] = None
                        # esram 地址分配,256字节对齐
                        esram_len = ((out_len_ // 256) + 1) * 256
                        # self.out_esram_addr[tile_name] = [esram_start_addr, out_len_]
                        self.output_buffer_addr[node_name] = {'base':hex(0x68000000), # TODO
                                                             'start':hex(esram_start_addr),
                                                             'end':hex(esram_start_addr + esram_len)}
                        
                        esram_start_addr += esram_len
                    
                    # 对齐输入地址
                    # if index == 0:
                    #     tile_start_addr += (tile_in_len + tile_out_len)
                    # else:
                    #     tile_start_addr += tile_out_len
                    # tile_start_addr += buf_size_type[self.in_buf_type[node_name]]
                        
                    if tile_start_addr >= (32 * 1024):
                        # print(tile)
                        # raise ValueError(f'{self.input_tile_buffer_addr} 超过tile buffer 内存上限 32KB')
                        warnings.warn(f'{self.input_buffer_addr[node_name]} 超过tile buffer 内存上限 32KB, 该层{node_name}需要拆分进行多次计算')
                        # self.in_buf_type[node_name] = 6
        # CIMA 资源分配
        elif 'cima' in self.hd_name[0]:
            
            if self.insert_mul_add_op != None:
                self.ir, self.insert_op_name_dict = insert_mul_add_op(self.ir, mul_add_op=self.insert_mul_add_op)
            else:
                self.insert_op_name_dict = None
            # remove flatten op
            self.ir = remove_flatten_op(self.ir)
                
            # self.ir.dump_json(file='insert_mul_add_op.yaml')
            # exit(1)
            # 获取可mapping的各层与前一层的对应关系
            layer_ref = {}
            for k,v  in self.node_info.items():
                layer_ref[k] = v['ref']
            
            # 可以mapped的xb列表
            available_nodes_xb = copy.deepcopy(self.hd_name)
            
            # 获取mesh的高宽
            device_name = list(self.ir.devices.keys())[0]
            mesh_height = self.ir.devices[device_name].height
            mesh_width = self.ir.devices[device_name].width
            
            # IO cost 系数
            alpha = CIMA_alpha
            # 最大可扩展的子节点数目
            # limit_child_num = self.limit_child_num
            # print(self.dmac_layer)
            if CIMA_method.lower() == 'workload_balance':
                # Workload balance search
                self.node_mapping_info_list, self.record_io_workload, self.transfer_thread_num  = packaged_Workload_balance_search(layer_ref, self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                self.node_info,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True,
                                                                                dmac_layer = self.dmac_layer)
            
            elif CIMA_method.lower() in ['lru_search', 'a_search']: 
                # LRU search (Least Recently Used)
                self.node_mapping_info_list, self.record_io_workload = packaged_LRU_search(layer_ref, self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True,
                                                                                dmac_layer = self.dmac_layer)
            elif CIMA_method.lower() == 'random_search':
                # random search
                self.node_mapping_info_list, self.record_io_workload, self.transfer_thread_num  = packaged_random_search(layer_ref, self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True, 
                                                                                dmac_layer = self.dmac_layer)
            elif CIMA_method.lower() == 'onebyone_search':
                # onebyone search
                self.node_mapping_info_list, self.record_io_workload = onebyone_search(layer_ref,self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True)
            else:
                raise ValueError(f'暂不支持 {CIMA_method} !!!')
            
            self.in_line_buffer_addr = {}
            self.credit_len = {}
            
            # device_linebuffer_assigned
            linebuf_assigned = {}
            
            # 获取各layer的下一个CIM-friendly op
            assert self.ir != None
            next_layer_dict = get_next_layer(self.ir.layers)
            pre_layer_dict = get_pre_layer(self.ir.layers)
            
            # 获取各层的layer
            layers = self.ir.layers
            layers_name = list(layers.keys())
            
            # 逆序排列
            layers_name.reverse()
            
            # 记录mapped的mesh node
            mapping_mesh_node = {}
            
            # 获取所有的core id
            # hosti_core = [(3,0)]
            # ddr_core = [(3,5)]
            # none_core = [(1,5), (2,5), (4,5)]
            # "Empty_Core":["Core0_5", "Core3_5", "Core3_6"],
            # "Hosti_Core":"Core0_4",
            # "DDRI_Core":"Core3_4"
            
            hosti_core = [(0,3), (0, 4), (0,5)]
            ddr_core = [(3, 4), (3,5)]
            
            self.ddr_core_id = (3, 4)
            self.hosti_core_id = (0, 4)
            none_core = []
            cant_mapped_core = hosti_core + ddr_core + none_core
            
            
            all_points = []
            for i in range(mesh_height):
                for j in range(mesh_width):
                    if (i, j) not in cant_mapped_core:
                        all_points.append((i,j))
            
            # 定义全局的Memory 缓存大小
            self.Max_Memory_Size = 0x100000 // 2
            
            # 定义中转线程记录字典 self.transfer_thread_num {Core_id: transfer thread num}
            self.Max_Transfer_Thread_Num = 32
            
            # 定义 thread 在 Dmem中的起始地址为 20KB = 16 * 40 = 640 flit
            self.Thread_Dmem_Base_Addr = 640
            
            # 追踪各个core中mfop线程的数量
            self.Max_MFOP_Num = 1
            self.Mapped_MFOP_Core_Full = []
            self.Mapped_MFOP_Num = {}
            
            for name in layers_name:
                
                if layers[name].type != 'op':
                    continue
                
                if layers[name].op.op_id in ['flatten', 'reshape', 'constant']:
                    continue
                
                if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', 'silu', 'resize', 'relu',
                                             'split', 'add', 'fused_add', 'fused_concat', 'concat', 'mul_add', 'pad', 
                                             'type_conversion', 'identity'] \
                        and name not in self.node_mapping_info_list.keys():
                    # assert name in next_layer_dict.keys()
                    # assert name in pre_layer_dict.keys()
                    
                    # 获取当前层信息
                    in_channel = layers[name].inputs[0].channel
                    if in_channel == 255:
                        in_channel += 1
                        warnings.warn(f'当前层 {name} 插入fake通道, 通道数从{in_channel-1} 变为 {in_channel}!!!')
                    if in_channel % 16 != 0:
                        t = 0
                        while (in_channel + t) % 16 != 0:
                            t += 1
                        warnings.warn(f'当前层 {name} 插入fake通道, 通道数从{in_channel} 变为 {in_channel + t}!!!')
                        in_channel = in_channel + t
                        
                    #     
                    height = layers[name].inputs[0].height
                    width = layers[name].inputs[0].width
                    if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', ]:
                        kernel_size = layers[name].op.kernel
                        len_ = in_channel * width * kernel_size
                    elif layers[name].op.op_id in ['global_avg_pool2d']:
                        kernel_size = height
                        len_ = in_channel * width * kernel_size
                    elif layers[name].op.op_id in ['concat', 'fused_concat']:
                        len_ = in_channel * width * 4
                        # if name in ['Concat_208'] and CIMA_datawidth == 8:
                        #     len_ *= 4
                    else:
                        len_ = in_channel * width
                        if layers[name].op.op_id in ['add', 'fused_add'] and CIMA_datawidth == 8:
                            len_ *= 4
                        # if name in ['Add_117']:
                        #     len_ *= 4    
                    
                    # 避开紧邻的下一层以及上一层的卷积或者全连接的地址，选择离二者最近的中间地址
                    
                    nl = []
                    if name in next_layer_dict.keys():
                        nl = next_layer_dict[name]
                    pl = []
                    if name in pre_layer_dict.keys():
                        pl = pre_layer_dict[name]
                    relative_name = nl + pl
                    occupied_core = []
                    for n in relative_name:
                        addr_ = None
                        if n in self.node_mapping_info_list.keys():
                            addr_ = self.node_mapping_info_list[n]
                        elif n + '.0.0.0' in self.node_mapping_info_list.keys():
                            n_ = n + '.0.0.0'
                            if n_ in self.node_mapping_info_list.keys():
                                addr_ = self.node_mapping_info_list[n_]
                        elif n in  mapping_mesh_node.keys():
                            addr_ = mapping_mesh_node[n]
                            
                        if addr_ != None:
                            core_id = int(addr_.split('.')[1].split(':')[1])
                            core = (core_id//mesh_width,core_id%mesh_width)
                            if core not in occupied_core:
                                occupied_core.append(core)
                    
                    rest_possible_nodes = []
                    for x in all_points:
                        # index_rpn = x[0] * mesh_width + x[1]
                        # device_ref_rpn = f'{device_name}.cima-node:{index_rpn}'
                        # if device_ref_rpn in linebuf_assigned.keys():
                        #     mem_occupied_size = linebuf_assigned[device_ref_rpn][1]
                        # else:
                        #     mem_occupied_size = 0
                        if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', 'resize']:
                            occupied_core.extend(self.Mapped_MFOP_Core_Full)
                            
                        if x not in occupied_core:
                            rest_possible_nodes.append(x)
                    
                    if rest_possible_nodes == []:
                        raise ValueError(f'满足 mapping 要求的节点 {name} 所需内存空间不足 !!! ')
                        
                    try:
                        if occupied_core != []:    
                            closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=occupied_core)
                        else:
                            # 选择离 输入节点 (3,0) 最近的
                            closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=[(3,0)])
                    
                    except (np.AxisError) :
                        print(f'当前层 : {name}')
                        print(f'前相关层 : {pl}')
                        print(f'后相关层 : {nl}')
                        print(f'可以部署的 core : {rest_possible_nodes}')
                        print(f'不能部署的 core : {occupied_core}')
                        print(f'Allocate Non-PE Thread Error !!!')
                        exit(1)
                            
                    index_ = closest_point[0] * mesh_width + closest_point[1]

                    device_ref = f'{device_name}.cima-node:{index_}' 
                    current_node = device_ref
                    
                    # 判断当前mapped core中含有的mfop算子数量
                    
                    if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', 'resize']:
                        if device_ref not in self.Mapped_MFOP_Num.keys():
                            self.Mapped_MFOP_Num[device_ref] = 1
                        elif self.Mapped_MFOP_Num[device_ref] < self.Max_MFOP_Num:
                            self.Mapped_MFOP_Num[device_ref] += 1
                        # 
                        if self.Mapped_MFOP_Num[device_ref] == self.Max_MFOP_Num:
                            self.Mapped_MFOP_Core_Full.append(closest_point)
                    
                    # 实例化device mapping info
                    mapping_info = CIMADeviceMappingInfo(index = [0,0,0], device=device_ref, address=0)
                    if name not in self.node_mapping_info.keys():
                        self.node_mapping_info[name] = []
                    self.node_mapping_info[name].append(mapping_info)
                    
                    # 分配credit (按照pixel的个数分配)，分配一行
                    if name not in self.credit_len.keys():
                        self.credit_len[name] = []
                    
                    # credit 分配为图像的宽
                    self.credit_len[name].append( width)
                    # 分配linebuf 地址
                    if current_node not in linebuf_assigned.keys():
                        linebuf_assigned[current_node] = [self.Thread_Dmem_Base_Addr, self.Thread_Dmem_Base_Addr] # [起始地址，结束地址]
                    
                    # 分配地址空间，整个feature map
                    if name not in self.in_line_buffer_addr.keys():
                        self.in_line_buffer_addr[name] = []
                        
                    self.in_line_buffer_addr[name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
                    
                    # 更新assigned linebuf, 首地址 32byte 对齐
                    linebuf_assigned[current_node][1] += len_
                    while True:
                        if linebuf_assigned[current_node][1] % 32 == 0:
                            break
                        linebuf_assigned[current_node][1] += 1
                        
                    mapping_mesh_node[name] = current_node
                    
                elif layers[name].op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc', 
                                               'split', 'add', 'fused_add', 'fused_concat', 'concat', 'mul_add', 
                                               'silu', 'pad', 'relu', 'type_conversion']:
                    # 获取各层拆分信息
                    # 拆分之后的层名称
                    # nl_ = name + f'.0.{h1}.{w1}'
                    if layers[name].op.op_id in ['split', 'add', 'fused_add', 'fused_concat', 'concat']:
                        nl_ = name
                    else:
                        # 默认每一层只有一个地址
                        nl_ = name + f'.0.0.0'
                    
                    assert nl_ in self.node_mapping_info_list.keys()
                    addr = self.node_mapping_info_list[nl_]
                    # print(self.node_mapping_info_list[nl_])
                    
                    index_ = [0, 0, 0]
                    if name not in self.node_mapping_info.keys():
                        self.node_mapping_info[name] = []
                        
                    # 获取device_name
                    # device_ref = ".".join(addr.split('.')[:2])
                    
                    if layers[name].op.op_id in ['split', 'add', 'fused_add', 'fused_concat', 'concat']:
                        value = 0
                        device_ref = ".".join(addr.split('.')[:2])
                    else:
                        if 'cima-dmac' in addr:
                            value = 0
                            device_ref = addr
                        else:
                            device_ref = ".".join(addr.split('.')[:-1])
                            value_ = addr.split('.')[-1].split(' ')
                            value = []
                            for v in range(len(value_)):
                                if v == 0:
                                    value.append(int(value_[v].split('[')[1].split(',')[0]))
                                elif v == 3:
                                    value.append(int(value_[v].split(']')[0]))
                                else:
                                    value.append(int(value_[v].split(',')[0]))
                    
                    # 实例化device mapping info
                    mapping_info = CIMADeviceMappingInfo(index = index_, device=device_ref, address=value)
                    self.node_mapping_info[name].append(mapping_info)
                    # 当前节点名称
                    current_node = ".".join(addr.split('.')[:2])
                    
                    if current_node not in linebuf_assigned.keys():
                        linebuf_assigned[current_node] = [self.Thread_Dmem_Base_Addr, self.Thread_Dmem_Base_Addr] # [起始地址，结束地址]
                    # 分配credit (按照pixel的个数分配)，分配一行
                    if name not in self.credit_len.keys():
                        self.credit_len[name] = []
                        
                    # 分配line buffer 地址
                    node_info = self.node_info[name]
                    if node_info['in_channel'] == 255:
                        warnings.warn(f'当前层 {name} 插入fake通道, 通道数从{in_channel} 变为 256!!!')
                        node_info['in_channel'] += 1
                    if isinstance(node_info['in_channel'], int):    
                        if node_info['in_channel'] % 16 != 0:
                            t = 0
                            while (node_info['in_channel'] + t) % 16 != 0:
                                t += 1
                            warnings.warn(f"当前层 {name} 插入fake通道, 通道数从{node_info['in_channel']} 变为 {node_info['in_channel'] + t}!!!")
                            node_info['in_channel'] = node_info['in_channel'] + t
                    
                    if node_info['op_type'] in ['matmul', 'fused_fc']:
                        len_ = node_info['in_channel']
                        self.credit_len[name].append(1)
                    elif node_info['op_type'] in ['conv2d', 'fused_conv2d']:
                        # line buffer 分配 图像的宽 * 通道 * max(kernel, stride) 大小
                        len_ = node_info['input_shape'][1] * node_info['in_channel'] * max(node_info['kernel_size'], node_info['stride'])
                        # credit len 配置成 图像的宽
                        self.credit_len[name].append(node_info['input_shape'][1])
                    elif node_info['op_type'] in ['fused_concat', 'concat']:
                        len_ = node_info['in_channel'][0] * node_info['input_shape'][0][1] * len(layers[name].inputs)
                        if name in ['Concat_208', 'Concat_281', 'Concat_671']:
                            len_ *= 2
                        if name in ['Concat_390']:
                            len_ *= 4
                        self.credit_len[name].append(node_info['input_shape'][0][1])
                    elif node_info['op_type'] in ['split', 'add', 'fused_add', 'mul_add']:
                        len_ = node_info['in_channel'][0] * node_info['input_shape'][0][1]
                        if CIMA_datawidth == 8:
                            if node_info['op_type'] in ['add', 'fused_add']:
                                len_ *= 4
                                # if name in ['Add_56', 'Add_49', 'Conv_199_Add']:
                                #     len_ *= 2
                        # if name in ['Add_5', 'Add_10', 'Add_15', 'Add_20', 'Add_25', 'Add_31', '']
                        # len_ *= 4
                        # if name in ['Add_74']:
                        #     len_ *= 4
                        self.credit_len[name].append(node_info['input_shape'][0][1])
                    else:
                        raise ValueError(f"暂不支持 op_type: {node_info['op_type']}")
                    if name not in self.in_line_buffer_addr.keys():
                        self.in_line_buffer_addr[name] = []
                    # self.in_line_buffer_addr[name].append([linebuf_assigned[current_node][1], len_])
                    self.in_line_buffer_addr[name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
                    # 更新assigned linebuf, 32 byte 对齐
                    linebuf_assigned[current_node][1] += len_
                    while True:
                        if linebuf_assigned[current_node][1] % 32 == 0:
                            break
                        linebuf_assigned[current_node][1] += 1
                        
                    # 记录mapping_mesh_node
                    mapping_mesh_node[name] = current_node
                
                else:
                    raise ValueError(f'暂不支持的op: {layers[name].op.op_id}')
                
            count = 0
            
            # 需要插入DRAM 中转线程作为中转，
            # 算子的不同输入来源于 多个计算起始时间相隔很长的算子，此时由于芯片工作在流水的模式下，
            # 不同输入的数据产生速度不匹配，因此需要较大的缓存来存储中间结果.
            
            for layer_name in next_layer_dict.keys():
                if 'graph_input' in layer_name:
                    continue
                current_layer = self.ir.layers[layer_name]
                if current_layer.type == 'op' and current_layer.op.op_id in ['fused_concat', 'concat', 'fused_add', 'add']:
                    distance_thr = 1000 # 自定义
                    current_index_num = int(layer_name.split('_')[1])
                    for pl_ in pre_layer_dict[layer_name]:
                        nums = re.findall(r'\d+', pl_)
                        pre_index_num = int(nums[0])
                        # LastOpDistance.append(current_index_num - pre_index_num)
                        IsInsertDram = False
                        # if self.ir.layers[pl_].op.op_id == 'identity' and self.ir.layers[pl_].inputs[0].ref in ['Conv_161_Concat_0', 'Conv_85']:
                        #     IsInsertDram = True
                        # if pl_ in ['Conv_161_Concat_0', 'Conv_85', 'Conv_245_Concat_0', 'Conv_202_Add', 'Conv_161', 'Conv_245']:
                        # if pl_ in ['Concat_197', 'Concat_70']:
                        #     IsInsertDram = True
                        if (current_index_num - pre_index_num) > distance_thr or IsInsertDram:
                            # 当前算子的前一级节点
                            pl = pre_layer_dict[pl_]
                            # 当前算子的下一级节点
                            nl = [layer_name]
                            # 
                            pre_layer = self.ir.layers[pl_]
                            pre_node = int(mapping_mesh_node[pl_].split('.')[1].split(':')[1])
                            pre_node_coor = (pre_node//mesh_width, pre_node%mesh_width)
                            
                            # 添加新算子identity，开辟内存
                            all_possible_nodes = ['DDR']
                            count, linebuf_assigned = self.make_CIMA_transfer_thread(count, pre_layer, pl_, mapping_mesh_node, 
                                                                                    pre_node_coor, [pre_node_coor], mesh_width, 
                                                                                    pl, nl, linebuf_assigned, all_possible_nodes)
            # 更新 next_layer_dict 
            next_layer_dict = get_next_layer(self.ir.layers)
            
            # 添加 中转线程
            for layer_name in next_layer_dict.keys():
                
                if 'graph_input' in layer_name:
                    continue
                current_layer = self.ir.layers[layer_name]
                
                if current_layer.type == 'op' and current_layer.op.op_id in ['conv2d','fc','linear', 'matmul', 'fused_conv2d', 'fused_fc',
                                                                             'maxpool2d', 'avgpool2d', 'global_avg_pool2d']:
                    
                    # 获取当前层 mapping 的名称
                    mapped_name = layer_name + '.0.0.0'
                    # 获取当前层的 node 坐标 [x, y]
                    if current_layer.op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', ]:
                        current_node = int(mapping_mesh_node[layer_name].split('.')[1].split(':')[1])
                    else:
                        current_node = int(self.node_mapping_info_list[mapped_name].split('.')[1].split(':')[1])
                    current_node_coor = [current_node//mesh_width, current_node%mesh_width]
                    
                    # 获取前后相关层
                    nl = next_layer_dict[layer_name]
                    
                    pl = []
                    if layer_name in pre_layer_dict.keys():
                        pl = pre_layer_dict[layer_name]
                    
                    # 需要拆入 identity 的情况：
                    # 1. PE输出方向和路由规则的限制
                    # 2. Conv/FC输出不能直接多播
                    if current_layer.op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc'] and len(next_layer_dict[layer_name]) == 1:
                        
                        # 如果卷积或者全连接的下一层也是卷积或者全连接，
                        # 当前一层节点 往南传输数据，后一层节点必须位于其垂直下方；
                        # 当往北传输数据时，后一层节点必须位于其垂直上方。
                        # 否则需要插入identity算子，在邻近的上方（往北）或者下方（往南）节点中开辟新内存
                    
                        # 判断当前层的pe相对位置是否处于南或者北
                        pe_relative = self.node_mapping_info_list[mapped_name].split('.')[2]
                        # 0, 2分别为北和南
                        pe_number = int(pe_relative.split(':')[-1])
                        
                        # 如果当前节点位于最上或者最下，则输出往北和往南的情况，不需要再加中转线程
                        # 如果当前节点位于最右或者最左，则输出往东和往西的情况，不需要再加中转线程
                        
                        if current_node_coor[0] == 0 and pe_number == 0:
                            continue
                        if current_node_coor[0] == mesh_height - 1 and pe_number == 2:
                            continue
                        if current_node_coor[1] == mesh_width - 1 and pe_number == 1:
                            continue
                        if current_node_coor[1] == 0 and pe_number == 3:
                            continue
                        
                        if 'graph_output' in nl:
                            next_node = self.hosti_core_id[0] * mesh_width + self.hosti_core_id[1]
                        else:
                            next_node = int(mapping_mesh_node[nl[0]].split('.')[1].split(':')[1])
                            
                        next_node_coor = (next_node//mesh_width, next_node%mesh_width)
                        
                        if pe_number == 0:
                            # print(nl)
                            # 输出往北，但是下一节点不在该节点的正北方向
                            if not (next_node_coor[1] == current_node_coor[1] and next_node_coor[0] < current_node_coor[0]):
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[0]):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                # 添加新算子identity，开辟内存
                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, 
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl, 
                                                                                linebuf_assigned, all_possible_nodes)
                                
                        elif pe_number == 2:
                            # 输出往南，但是下一节点不在该节点的正南方向
                            if not (next_node_coor[1] == current_node_coor[1] and next_node_coor[0] > current_node_coor[0]):
                                all_possible_nodes = []
                                for i in range(current_node_coor[0]+1, mesh_height):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                
                                # 添加新算子identity，开辟内存
                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, 
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl, 
                                                                                linebuf_assigned, all_possible_nodes)
                            
                        elif pe_number == 1:
                            # 输出往东，但是下一节点不在该节点的东边（东北或者东南）
                            if next_node_coor[1] <= current_node_coor[1]:
                                all_possible_nodes = []
                                for i in range(current_node_coor[1]+1, mesh_width):
                                    if (current_node_coor[0], i) not in none_core:
                                        all_possible_nodes.append((current_node_coor[0], i))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                
                                # 添加新算子identity，开辟内存
                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, 
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl, 
                                                                                linebuf_assigned, all_possible_nodes)
                        
                        elif pe_number == 3:
                            # 输出往西，但是下一节点不在该节点的西边（西北或者西南）
                            if next_node_coor[1] >= current_node_coor[1]:
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[1]):
                                    if (current_node_coor[0], i) not in none_core:
                                        all_possible_nodes.append((current_node_coor[0], i))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core    
                                # 添加新算子identity，开辟内存
                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, 
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl, 
                                                                                linebuf_assigned, all_possible_nodes)
                        
                        else:
                            raise ValueError(f'pe 相对位置错误!!! 不应该出现 {pe_number}!!!')

                    else:
                        
                        # 如果下一层相邻节点大于1个，则也需要插入identity 进行多播
                        next_node_coor = []
                        for nl_ in nl:
                            
                            if 'graph_output' in nl_:
                                next_node_coor.append(self.hosti_core_id)
                            elif 'dram' in mapping_mesh_node[nl_]:
                                for n_ddr in ddr_core:
                                    next_node_coor.append(n_ddr)
                            else:
                                next_node = int(mapping_mesh_node[nl_].split('.')[1].split(':')[1])   
                                next_node_coor.append((next_node//mesh_width, next_node%mesh_width))
                            # if layer_name == 'Conv_165':
                            #     print(nl_)
                            #     print(mapping_mesh_node[nl_])
                            #     print(next_node_coor)
                            #     input()
                        # 中转线程所有可能的节点
                        all_possible_nodes = all_points + hosti_core + ddr_core
                        
                        # 下一相邻的层有多个不同的目的地址，则需要中转，如果是Conv/FC 则需要依据PE的方向进行中转，
                        if current_layer.op.op_id in ['conv2d','fc','linear', 'matmul', 'fused_conv2d', 'fused_fc']:
                            
                            # 判断当前层的pe相对位置是否处于南或者北
                            pe_relative = self.node_mapping_info_list[mapped_name].split('.')[2]
                            
                            pe_number = int(pe_relative.split(':')[-1])
                            # 0, 2分别为北和南, 则中转线程只能在北侧或者南侧
                            if pe_number == 0 and current_node_coor[0] != 0:
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[0]):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                    
                            elif pe_number == 2 and current_node_coor[0] != mesh_height - 1:
                                all_possible_nodes = []
                                for i in range(current_node_coor[0], mesh_height):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                    
                            # 1, 3 分别为东和西, 则中转线程只能在东侧或者西侧
                            elif pe_number == 1:
                                all_possible_nodes_new = []
                                for node in all_possible_nodes:
                                    if node[1] > current_node_coor[1] and node not in none_core:
                                        all_possible_nodes_new.append(node)
                                if all_possible_nodes_new == []:
                                    all_possible_nodes_new += ddr_core
                                
                                all_possible_nodes = all_possible_nodes_new
                                
                            elif pe_number == 3:
                                all_possible_nodes_new = []
                                for node in all_possible_nodes:
                                    if node[1] < current_node_coor[1] and node not in none_core: 
                                        all_possible_nodes_new.append(node)
                                if all_possible_nodes_new == []:
                                    all_possible_nodes_new += ddr_core
                                all_possible_nodes = all_possible_nodes_new
                         
                        # 添加新算子identity，开辟内存
                        count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node, 
                                                                                current_node_coor, next_node_coor, mesh_width, pl, nl, 
                                                                                linebuf_assigned, all_possible_nodes)
                
                elif current_layer.type == 'op' and current_layer.op.op_id in ['fused_concat', 'concat', 'fused_add', 'add']:
                    # concat 和 add 的算子有以下情况可能需要插入中转线程
                    # 算子有多个输入，不同输入可能和当前算子位置冲突，因此需要中转线程进行规避，此时是在concat / add 算子之前插入中转线程.
                    # 算子有多个输出，不同输出可能和当前算子位置冲突，因此需要中转线程进行规避，此时是在concat / add 算子之后插入中转线程.
                    
                    # 获取当前节点的位置
                    current_node = int(mapping_mesh_node[layer_name].split('.')[1].split(':')[1])
                    current_node_coor = [current_node//mesh_width, current_node%mesh_width]
                    
                    # 校验 情况1
                    for il in current_layer.inputs:
                        
                        # 中转线程所有可能的节点
                        all_possible_nodes = all_points + hosti_core + ddr_core
                        
                        # last_node_coor = []
                        ref_name = il.ref
                        last_layer_name = ref_name
                        if ':' in ref_name:
                            last_layer_name = ref_name.split(':')[0]
                        
                    
                        # 获取前一算子的core位置
                        last_node = int(mapping_mesh_node[last_layer_name].split('.')[1].split(':')[1])
                        last_node_coor = (last_node//mesh_width, last_node%mesh_width)
                        
                        if tuple(current_node_coor) == last_node_coor:
                            # input()
                            last_layer = self.ir.layers[last_layer_name]
                            
                            # 当前节点
                            nl = [layer_name]
                            
                            # 前一节点的相关输入节点
                            pl = []
                            for pln in last_layer.inputs:
                                pl.append(pln.ref)
                            
                            # 移除掉当前的节点
                            if last_node_coor in all_possible_nodes:    
                                all_possible_nodes.remove(last_node_coor)
                            
                            # 添加新算子identity，开辟内存
                            count, linebuf_assigned = self.make_CIMA_transfer_thread(count, last_layer, ref_name, mapping_mesh_node, 
                                                                                    last_node_coor, [current_node_coor], mesh_width, 
                                                                                    pl, nl, linebuf_assigned, all_possible_nodes)    
                            
            # 将layer排序
            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
            
        else:
            raise ValueError(f'暂不支持设备 {self.hd_name[0]}的mapping!!!')
    
    def make_CIMA_transfer_thread(self, count, current_layer, layer_name, mapping_mesh_node, current_node_coor, next_node_coor,
                                        mesh_width, pl, nl, linebuf_assigned, all_possible_nodes):
        
        # 添加新算子identity，开辟内存
        identity_name = f'identity_{count}'
        count += 1
        input_shape = current_layer.outputs[0]
        
        # 获取当前层信息
        in_channel = input_shape.channel
        if in_channel == 255:
            warnings.warn(f'当前层 {identity_name} 插入fake通道, 通道数从{in_channel} 变为 256!!!')
            in_channel += 1
        if in_channel % 32 != 0:
            t = 0
            while (in_channel + t) % 32 != 0:
                t += 1
            warnings.warn(f'当前层 {identity_name} 插入fake通道, 通道数从{in_channel} 变为 {in_channel + t}!!!')
            in_channel = in_channel + t
            
        height = input_shape.height
        width = input_shape.width
        len_ = math.ceil(in_channel * width)
                
        # 上一层有多个输出时，需要判断当前identity属于上一层的哪一个输出
        self.make_identity_op(input_shape, layer_name, identity_name)
        
        # 添加 mapping 信息
        device_name = mapping_mesh_node[layer_name].split('.')[0]
        
        if all_possible_nodes == ['DDR']:
            # DDR 中转线程
            closest_point = self.ddr_core_id
            device_ref = f'{device_name}.cima-dram'
            
            index_ = closest_point[0] * mesh_width + closest_point[1]
            current_node = f'{device_name}.cima-node:{index_}'

            # DDRI 线程需要分配两倍的中转缓存，读缓存和写缓存分开
            len_ *= 2
            
        else:
            # all_possible_nodes = []
            # for i in range(0, current_node_coor[0]):
            #     all_possible_nodes.append((i, current_node_coor[1]))
            rest_possible_nodes = []
            
            for x in all_possible_nodes:
                index_rpn = x[0] * mesh_width + x[1]
                device_ref_rpn = f'{device_name}.cima-node:{index_rpn}'
                
                if device_ref_rpn in linebuf_assigned.keys():
                    mem_occupied_size = linebuf_assigned[device_ref_rpn][1]
                else:
                    mem_occupied_size = 0
                        
                if list(x) != current_node_coor and x not in next_node_coor: 
                    rest_possible_nodes.append(x)
                    
                if mem_occupied_size > self.Max_Memory_Size:
                    warnings.warn(f' 若算子 {identity_name} mapping 在 Core{x}, 则内存空间将超出限制, 内存空间大小为 {hex(mem_occupied_size)} !!!')
                    
            # if rest_possible_nodes == []:
            #     raise ValueError(f'mapping 算子 {identity_name} 所需内存空间不足 !!!')  
                
            # occupied_core = [tuple(current_node_coor), tuple(next_node_coor)] 
            occupied_core = [tuple(current_node_coor)] + next_node_coor
            
            try:
                if occupied_core != []:    
                    closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=occupied_core)
                else:
                    # 选择离 输入节点 (3,0) 最近的
                    closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=[self.hosti_core_id])
            
            except (np.AxisError) :
                print(f'当前层 : {layer_name}, 位置 {current_node_coor}')
                print(f'前相关层 : {pl}')
                print(f'后相关层 : {nl}, 位置 {next_node_coor}')
                print(f'可以部署的 core : {rest_possible_nodes}')
                print(f'不能部署的 core : {occupied_core}')
                print(f'Make Transfer Thread Error !!!')
                exit(1)
        
            index_ = closest_point[0] * mesh_width + closest_point[1]
            device_ref = f'{device_name}.cima-node:{index_}'
            current_node = device_ref
            
        if current_node not in linebuf_assigned.keys():
            linebuf_assigned[current_node] = [self.Thread_Dmem_Base_Addr, self.Thread_Dmem_Base_Addr] # [起始地址，结束地址]
            
        # 实例化device mapping info
        mapping_info = CIMADeviceMappingInfo(index = [0,0,0], device=device_ref, address=0)
        if identity_name not in self.node_mapping_info.keys():
            self.node_mapping_info[identity_name] = []
        self.node_mapping_info[identity_name].append(mapping_info)
        
        # 记录mapping_mesh_node
        mapping_mesh_node[identity_name] = current_node
        
        # 分配credit (按照pixel的个数分配)，分配一行
        if identity_name not in self.credit_len.keys():
            self.credit_len[identity_name] = []
        
        
        if current_layer.op.op_id in ['matmul', 'fused_fc']:
            self.credit_len[identity_name].append(1)
        else:
            self.credit_len[identity_name].append(width)
            
        # 分配地址空间，整个feature map
        if identity_name not in self.in_line_buffer_addr.keys():
            self.in_line_buffer_addr[identity_name] = []
        # self.in_line_buffer_addr[identity_name].append([linebuf_assigned[current_node][1], len_])
        self.in_line_buffer_addr[identity_name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
        # 更新assigned linebuf, 32Byte 对齐
        linebuf_assigned[current_node][1] += len_
        while True:
            if linebuf_assigned[current_node][1] % 32 == 0:
                break
            linebuf_assigned[current_node][1] += 1
        
        # 更改相应其他算子的ref
        for nl_ in nl:
            layer_inputs = self.ir.layers[nl_].inputs
            for li in layer_inputs:
                if li.ref == layer_name:
                    li.ref = identity_name
        
        return count, linebuf_assigned

    def find_closest_point(self, points, linebuf_assigned, data_len, mesh_width=6,  exclude_points=[]):
        points = np.array(points)
        exclude_points = np.array(exclude_points)
        
        # 计算与给定点集的距离矩阵
        try:
            distances = cdist(points, exclude_points, metric='cityblock')
        except ValueError:
            print(points)
            print(exclude_points)
            raise ValueError(f'Error!!!')
        distances = np.sum(distances,axis=1)
        
        # 找到曼哈顿距离最小的点对应的index
        # min_distance_index = np.argmin(distances) if distances is not None else None
        
        sorted_index_list = sorted(range(len(list(distances))), key=lambda k:distances[k])
        
        # index = np.where(distances == distances[min_distance_index])
        
        # 在所有的最小距离的core id 中，根据当前core内存的占用情况，找到内存占用最小的core id, 并且中转线程数目小于最大中转线程数
        if len(linebuf_assigned.keys()) != 0:
            
            device_name = list(linebuf_assigned.keys())[0].split('.')[0]
            
            mem_size = []
            
            for p_id in sorted_index_list:
                
                core_id = points[p_id]
                index_ = core_id[0] * mesh_width + core_id[1]
                device_ref = f'{device_name}.cima-node:{index_}'
                
                # 判断当前core内的中转线程数量，如果大于最大中转线程数目则跳过
                if device_ref in self.transfer_thread_num.keys() and self.transfer_thread_num[device_ref] >= self.Max_Transfer_Thread_Num:
                    mem_size.append(0x100000)
                    continue
                
                # 判断内存使用情况
                # if device_ref in linebuf_assigned.keys():
                #     if (linebuf_assigned[device_ref][1] + data_len) <= self.Max_Memory_Size:
                #         min_mem_size_id = p_id
                #         CanMap = True
                #         break
                # else:
                #     min_mem_size_id = p_id
                #     CanMap = True
                #     break

                if device_ref in linebuf_assigned.keys():
                    mem_size.append(linebuf_assigned[device_ref][1])
                else:
                    mem_size.append(0)
            
            # 对mem size进行排序
            sorted_mem_size_id = sorted(range(len(list(mem_size))), key=lambda k:mem_size[k])
            
            CanMap = False
            for id in sorted_mem_size_id:
                
                min_mem_size_id = sorted_index_list[id]
                # 确定core name
                core_id = points[min_mem_size_id]
                index_ = core_id[0] * mesh_width + core_id[1]
                device_ref = f'{device_name}.cima-node:{index_}'
                # device reference
                if device_ref in linebuf_assigned.keys():
                    if (linebuf_assigned[device_ref][1] + data_len) <= self.Max_Memory_Size:
                        CanMap = True
                        break
                else:
                    CanMap = True
                    break
            # 
            if not CanMap:
                for p in points:
                    index_ = p[0] * mesh_width + p[1]
                    core_name = f'{device_name}.cima-node:{index_}'
                    print(core_name)
                # print(points)
                # print(exclude_points)
                print(self.transfer_thread_num)
                raise ValueError(f'所有可能的Core都无法部署当前算子, 算子所需要的数据空间为：{data_len}!!!')
            
            if device_ref not in self.transfer_thread_num.keys():
                self.transfer_thread_num[device_ref] = 1
            else:
                self.transfer_thread_num[device_ref] += 1
            
            # min_mem_size_id = sorted_index_list[np.argmin(np.array(mem_size))]
        else:
            
            min_mem_size_id = sorted_index_list[0]    
                
            
        # 返回结果
        closest_point = tuple(points[min_mem_size_id]) if min_mem_size_id is not None else None
        
        return closest_point
    
    def Is_CIMA_mapped_layer(self, layer):
        if layer.type == 'op' and layer.op.op_id in ['conv2d','fc','linear','matmul','fused_conv2d','fused_fc']:
            return True
        return False
    
    def make_identity_op(self, input_shape, ref_layer_name, identity_name):
        # from e100_irtool import make_op
        op_ = make_op('identity')
        inputs_ = [dict(ref=ref_layer_name,channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
        outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
        self.ir.add_layer(identity_name,op=op_,inputs=inputs_,outputs=outputs_)
    
    def update_info(self):
        '''
        按照 split_num 更新当前info信息，将节点名称替换
        return:
            新的node_info,字典形式
        '''
        node_info = {}
        out_loop = 0
        
        for i in self.node_info.keys():
            if self.window_copy:
                [p,r,w,h] = self.split_num[i]
                calc_num = math.ceil(self.node_info[i]['calc_num'] / (p*r))
                out_loop = p
            else:
                [r,w,h] = self.split_num[i]
                calc_num = math.ceil(self.node_info[i]['calc_num'] / r)
                out_loop = r
            for j in range(out_loop):
                for k in range(h):
                    for l in range(w):
                        if self.window_copy:
                            new_name = i+'.'+str(j)+'.'+str(k)+'.'+str(l)+'_wd'
                        else:
                            new_name = i+'.'+str(j)+'.'+str(k)+'.'+str(l)
                        shape = self.split_node_weight[new_name]
                        in_pre = self.node_info[i]['in_precision']
                        out_pre = self.node_info[i]['out_precision']
                        node_info[new_name] = dict(shape=shape,calc_num=calc_num,in_precision=in_pre,out_precision=out_pre)
        return node_info
    
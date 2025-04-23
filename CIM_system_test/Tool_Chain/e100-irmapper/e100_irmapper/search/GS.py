from ..helper import *
from ..placement import *
from ..esti_model import *
from .Base import Base

class GreedySearch(Base):
    
    def __init__(self,node_info,node_weight,hardware_config, average_copy=None, specify_para_num=None, 
                 window_copy=False,try_time=10, place_strategy=OneOnOne, evaluate_model=NumericialEstimation):
        '''
        try_time: 超参数，INT，用于探索的次数上限
        evaluate_model: 评估策略，类型是 ‘类’，默认为 NumericialEstimation
        '''
        self.try_time = try_time
        self.evaluate_model =  evaluate_model
        super().__init__(node_info,node_weight,hardware_config,average_copy=average_copy, specify_para_num=specify_para_num ,place_strategy=place_strategy,window_copy=window_copy)
        
    def run(self):
        '''
        return:
            返回各层的mapping_info，字典形式，{'node_name':[{'index':[r,h_num,w_num],'device_ref':STR,'device_addr':[h_start,w_start,h,w]},...],...}
            index 为三元素的列表形式用于描述复制和拆分之后的不同的块的位置，索引的顺序为[r][h][w]，例如[1,2,3]表示复制的第一份的权重中切分之后的行方向的第二和
            列方向第三 的权重块，device_ref 指向的硬件即为该权重块的运算设备，device_addr 即为位于该XB的物理位置。
        '''
        
        # 获取硬件信息
        self.get_hardware_info()
        
        # 自适应切分权重 node_name -> [r,w,h]
        self.split_average()
        
        # 按照给定的策略进行排布
        self.placed_nodes = self.place_strategy(self.split_node_weight,self.XB_size).run()
        rest_xb = self.XB_num - len(self.placed_nodes)
        
        if rest_xb >= 0:
            new_node_info = self.update_info()
        else:
            raise ValueError('按照当前策略无法放下！')
        # 初始化参数
        pre_try_max_time = 10**(8)
        try_time = 0
        # 根据适应性函数更新 split_num
        while True:
            split_num = copy.deepcopy(self.split_num)
            
            # 判断当前评估模型的方式
            eva_model = self.evaluate_model(new_node_info)
            eva_model_name = eva_model.__class__.__name__
            if eva_model_name == 'NumericialEstimation':
                layer_time = eva_model.run(self.dac_num,self.adc_num,self.dac_precision)
            elif eva_model_name == 'HARNSEvaluation':
                layer_time = eva_model.run()
            elif eva_model_name == 'IDEALEvaluation':
                layer_time = eva_model.run()
            else:
                raise ValueError(f'NOT IMPLEMENTED {eva_model_name}!!!')
            
            current_max_node = get_max_time_layer(layer_time)
            current_max_time = list(current_max_node.values())[0]
            
            # 拆分之后的名称 --> 原始模型层的名称
            max_node_name = list(current_max_node.keys())[0].split('.')[0]
            
            self.update_split_num(max_node_name)
            if self.window_copy:
                self.split_node_weight,self.split_num = split_node_window_duplicate(self.node_info,self.XB_size,self.split_num)
            else:
                self.split_node_weight = split_node(self.node_weight,self.split_num)
            
            # 按照给定的策略进行排布
            self.placed_nodes = self.place_strategy(self.split_node_weight,self.XB_size).run()
            
            rest_xb = self.XB_num - len(self.placed_nodes)
            if rest_xb >= 0 and try_time < self.try_time:
                if pre_try_max_time <= current_max_time:
                    try_time += 1
                else:
                    try_time = 0
                # 记录之前的最大时间
                pre_try_max_time = copy.deepcopy(current_max_time)
                # 更新节点信息
                new_node_info = self.update_info()
                continue
            else:
                # 回溯到刚好放下的拆分方式作为最后的结果
                if self.window_copy:
                    self.split_node_weight,self.split_num = split_node_window_duplicate(self.node_info,self.XB_size,split_num)
                else:
                    self.split_node_weight = split_node(self.node_weight,split_num)
                self.placed_nodes = self.place_strategy(self.split_node_weight,self.XB_size).run()
                break
                
        self.split_num = split_num

        # 给每一个阵列赋上device名称
        self.ref_to_device()
        
    def update_split_num(self,node_name):
        '''
        根据时间最长的层，更新 split_num
        '''
        if self.window_copy:
            assert len(self.split_num[node_name]) == 4
            cc = self.node_info[node_name]['copy_constraint']
            para = self.split_num[node_name][0]
            spr = self.split_num[node_name][1]
            _w,_h = self.split_num[node_name][2],self.split_num[node_name][3]
            if spr < cc :
                # repeat 次数小于 constraint, 复制的次数必须是 cc的因子
                spr += 1
                while cc % spr != 0 :
                    spr += 1
            else:
                para += 1
            
            # 每个层不能同时放在不同的tile中
            ub = para * _w * _h
            if ub > 8:
                para = para - 1
          
            self.split_num[node_name] = [para,spr,_w,_h]
        else:
            self.split_num[node_name][0] += 1
        
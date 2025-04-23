from ..helper import *
from ..placement import *
from ..esti_model import *
from .Base import Base

class GreedySearch(Base):

    def __init__(self, node_info, node_weight, hardware_config, average_copy=None, specify_para_num=None, window_copy=False, try_time=10, place_strategy=OneOnOne, evaluate_model=NumericialEstimation):
        
        self.try_time = try_time
        self.evaluate_model = evaluate_model
        super().__init__(node_info, node_weight, hardware_config, average_copy=average_copy, specify_para_num=specify_para_num, place_strategy=place_strategy, window_copy=window_copy)

    def run(self):
        
        self.get_hardware_info()
        self.split_average()
        self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()
        rest_xb = self.XB_num - len(self.placed_nodes)
        if rest_xb >= 0:
            new_node_info = self.update_info()
        else:
            raise ValueError('按照当前策略无法放下！')
        pre_try_max_time = 10 ** 8
        try_time = 0
        while True:
            split_num = copy.deepcopy(self.split_num)
            eva_model = self.evaluate_model(new_node_info)
            eva_model_name = eva_model.__class__.__name__
            if eva_model_name == 'NumericialEstimation':
                layer_time = eva_model.run(self.dac_num, self.adc_num, self.dac_precision)
            elif eva_model_name == 'HARNSEvaluation':
                layer_time = eva_model.run()
            elif eva_model_name == 'IDEALEvaluation':
                layer_time = eva_model.run()
            else:
                raise ValueError(f'NOT IMPLEMENTED {eva_model_name}!!!')
            current_max_node = get_max_time_layer(layer_time)
            current_max_time = list(current_max_node.values())[0]
            max_node_name = list(current_max_node.keys())[0].split('.')[0]
            self.update_split_num(max_node_name)
            if self.window_copy:
                (self.split_node_weight, self.split_num) = split_node_window_duplicate(self.node_info, self.XB_size, self.split_num)
            else:
                self.split_node_weight = split_node(self.node_weight, self.split_num)
            self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()
            rest_xb = self.XB_num - len(self.placed_nodes)
            if rest_xb >= 0 and try_time < self.try_time:
                if pre_try_max_time <= current_max_time:
                    try_time += 1
                else:
                    try_time = 0
                pre_try_max_time = copy.deepcopy(current_max_time)
                new_node_info = self.update_info()
                continue
            else:
                if self.window_copy:
                    (self.split_node_weight, self.split_num) = split_node_window_duplicate(self.node_info, self.XB_size, split_num)
                else:
                    self.split_node_weight = split_node(self.node_weight, split_num)
                self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()
                break
        self.split_num = split_num
        self.ref_to_device()

    def update_split_num(self, node_name):
        
        if self.window_copy:
            assert len(self.split_num[node_name]) == 4
            cc = self.node_info[node_name]['copy_constraint']
            para = self.split_num[node_name][0]
            spr = self.split_num[node_name][1]
            (_w, _h) = (self.split_num[node_name][2], self.split_num[node_name][3])
            if spr < cc:
                spr += 1
                while cc % spr != 0:
                    spr += 1
            else:
                para += 1
            ub = para * _w * _h
            if ub > 8:
                para = para - 1
            self.split_num[node_name] = [para, spr, _w, _h]
        else:
            self.split_num[node_name][0] += 1
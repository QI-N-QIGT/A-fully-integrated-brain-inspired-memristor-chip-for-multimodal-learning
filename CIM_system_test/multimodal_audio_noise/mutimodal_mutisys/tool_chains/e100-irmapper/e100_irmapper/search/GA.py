from ..helper import *
from ..placement import *
from ..esti_model import *
from .Base import Base
from sko.GA import GA
import numpy

class GeneticAlgorithm(Base):

    def __init__(self, node_info, node_weight, hardware_config, lb=None, ub=None, precision=None, window_copy=False, average_copy=None, specify_para_num=None, place_strategy=OneOnOne, evaluate_model=NumericialEstimation, size_pop=100, max_iter=500):
        
        self.evaluate_model = evaluate_model
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.pre = precision
        super().__init__(node_info, node_weight, hardware_config, average_copy=average_copy, specify_para_num=specify_para_num, place_strategy=place_strategy, window_copy=window_copy)

    def run(self):
        
        self.conv_node = []
        for i in self.node_weight.keys():
            if self.node_info[i]['op_type'] in ['conv2d', 'conv_transpose2d']:
                self.conv_node.append(i)
        if self.lb != None:
            self.dim = len(self.lb)
        else:
            raise ValueError('lb 参数传入错误!!!')
        ga = GA(func=self.func, size_pop=self.size_pop, n_dim=self.dim, max_iter=self.max_iter, lb=self.lb, ub=self.ub, constraint_ueq=[self.ueq_constraint1], precision=self.pre)
        (best_x, best_y) = ga.run()
        if best_y[0] != 10 ** 8:
            name_list = list(self.conv_node)
            for i in range(len(name_list)):
                node_name = name_list[i]
                assert best_x[i] == self.split_num[node_name][0]
            self.ref_to_device()
        else:
            raise ValueError('Do not find proper results, please increse the size pop or max iteration!!!')

    def func(self, x):
        
        rest_xb = -self.ueq_constraint1(x)
        ueq_2 = self.ueq_constraint2(x)
        if rest_xb < 0 or any(ueq_2 > 0):
            return 10 ** 8
        else:
            new_node_info = self.update_info()
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
            max_time = list(current_max_node.values())[0]
            return max_time

    def ueq_constraint1(self, x):
        
        self.get_hardware_info()
        self.split_average()
        if self.window_copy:
            assert len(x) == 2 * len(self.conv_node)
        else:
            assert len(x) == len(self.conv_node)
        t = len(self.conv_node)
        for i in range(len(self.conv_node)):
            node_name = self.conv_node[i]
            if self.window_copy:
                (p, r, w, h) = self.split_num[node_name]
                self.split_num[node_name] = [int(x[i]), int(x[i + t]), w, h]
            else:
                (r, w, h) = self.split_num[node_name]
                self.split_num[node_name] = [int(x[i]), w, h]
        if self.window_copy:
            (self.split_node_weight, self.split_num) = split_node_window_duplicate(self.node_info, self.XB_size, self.split_num)
        else:
            self.split_node_weight = split_node(self.node_weight, self.split_num)
        self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()
        rest_xb = self.XB_num - len(self.placed_nodes)
        return -rest_xb

    def ueq_constraint2(self, x):
        
        if self.window_copy:
            assert len(x) == 2 * len(self.conv_node)
        else:
            assert len(x) == len(self.conv_node)
        t = len(self.conv_node)
        y = numpy.zeros(t)
        for i in range(len(self.conv_node)):
            node_name = self.conv_node[i]
            if self.window_copy:
                (p, r, w, h) = self.split_num[node_name]
                self.split_num[node_name] = [int(x[i]), int(x[i + t]), w, h]
            else:
                (r, w, h) = self.split_num[node_name]
                self.split_num[node_name] = [int(x[i]), w, h]
            op_type = self.node_info[node_name]['op_type']
            if op_type in ['matmul', 'linear']:
                y[i] = int(x[i]) * int(x[i + t]) * w * h - 8
            elif op_type in ['conv2d', 'conv_transpose2d']:
                y[i] = int(x[i]) * w * h - 8
            else:
                raise ValueError(f'算子类型错误{op_type}')
        return y
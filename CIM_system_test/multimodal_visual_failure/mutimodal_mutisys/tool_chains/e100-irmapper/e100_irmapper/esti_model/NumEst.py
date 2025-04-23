import math

class NumericialEstimation(object):

    def __init__(self, node_info):
        
        self.node_info = node_info

    def run(self, dac_num, adc_num, dac_precision, XB_time=30, transmission_clk=10, transmission_bwidth=64):
        
        self.XB_time = XB_time
        self.t_clk = transmission_clk
        self.t_bw = transmission_bwidth
        self.adc_num = adc_num
        self.dac_num = dac_num
        self.dac_precision = dac_precision
        layer_time = {}
        for node_name in self.node_info.keys():
            data_num = self.node_info[node_name]['shape'][1] * self.node_info[node_name]['calc_num']
            t_trans = math.ceil(data_num * self.node_info[node_name]['in_precision'] / self.t_bw) * self.t_clk
            h_num = math.ceil(self.node_info[node_name]['shape'][1] / self.dac_num) * math.ceil(self.node_info[node_name]['in_precision'] / self.dac_precision)
            w_num = math.ceil(self.node_info[node_name]['shape'][0] / self.dac_num)
            t_calc = h_num * w_num * self.node_info[node_name]['calc_num'] * self.XB_time
            layer_time[node_name] = t_trans + t_calc
        return layer_time
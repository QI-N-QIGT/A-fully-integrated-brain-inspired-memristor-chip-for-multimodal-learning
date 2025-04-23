import math

class NumericialEstimation(object):    
    def __init__(self,node_info):
        '''
        node_info: 字典形式，涵盖节点信息，包括所有mapping在XB上的节点的基本信息
                   {'node_name':{'shape':[w,h],'calc_num':INT,'in_precision':INT,'out_precision':INT},...}
        
        '''
        self.node_info = node_info
        
    def run(self,dac_num,adc_num,dac_precision,XB_time=30,transmission_clk=10,transmission_bwidth=64):
        '''
        估计的延迟时间分为两部分，传输时间和计算时间
        input:
            dac_num: 单个XB的dac的数量
            adc_num: 单个XB的adc的数量
            XB_time: 单次XB的计算时间，单位为ns，默认值为30ns
            transmission_rate：XB数据传输周期，单位为ns，默认值为10ns（100Mhz）
            transmission_bwidth：XB数据传输带宽，单位为bit，默认为64bit
        '''
        self.XB_time = XB_time
        self.t_clk = transmission_clk
        self.t_bw = transmission_bwidth
        self.adc_num = adc_num
        self.dac_num = dac_num
        self.dac_precision = dac_precision
        # 记录各层的计算时间
        layer_time = {}
        for node_name in self.node_info.keys():
            # print(self.node_info[node_name])
            # 数据传输时间
            data_num = self.node_info[node_name]['shape'][1] * self.node_info[node_name]['calc_num']
            t_trans = math.ceil(data_num * self.node_info[node_name]['in_precision'] / self.t_bw) * self.t_clk
            
            # 计算时间
            h_num = math.ceil(self.node_info[node_name]['shape'][1] / self.dac_num) * math.ceil(self.node_info[node_name]['in_precision'] / self.dac_precision)
            w_num = math.ceil(self.node_info[node_name]['shape'][0] / self.dac_num)
            t_calc = h_num * w_num * self.node_info[node_name]['calc_num'] * self.XB_time
            
            layer_time[node_name] = t_trans + t_calc

        return layer_time

        
from e100_irmapper.self_defined_layer.a111_layer import *

# 声明A111 Pipeline graph
a111_pipe_graph = A111PipeGraph()

# 声明算法网络各层
InLayer = A111InputLayer('Conv_0', input_image_size=[32, 32])
Conv_0 = A111ConvLayer('Conv_0', tile_id=0, xb_id_list=[0, 1], in_channel=64, out_channel= 64, 
                       adc_range= 1, relu=True, shift_num=1, kernel_size=3, stride = 2, output_pad=[1, 0, 1, 0])
Conv_1 = A111ConvLayer('Conv_1', tile_id=0, xb_id_list=[2, 3], in_channel=64, out_channel= 64, 
                       adc_range= 1, relu=True, shift_num=1, kernel_size=3, stride = 2, output_pad=[1, 0, 1, 0])
Conv_2 = A111ConvLayer('Conv_2', tile_id=1, xb_id_list=[0, 1], in_channel=64, out_channel= 128, 
                       adc_range= 1, relu=True, shift_num=1, kernel_size=3, stride = 2, avgpool = True, output_pad=[0, 0, 0, 0])
OutLayer = A111OutputLayer('Conv_2')

# 根据自定义顺序将layer加入到graph中，一次加入一层，加入的顺序即为层连接的顺序
# a111_pipe_graph.add_layer(InLayer)
# a111_pipe_graph.add_layer(Conv_0)
# a111_pipe_graph.add_layer(Conv_1)
# a111_pipe_graph.add_layer(Conv_2)
# a111_pipe_graph.add_layer(OutLayer)

# 或者使用下面的实现
a111_pipe_graph.add_layer_list([InLayer, Conv_0, Conv_1, Conv_2, OutLayer])

# 将pipe_graph 转换为 ir，并生成yaml文件
ir = a111_pipe_graph.to_ir() # ir 可以直接传给 multi_layer 函数
ir.dump_json(file='test\\a111\\self_define\\test_3_layer_3.yaml')



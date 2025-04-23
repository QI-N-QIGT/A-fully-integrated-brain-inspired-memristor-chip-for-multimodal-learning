from e100_irmapper.self_define.CIMA_layer import *
InLayer = CIMAInputLayer('graph_input', input_image_size=[110, 110])
Identity_0 = CIMAIdentityLayer('Identity_0', core_id=[1, 0], in_channel=64, out_channel=64, credit_len=110)
Conv_0 = CIMAConvLayer('Conv_0', core_id=[0, 0], pe_cluster_id=0, pe_xb_id=0, relu=True, in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, credit_len=110)
Conv_1 = CIMAConvLayer('Conv_1', core_id=[0, 0], pe_cluster_id=1, pe_xb_id=0, relu=True, in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, credit_len=110)
Conv_3 = CIMAConvLayer('Conv_3', core_id=[0, 0], pe_cluster_id=3, pe_xb_id=0, relu=True, in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, credit_len=110)
Concat_0 = CIMAConcatLayer('Concat_0', core_id=[0, 1], in_channel=[64, 64, 64], out_channel=[64, 64, 64], split=[64, 64, 64])
Conv_6 = CIMAConvLayer('Conv_6', core_id=[1, 0], pe_cluster_id=2, pe_xb_id=0, relu=True, in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, credit_len=110)
Conv_7 = CIMAConvLayer('Conv_7', core_id=[1, 0], pe_cluster_id=3, pe_xb_id=0, relu=True, in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, credit_len=110)
Add_0 = CIMAAddLayer('Add_0', core_id=[2, 0], in_channel=[64, 64, 64], out_channel=64)
OutLayer = CIMAOutputLayer('graph_output')
CIMA_pipe_graph = CIMAPipeGraph(layer_info_list=[InLayer, Identity_0, Conv_0, Conv_1, Conv_3, Concat_0, Conv_6, Conv_7, Add_0, OutLayer], layer_graph={'graph_input': ['Identity_0'], 'Identity_0': ['Conv_0', 'Conv_1', 'Conv_3'], 'Conv_0': ['Concat_0'], 'Conv_1': ['Concat_0'], 'Conv_3': ['Concat_0'], 'Concat_0': ['Conv_6', 'Conv_7'], 'Conv_6': ['Add_0'], 'Conv_7': ['Add_0'], 'Add_0': ['graph_output']})
path = f'test\\CIMA\\self_define\\'
ir = CIMA_pipe_graph.to_ir()
ir.dump_json(file=path + 'test_5_layers.yaml')
config_path = path + f'5_layers.json'
CIMA_pipe_graph.to_systemc_config(config_path=config_path)
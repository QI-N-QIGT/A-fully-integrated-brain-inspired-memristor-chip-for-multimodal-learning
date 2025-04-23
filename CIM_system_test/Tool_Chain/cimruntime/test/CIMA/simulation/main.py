from cimruntime.CIMA.simulation.utils import *
import torch
from matplotlib import pyplot as plt

from cimruntime.c200.utils import feature_map_to_input, output_to_feature_map

if __name__ == "__main__":
    
    # load attr
    backbone_Conv_3 = torch.load(f'data\\hard_params_dict_bypass.pth')['backbone.Conv_3']
    backbone_Conv_3_bn = torch.load(f'data\\hard_params_dict.pth')['backbone.Conv_3_bn']['info0']
    backbone_Conv_3_bn.update(torch.load(f'data\\hard_params_dict.pth')['backbone.Conv_3_bn']['info1'])
    
    # load feature
    feature_Conv_3 = torch.load(f'data\\layer_feas_dict_bypass.pth')['backbone.Conv_3']
    input_data = feature_Conv_3['input_int'].detach().cpu().numpy()[0]
    output_data = feature_Conv_3['output_int'].detach().cpu().numpy()[0]
    
    feature_Conv_3_bn = torch.load(f'data\\layer_feas_dict.pth')['backbone.Conv_3_bn']
    output_data_bn = feature_Conv_3_bn['output_int'].detach().cpu().numpy()[0]
    
    # load weight
    weight_data = torch.load(f'data\\hard_params_dict.pth')['backbone.Conv_3']['info1']['weight_int'].detach().cpu().numpy()
    
    # bn attribute
    bn_scale = backbone_Conv_3_bn['scale_mul'].numpy()
    bn_offset = backbone_Conv_3_bn['offset'].numpy()
    bn_scale_shift = backbone_Conv_3_bn['scale_shift']
    
    # check scale
    total_scale = backbone_Conv_3['soft_scale']
    accumulate_shift_num = 0
    print(accumulate_shift_num)
    # hardware scale
    hardware_in_scale = 0.0957 / 7
    hardware_w_scale = 36 / 127
    c = 0
    
    # conv attribute
    kernel_size = 3
    stride = 2
    padding = 1
    out_h = output_data.shape[1]
    out_w = output_data.shape[2]
        
    # array_input
    array_input = feature_map_to_input(input_data, kernel_size, stride, padding, multi_batch=False)
    array_input = array_input.transpose(1, 0).astype(np.int32)
    
    plt.figure(figsize=(20, 10))
    for o_r in [32, 40, 64, 80, 120, 160, 200]:
        
        plt.subplot(3, 3, c+1)
        
        # get w_int
        hardware_out_scale = 127 / o_r
        w_int = round(total_scale / (hardware_in_scale * hardware_w_scale * hardware_out_scale))
        
        # array weight
        array_weight = weight_data.reshape(-1, 576).transpose(1,0).astype(np.int32) * w_int

        # conductance noise
        cn = 0
        
        # array output 
        array_output = CIMA_analog_MAC(array_input, array_weight, ADC_qunat_level=c, conductance_noise=cn,
                                       accumulate_shift_num=accumulate_shift_num, 
                                    #    scale=bn_scale, offset=bn_offset, scale_shift_num=bn_scale_shift
                                       )
        c += 1
        
        # 输出还原成4维
        array_output = output_to_feature_map(array_output, out_h, out_w, multi_batch=False)
        print(array_output.shape)
        # 统计误差
        max_error = (array_output - output_data).max() / output_data.max() * 100
        print(f'Max Error: {max_error} %')
        
        plt.scatter(array_output[0].flatten(), output_data[0].flatten(), alpha=0.5)
        plt.xlabel(f'Simulation Results')
        plt.ylabel(f'Pytorch Results')
        plt.title(f'Max Error: {round(max_error, 2)} %, ADC quant level = {c}, Chip Weight = {w_int}')
        # plt.tight_layout()
            
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'fig\\PE_noise_0.05.png')
    plt.close()
        
    
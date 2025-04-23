import copy
import glob
import math
import os.path
import pickle
import numpy as np
from utilize import *

def process_model_adapter_datas(simulated_data_root, save_inputs_path, save_labels_path, all_number=10000):
    pt_inputs = []
    for i in range(0, all_number, 100):
        pt_input_path = os.path.join(simulated_data_root, 'features_save', 'soft_ware_int_batch_%s.npy' % str(i))
        try:
            temp = np.load(pt_input_path, allow_pickle=True)
            pt_inputs.append(temp)
        except:
            print()
    pt_s_path = os.path.join(simulated_data_root, 'features_save', 'soft_ware_scale.npy')
    s = np.load(pt_s_path, allow_pickle=True).item()
    data_s = {'data': np.vstack(pt_inputs), 's': s}
    np.save(save_inputs_path, data_s)
    pt_labels = []
    for i in range(0, all_number, 100):
        pt_label_path = os.path.join(simulated_data_root, 'features_save', 'target_batch_%s.npy' % str(i))
        try:
            temp = np.load(pt_label_path, allow_pickle=True)
            pt_labels.append(temp)
        except:
            print()
    pt_label = np.concatenate(pt_labels)
    np.save(save_labels_path, pt_label)
    print()

def get_model_middle_results_use_tool_chains(simulated_data_root, begin_layer_name='graph_input', step=5):
    model_onnx_path = os.path.join(simulated_data_root, 'model.onnx')
    pt_inputs_path = os.path.join(simulated_data_root, 'inputs.npy')
    pt_labels_path = os.path.join(simulated_data_root, 'labels.npy')
    model_quant_args_path = glob.glob(os.path.join(simulated_data_root, '*.pickle'))[0]
    if not os.path.exists(pt_inputs_path) or not os.path.exists(pt_labels_path):
        process_model_adapter_datas(simulated_data_root, pt_inputs_path, pt_labels_path)
    pt_layer1_input = np.load(pt_inputs_path, allow_pickle=True).item()
    pt_label = np.load(pt_labels_path, allow_pickle=True)
    onnx_obj = ConvertONNX(model_onnx_path, fix_layer_name=True)
    with open(model_quant_args_path, 'rb') as f:
        quant_info = pickle.load(f)
    set_care_node_names = list(onnx_obj.ir.layers.keys())
    for key in set_care_node_names:
        if quant_info.get(key) is not None:
            if quant_info[key].get('hard_params') is not None:
                quant_info[key]['hard_params']['int_flag'] = True
    onnx_layer_info = copy.deepcopy(quant_info)
    layer_keys = list(onnx_obj.ir.layers.keys())
    begin_layer_name_index = layer_keys.index(begin_layer_name)
    care_node_names = []
    for (ith, name) in enumerate(layer_keys):
        if ith > begin_layer_name_index and ('Conv' in name or 'Relu' in name or 'Flatten' in name or ('MatMul' in name)):
            care_node_names.append(name)
    input_node_name = begin_layer_name
    input_data = pt_layer1_input
    output_node_name = care_node_names
    save_path = os.path.join(simulated_data_root, '%s.npy' % input_node_name)
    if not os.path.exists(save_path):
        np.save(save_path, input_data)
    batch_size = len(pt_label) // step
    num_iters = step
    for ith in range(num_iters):
        input_node_and_data = {input_node_name: {'data': input_data['data'][ith * batch_size:(ith + 1) * batch_size], 's': input_data['s']}}
        conv_relu_out_save_path = os.path.join(simulated_data_root, 'conv_relu_out_%d.npy' % ith)
        if not os.path.exists(conv_relu_out_save_path):
            save_path = conv_relu_out_save_path
            get_output_from_specified_node(onnx_obj, input_node_and_data, output_node_name, onnx_layer_info, callback, save_path, paras=quant_info)
            print()
    del pt_layer1_input, input_data, input_node_and_data
    num_step = step
    sub_relu_out_and_conv_s_path = os.path.join(simulated_data_root, 'sub_relu_out_and_conv_s.npy')
    if os.path.exists(sub_relu_out_and_conv_s_path):
        sub_relu_out_and_conv_s = np.load(sub_relu_out_and_conv_s_path, allow_pickle=True).item()
    else:
        sub_relu_out_and_conv_s = {key: {'data': None, 's': 1.0} for key in care_node_names}
        print()
        for care_node_name in care_node_names:
            node_save_path = os.path.join(simulated_data_root, '%s.npy' % care_node_name)
            care_node_data = []
            care_node_s = 1
            if not os.path.exists(node_save_path):
                for ith in range(num_iters):
                    conv_relu_out_save_path = os.path.join(simulated_data_root, 'conv_relu_out_%d.npy' % ith)
                    conv_relu_outs = np.load(conv_relu_out_save_path, allow_pickle=True).item()
                    data_s = conv_relu_outs[care_node_name]
                    care_node_data.append(data_s['data'])
                    care_node_s = data_s['s']
                data_s = {'data': np.vstack(care_node_data), 's': care_node_s}
                np.save(node_save_path, data_s)
            data_s = np.load(node_save_path, allow_pickle=True).item()
            if 'Relu' in care_node_name or 'Flatten' in care_node_name:
                sub_relu_out_and_conv_s[care_node_name]['data'] = data_s['data'][::num_step]
            sub_relu_out_and_conv_s[care_node_name]['s'] = data_s['s']
            del data_s
        np.save(sub_relu_out_and_conv_s_path, sub_relu_out_and_conv_s)
        print()
    care_node_path = os.path.join(simulated_data_root, '%s.npy' % care_node_names[-1])
    rram_output = np.load(care_node_path, allow_pickle=True).item()['data']
    targets = pt_label
    (premax_num, top1, top5) = cal_accuracy_use_numpy(targets, rram_output)
    print()
    print()
    print()
    print()
    return (pt_label, onnx_obj, onnx_layer_info, quant_info, sub_relu_out_and_conv_s)

def get_first_read_weight_means(chip_num, show=False, save_path=None, chip_id=0):
    chip_first_read_weights_mean = np.zeros([6, 8])
    for tile in range(6):
        for xb in range(0, 8):
            rram_read_weight = a111_read_weight(chip_id, tile_id=tile, xb_id=xb)
            chip_first_read_weights_mean[tile, xb] = rram_read_weight.mean()
    plt.imshow(chip_first_read_weights_mean, cmap='cool', vmin=0, vmax=chip_first_read_weights_mean.max())
    plt.colorbar()
    plt.title('chip-%d_first_read_weight_mean' % chip_num)
    plt.xlabel('xb')
    plt.ylabel('tile')
    plt.savefig(save_path.replace('.npy', '.png'))
    if show:
        plt.show()
    plt.close()
    np.save(save_path, chip_first_read_weights_mean)
    return chip_first_read_weights_mean

def get_seppch_each_conv_or_fc_inputs(simulated_data_root, each_layer_outs_s, step):
    relu0_path = os.path.join(simulated_data_root, 'Relu_0.npy')
    relu0_data_s = np.load(relu0_path, allow_pickle=True).item()
    relu1_data_s = each_layer_outs_s['Relu_1']
    fc1_input_data_s = relu0_data_s['data'] * relu0_data_s['s']
    fc2_input_data_s = relu1_data_s['data'] * relu1_data_s['s']
    each_conv_layer_inputs = {'MatMul_1': fc1_input_data_s[::step], 'MatMul_2': fc2_input_data_s}
    return each_conv_layer_inputs

def get_radar_each_conv_or_fc_inputs(simulated_data_root, each_layer_outs_s, step):
    relu0_path = os.path.join(simulated_data_root, 'Relu_0.npy')
    relu0_data_s = np.load(relu0_path, allow_pickle=True).item()
    relu1_data_s = each_layer_outs_s['Relu_1']
    fc1_input_data_s = relu0_data_s['data'] * relu0_data_s['s']
    fc2_input_data_s = relu1_data_s['data'] * relu1_data_s['s']
    each_conv_layer_inputs = {'MatMul_1': fc1_input_data_s[::step], 'MatMul_2': fc2_input_data_s}
    return each_conv_layer_inputs

def get_mutimodal_each_conv_or_fc_inputs(simulated_data_root, each_layer_outs_s, step):
    relu0_path = os.path.join(simulated_data_root, 'Relu_0.npy')
    relu0_data_s = np.load(relu0_path, allow_pickle=True).item()
    relu1_data_s = each_layer_outs_s['Relu_1']
    fc1_input_data_s = relu0_data_s['data'] * relu0_data_s['s']
    fc2_input_data_s = relu1_data_s['data'] * relu1_data_s['s']
    each_conv_layer_inputs = {'MatMul_1': fc1_input_data_s[::step], 'MatMul_2': fc2_input_data_s}
    return each_conv_layer_inputs

def get_img_each_conv_or_fc_inputs(simulated_data_root, each_layer_outs_s, step):
    relu0_path = os.path.join(simulated_data_root, 'Relu_0.npy')
    relu0_data_s = np.load(relu0_path, allow_pickle=True).item()
    relu1_data_s = each_layer_outs_s['Relu_1']
    relu2_data_s = each_layer_outs_s['Relu_2']
    relu3_data_s = each_layer_outs_s['Relu_3']
    avrage_data_s = each_layer_outs_s['Flatten_0']
    conv1_input_data_s = relu0_data_s['data'] * relu0_data_s['s']
    conv2_input_data_s = relu1_data_s['data'] * relu1_data_s['s']
    faltten_input_data_s = relu2_data_s['data'] * relu2_data_s['s']
    matmul0_input_data_s = avrage_data_s['data'] * avrage_data_s['s']
    matmul1_input_data_s = relu3_data_s['data'] * relu3_data_s['s']
    each_conv_layer_inputs = {'Conv_1': conv1_input_data_s[::step], 'Conv_2': conv2_input_data_s, 'MatMul_0': matmul0_input_data_s, 'MatMul_1': matmul1_input_data_s}
    return each_conv_layer_inputs

def get_care_name_and_input_shapes(onnx_obj, onnx_layer_info, quant_info, input_shapes, begin_layer_name='Relu_0', begin_layer_shape=[64, 28, 28]):
    layer_keys = list(onnx_obj.ir.layers.keys())
    begin_layer_name_index = layer_keys.index(begin_layer_name)
    care_node_names = []
    for (ith, name) in enumerate(onnx_obj.ir.layers.keys()):
        if ith > begin_layer_name_index and ('Conv' in name or 'MatMul' in name):
            care_node_names.append(name)
    _ = get_output_from_specified_node(onnx_obj, input_node_and_data={begin_layer_name: {'data': np.zeros([1] + begin_layer_shape, dtype=np.float32), 's': 1}}, output_node=layer_keys[-2], onnx_layer_info=onnx_layer_info, func=callback, paras=quant_info)
    care_name_input_shapes = {key: [20] + input_shapes[key][1:] for key in care_node_names}
    for key in care_node_names:
        if 'MatMul' in key:
            care_name_input_shapes[key][-1] = 320
    return (care_node_names, care_name_input_shapes)

def speech_care_node_name_2_input_node_name():
    return {'MatMul_1': 'Relu_0', 'MatMul_2': 'Relu_1'}

def radar_care_node_name_2_input_node_name():
    return {'MatMul_1': 'Relu_0', 'MatMul_2': 'Relu_1'}

def img_node_name_2_input_node_name():
    return {'Conv_1': 'Relu_0', 'Conv_2': 'Relu_1', 'Flatten_0': 'Relu_2', 'MatMul_0': 'Flatten_0', 'MatMul_1': 'Relu_3'}

def mutimodal_care_node_name_2_input_node_name():
    return {'MatMul_1': 'Relu_0', 'MatMul_2': 'Relu_1'}

def get_chip_mapped_pools():
    mapped_pools = {'%d-%d' % (tile, xb) for tile in range(6) for xb in range(0, 8, 2)}
    not_available_mapped_pools = {'2-2', '2-6'}
    available_mapped_pools = set(mapped_pools) - set(not_available_mapped_pools)
    available_mapped_pools = sorted(available_mapped_pools)
    return available_mapped_pools

def linear_fit_channel_scale_bias(x, y, plot_flag=False):
    (m, n) = x.shape
    scales = []
    biases = []
    for i in range(n):
        if x[:, i].sum() == 0 or y[:, i].sum() == 0:
            (scale, bias) = (0, 0)
        else:
            (scale, bias) = np.polyfit(x[:, i], y[:, i], 1)
        scales.append(scale)
        biases.append(bias)
        if plot_flag:
            plt.scatter(x[:, i], y[:, i])
            plt.plot(x[:, i], scale * x[:, i] + bias)
    if plot_flag:
        plt.show()
    return (np.array(scales), np.array(biases))

def std_cal(x, y, method=1, XB_num=0, shift_num=0):
    x = x.flatten()
    y = y.flatten()
    x_set = list(set(x))
    max_x = max(x_set)
    min_x = min(x_set)
    if method == 1:
        y_range = np.max(y) - np.min(y)
    elif method == 2:
        y_range = np.mean(y[np.where(x == max_x)]) - np.mean(y[np.where(x == min_x)])
    if method == 3:
        if not XB_num:
            raise ValueError('need XB_num setting for computing std_cal. ')
        temp_range = math.floor(XB_num * 127 * 17 / 2 ** shift_num) - math.floor(XB_num * -128 * 17 / 2 ** shift_num)
        y_range = min(511, temp_range)
        assert y_range > 0
    if y_range <= 0:
        y_range = 1.0
    std_list = []
    for data in x_set:
        ind = np.where(x == data)
        std_y = np.std(y[ind])
        std_list.append(std_y)
    return np.mean(std_list) / y_range

def get_fit_weight_noise(rram_data, soft_data, method=1):
    noise_list = []
    std_list = []
    soft_std = []
    nonoise_data = soft_data['noise_0']
    rram_std = std_cal(nonoise_data, rram_data, method=method)
    for k in soft_data:
        if k.split('_')[-1] == '0':
            continue
        noise_list.append(float(k.split('_')[-1]))
        std = std_cal(nonoise_data, soft_data[k], method=method)
        soft_std.append(std)
        std_list.append(abs(rram_std - std))
    noise_list = np.array(noise_list)
    std_list = np.array(std_list)
    soft_std = np.array(soft_std)
    return (noise_list[std_list == min(std_list)][0] / 100.0, rram_std, soft_std, noise_list)

def map_weight_and_adjust_offset_and_search_cal_weight_noise_ratio(onnx_obj, onnx_layer_info, quant_info, input_shapes, offset_row_begin, each_conv_layer_inputs, save_root, cal_weight_noise_ratio_threshold=0.6, offset_abs_mean_threshold=80, channel_slope_min=0.1, begin_layer_name='Relu_0', begin_layer_shape=[64, 28, 28], modal='radar', debug=False, chip_id=0):
    (care_node_names, care_name_input_shapes) = get_care_name_and_input_shapes(onnx_obj, onnx_layer_info, quant_info, input_shapes, begin_layer_name=begin_layer_name, begin_layer_shape=begin_layer_shape)
    if modal == 'img':
        care_node_name_2_input_node_names = img_node_name_2_input_node_name()
    elif modal == 'speech':
        care_node_name_2_input_node_names = speech_care_node_name_2_input_node_name()
    elif modal == 'radar':
        care_node_name_2_input_node_names = radar_care_node_name_2_input_node_name()
    elif modal == 'mutimodal':
        care_node_name_2_input_node_names = mutimodal_care_node_name_2_input_node_name()
    save_path = os.path.join(save_root, 'map_infos_temp.npy')
    if os.path.exists(save_path):
        map_infos_temp = np.load(save_path, allow_pickle=True).item()
        care_name_2_map_infos = map_infos_temp['care_name_2_map_infos']
        available_mapped_pools = map_infos_temp['available_mapped_pools']
    else:
        available_mapped_pools = get_chip_mapped_pools()
        care_name_2_map_infos = {key: {} for key in care_node_names}
    save_root_about_map = os.path.join(save_root, 'about_map')
    os.makedirs(save_root_about_map, exist_ok=True)
    begin = time.time()
    print_file = open(os.path.join(save_root, 'map_info.txt'), 'a')
    for (ith, care_node_name) in enumerate(care_node_names):
        temp = care_name_2_map_infos[care_node_name].get('xb', None)
        if temp is not None:
            continue
        input_node_name = care_node_name_2_input_node_names[care_node_name]
        inputs = each_conv_layer_inputs[care_node_name]
        input_node_and_data = {input_node_name: {'data': inputs, 's': 1}}
        map_succeed = False
        while not map_succeed:
            if len(available_mapped_pools):
                tile_xb = available_mapped_pools.pop(0)
            else:
                np.save(save_path, {'care_name_2_map_infos': care_name_2_map_infos, 'available_mapped_pools': available_mapped_pools})
                for (key, v) in care_name_2_map_infos.items():
                    if not v:
                        continue
                    print()
                print()
                print_file.close()
                exit('available_mapped_pools is empty list, please replace the chip.\navailable_mapped_pools is empty list, please replace the chip.\navailable_mapped_pools is empty list, please replace the chip.\n')
            (tile, xb) = list(map(int, tile_xb.split('-')))
            input_shape = care_name_input_shapes[care_node_name]
            pt_weight = onnx_obj.model_parser.weight_numpy['%s.weight' % care_node_name]
            care_node_layer = onnx_obj.ir.layers[care_node_name]
            rram_input_zeros = np.zeros(input_shape).astype(np.uint8)
            if 'Conv' in care_node_name:
                layer_infos = map_one_layer_conv_and_adjust_offset(tile, xb, offset_row_begin, pt_weight, care_node_layer, rram_input_zeros, debug=debug, chip_id=chip_id)
            elif 'MatMul' in care_node_name:
                layer_infos = map_one_layer_fc_and_adjust_offset(tile, xb, offset_row_begin, pt_weight, care_node_layer, rram_input_zeros, debug=debug, chip_id=chip_id)
            else:
                exit('%s not in Conv or MatMul.' % care_node_name)
            temp_map_infos = {care_node_name: layer_infos}
            results = cal_care_node_result(care_node_name, inputs, temp_map_infos, onnx_layer_info, onnx_obj.ir.layers[care_node_name], offset_row_begin=offset_row_begin, save_root=save_root_about_map, return_dict=True, chip_id=chip_id)
            output_offset = results['rram_out']
            offset = results['offset']
            out_data_s = get_output_from_specified_node(onnx_obj, input_node_and_data, care_node_name, None, paras=quant_info)
            sim_out = out_data_s['data']
            if 'Conv' in care_node_name:
                num_column = output_offset.shape[1]
                rram_data = output_offset.transpose((1, 0, 2, 3)).reshape(num_column, -1).T
                sim_data = sim_out.transpose((1, 0, 2, 3)).reshape(num_column, -1).T
            elif 'MatMul' in care_node_name:
                rram_data = output_offset
                sim_data = sim_out
            (channel_slope, channel_bias) = linear_fit_channel_scale_bias(sim_data, rram_data)
            output_noise_ratio = std_cal(sim_out, output_offset, method=1, XB_num=len(layer_infos['xb']), shift_num=round(onnx_layer_info[care_node_name]['hard_params']['shift_num']))
            soft_output = {}
            soft_output['noise_{}'.format(0)] = sim_out
            quant_info_copy = copy.deepcopy(quant_info)
            quant_info_copy[care_node_name]['hard_params']['Gain_Error'] = channel_slope
            quant_info_copy[care_node_name]['hard_params']['offset_vector'] = channel_bias
            for noise in range(5, 81, 5):
                quant_info_copy[care_node_name]['w_args']['noise_scale'] = noise / 100
                out_data_s = get_output_from_specified_node(onnx_obj, input_node_and_data, care_node_name, None, paras=quant_info_copy)
                out_data = out_data_s['data']
                soft_output['noise_{}'.format(noise)] = out_data
            (fit_noise, rram_std, soft_std, noise_list) = get_fit_weight_noise(output_offset, soft_output, method=1)
            offset_abs_max = np.abs(offset).max()
            offset_median_abs = np.abs(np.median(offset))
            offset_abs_mean = np.abs(offset).mean()
            channel_slope_min_v = channel_slope.min()
            channel_slope_max_v = channel_slope.max()
            channel_slope_flag = channel_slope_min_v > channel_slope_min
            offset_flag = offset_abs_mean < offset_abs_mean_threshold
            cal_wn_flag = fit_noise < cal_weight_noise_ratio_threshold
            if care_node_name != care_node_names[-1]:
                flag = cal_wn_flag and offset_flag and channel_slope_flag
            else:
                flag = cal_wn_flag and offset_flag
            if flag:
                map_succeed = True
            log_info = '%s try tile %d xb %d, cal_weight_noise_ratio:%.3f, output_noise_ratio:%.3f,\n offset_abs_mean:%.2f, channel_slope_min_v:%.2f, channel_slope_max_v:%.2f' % (care_node_name, tile, xb, fit_noise, output_noise_ratio, offset_abs_mean, channel_slope_min_v, channel_slope_max_v)
            print()
            print()
        care_name_2_map_infos[care_node_name]['tile'] = layer_infos['tile']
        care_name_2_map_infos[care_node_name]['xb'] = copy.deepcopy(layer_infos['xb'])
        care_name_2_map_infos[care_node_name]['bias_input_value'] = layer_infos['bias_input_value']
        care_name_2_map_infos[care_node_name]['input_shape'] = input_shape
        care_name_2_map_infos[care_node_name]['offset'] = offset
        care_name_2_map_infos[care_node_name]['channel_slope'] = channel_slope[None]
        care_name_2_map_infos[care_node_name]['channel_bias'] = channel_bias[None]
        care_name_2_map_infos[care_node_name]['cal_weight_noise'] = fit_noise
        care_name_2_map_infos[care_node_name]['output_noise'] = output_noise_ratio
        np.save(save_path, {'care_name_2_map_infos': care_name_2_map_infos, 'available_mapped_pools': available_mapped_pools})
    print()
    print_file.close()
    return care_name_2_map_infos

def cal_conv_fc_used_a111_chip_and_other_used_tool_chains(onnx_obj, onnx_layer_info, quant_info, each_layer_outs_s, repaired_care_name_2_map_infos, offset_row_begin, inputs, outputs, save_root=None, gap_tile=0, chip_id=0):
    ir = onnx_obj.ir
    layers = ir.flatten_layers()
    (inp, oup) = ir.get_io_layers(layers)
    (inl, oul) = (layers[inp], layers[oup])
    if isinstance(inputs, dict):
        data = {k: tuple(v) if isinstance(v, (tuple, list)) else (v,) for (k, v) in inputs.items() if v is not None}
    elif isinstance(inputs, (tuple, list)):
        assert len(inputs) == len(inl.inputs)
        data = {inp: tuple(inputs)}
    else:
        data = {inp: (inputs,)}
    for (k, v) in data.items():
        assert k in layers, f'invalid input name {k!r}'
        assert isinstance(v, (tuple, list)), f'invalid inputs type {type(v)}'
    ons = None
    if outputs is not None:
        if isinstance(outputs, str):
            ons = set(outputs)
        elif outputs is True:
            ons = set(layers.keys()) - {inp, oup}
        elif isinstance(outputs, (tuple, list, set)):
            ons = set(outputs)
        for k in ons:
            assert k in layers, f'invalid output name {k!r}'
    for (name, layer) in layers.items():
        if layer.type == 'op' and layer.op.op_id in ['constant']:
            continue
        if name in data:
            continue
        if any((dd.parse_ref()[0] not in data for dd in layer.inputs)):
            continue
        if name == oup:
            break
        x = []
        last_care_node_names = []
        for dd in layer.inputs:
            (nm, idx) = dd.parse_ref()
            if len(data[nm]) == 1:
                x.append(data[nm][0])
            else:
                x.append(data[nm])
            last_care_node_names.append(nm)
        care_node_name = name
        if 'Conv' in name or 'MatMul' in name:
            s = each_layer_outs_s[care_node_name]['s']
            assert len(x) == 1, 'Convolution and fully connected input can only be one !!!'
            if isinstance(x[0], tuple):
                inputs = x[0][0] * x[0][1]
            else:
                inputs = x[0]
            offset = repaired_care_name_2_map_infos[care_node_name].get('offset', None)
            rram_output_offset = cal_care_node_result(care_node_name, inputs, repaired_care_name_2_map_infos, onnx_layer_info, onnx_obj.ir.layers[care_node_name], offset_row_begin=offset_row_begin, save_root=save_root, offset=offset, chip_id=chip_id)
            channel_slope = repaired_care_name_2_map_infos[care_node_name]['channel_slope']
            channel_bias = repaired_care_name_2_map_infos[care_node_name]['channel_bias']
            if 'Conv' in name:
                channel_slope = channel_slope[:, :, None, None]
                channel_bias = channel_bias[:, :, None, None]
                rram_output_offset = rram_output_offset - channel_bias
                rram_output_offset = rram_output_offset.clip(-256, 255).astype(np.int16).astype(np.float32)
            y = (rram_output_offset, s)
        else:
            output_node_name = care_node_name
            if len(x) == 1:
                if isinstance(x[0], tuple):
                    input_data = x[0][0]
                    input_s = x[0][1]
                else:
                    input_data = x[0]
                    input_s = 1
                input_node_and_data = {last_care_node_names[0]: {'data': input_data, 's': input_s}}
                if 'GlobalAveragePool' in name:
                    out = calculate_GlobalAvgPool(input_data=input_data, tile_id=gap_tile, chip_id=chip_id)
                    y = tuple([out, input_s])
                else:
                    out_s = get_output_from_specified_node(onnx_obj, input_node_and_data, output_node_name, onnx_layer_info, None, paras=quant_info)
                    y = tuple(out_s.values())
            elif len(x) == 2:
                input_node_name1 = last_care_node_names[0]
                input_node_name2 = last_care_node_names[1]
                output_node_name = care_node_name
                if isinstance(x[0], tuple) and isinstance(x[1], tuple):
                    input_node_and_data = {input_node_name1: {'data': x[0][0], 's': x[0][1]}, input_node_name2: {'data': x[1][0], 's': x[1][1]}}
                else:
                    input_node_and_data = {input_node_name1: {'data': x[0], 's': each_layer_outs_s[input_node_name1]['s']}, input_node_name2: {'data': x[1], 's': each_layer_outs_s[input_node_name2]['s']}}
                out_s = get_output_from_specified_node(onnx_obj, input_node_and_data, output_node_name, onnx_layer_info, None, paras=quant_info)
                y = tuple(out_s.values())
        if not isinstance(y, (tuple, list)):
            y = (y,)
        data[name] = tuple(y)
        if ons is not None and all((k in data for k in ons)):
            break
    return data
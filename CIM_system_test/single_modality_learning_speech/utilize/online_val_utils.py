
import seaborn as sns
from utilize import *
from utilize.api import *
from a111sdk import *
from quantization_and_noise.quant_layer import *
import pandas as pd

def save_np_array_xlsx(data, data_path):
    df = pd.DataFrame(data)
    df.to_excel(data_path, index=False, header=False)

def cal_one_layer_fc_output_and_offset(tile, xb, adc_range, shift_num, offset_row_begin, bias_input_value, rram_input, batch_size, num_col, offset_sub_flag=True):
    print()
    print()
    print()
    print()
    print()
    input_shape = rram_input.shape
    temp = np.zeros([input_shape[0], 320 * ((input_shape[1] - 1) // 320 + 1)])
    temp[:, offset_row_begin:offset_row_begin + input_shape[1]] = rram_input
    pt_data_2_rram = temp.astype(np.uint8)
    pt_data_2_rram[:, :offset_row_begin] = bias_input_value
    rram_input = pt_data_2_rram
    fc_iters = int((rram_input.shape[0] - 1) // batch_size) + 1
    outputs = []
    for ith in range(fc_iters):
        r_input = rram_input[ith * batch_size:(ith + 1) * batch_size]
        output = calculate_FC_one_layer(r_input, tile=tile, xb=xb, num_column=num_col, shift_num=[shift_num], adc_range=[adc_range], relu=False)
        outputs.append(output)
    rram_output = np.vstack(outputs)
    if offset_sub_flag:
        r_input = np.zeros_like(rram_input[:batch_size]).astype(np.uint8)
        r_input[:, :offset_row_begin] = bias_input_value
        output = calculate_FC_one_layer(r_input, tile=tile, xb=xb, num_column=num_col, shift_num=[shift_num], adc_range=[adc_range], relu=False)
        rram_output = (rram_output - output.mean(axis=0)[None]).clip(-256, 255)
    rram_output = rram_output.astype(np.float32)
    return rram_output

def trans_pt_data_2_rram(pt_data, voltage=136):
    rram_data = pt_data * voltage
    rram_data = rram_data.astype(np.uint8)
    return rram_data

def pt_sequence_2_rram_discretization(pt_sequence):
    (pt_sequence_row, pt_sequence_colum) = pt_sequence.shape
    rram_discretization = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum = np.zeros([pt_sequence_row, 128])
    pt_sequence_128colum[:, :pt_sequence_colum] = pt_sequence
    for rram_colum in range(127):
        mapping_index = 4 * rram_colum % 127
        rram_discretization[:, mapping_index] = pt_sequence_128colum[:, rram_colum]
    rram_discretization[:, 127] = pt_sequence_128colum[:, 127]
    return rram_discretization

def trans_pt_weight_2_rram_row_direct(pt_weight, row_begin=8, pos_sa=3, neg_sa=3):
    (row, colum) = pt_weight.shape
    rram_weight = np.zeros([row * 2, colum])
    pos_weight = np.zeros_like(pt_weight)
    neg_weight = np.zeros_like(pt_weight)
    flag = pt_weight > 0
    pos_weight[flag] = pos_sa
    flag = pt_weight < 0
    neg_weight[flag] = neg_sa
    rram_weight[::2, :] = pos_weight
    rram_weight[1::2, :] = neg_weight
    return rram_weight

def trans_pt_weight_2_rram(pt_weight, row_begin=8, pos_sa=3, neg_sa=3):
    (row, colum) = pt_weight.shape
    rram_weight = np.zeros([row * 2, colum])
    pos_weight = np.zeros_like(pt_weight)
    neg_weight = np.zeros_like(pt_weight)
    flag = pt_weight > 0
    pos_weight[flag] = pos_sa
    flag = pt_weight < 0
    neg_weight[flag] = neg_sa
    rram_weight[::2, :] = pos_weight
    rram_weight[1::2, :] = neg_weight
    sub_mapping_weight = pt_sequence_2_rram_discretization(rram_weight)
    mapping_weight = np.zeros([640, 128])
    mapping_weight[row_begin * 2:row_begin * 2 + rram_weight.shape[0]] = sub_mapping_weight
    mapping_weight = mapping_weight.astype(np.uint8)
    return mapping_weight

def trans_rram_weight_pt(read_weight, SA_scale=3):
    read_weight = read_weight.astype(np.float32)
    (dim1, dim2) = read_weight.shape
    data_weight = read_weight[::2, :] - read_weight[1::2, :]
    data_weight = data_weight / SA_scale
    new_data_weight = np.zeros([dim1 // 2, dim2])
    assert dim2 // 4 == 32, 'array shape error'
    for t in range(4):
        new_data_weight[:, 32 * t:(t + 1) * 32] = data_weight[:, t::4]
    return new_data_weight

def trans_rram_weight_pt_col_trans(read_weight, SA_scale=3):
    read_weight = read_weight.astype(np.float32) / SA_scale
    (dim1, dim2) = read_weight.shape
    new_data_weight = np.zeros([dim1, dim2])
    assert dim2 // 4 == 32, 'array shape error'
    for t in range(4):
        new_data_weight[:, 32 * t:(t + 1) * 32] = read_weight[:, t::4]
    return new_data_weight

def plot_heatmap(matrix, epoch, batch_idx, fig_path, fc_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='cool')
    plt.title(f'Weights - Epoch: {epoch} Batch: {batch_idx}')
    plt.savefig(fig_path + f'{fc_name}_weight_epoch{epoch}_batch_idx{batch_idx}.png')
    plt.show()

def filter_by_labels(dataset, labels):
    
    indices = []
    for (idx, (_, target)) in enumerate(dataset):
        if target in labels:
            indices.append(idx)
    return indices

def evaluate(model, data_loader, infer_batch_num=32):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (batch_idx, (data, target)) in enumerate(data_loader):
            if batch_idx >= infer_batch_num:
                break
            (data, target) = (data.to(device), target.to(device))
            (output, _) = model(data)
            (_, predicted) = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def evaluate_lastlayer_rram(model, data_loader, tile, xb):
    model.eval()
    correct = 0
    total = 0
    input_quantizer = uniform_quantizer(symmetric=False, bit=1, clamp_std=0, th_point='max', th_scale=0.5, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
    with torch.no_grad():
        for (batch_idx, (data, target)) in enumerate(data_loader):
            (data, target) = (data.to(device), target.to(device))
            (output, fc3_input) = model(data)
            (fc3_input_int, fc3_input_scale) = input_quantizer(fc3_input)
            batch_size = 8
            rram_input = trans_pt_data_2_rram(fc3_input_int.detach().numpy(), voltage=136)
            rram_out = cal_one_layer_fc_output_and_offset(tile, xb, adc_range=16, shift_num=3, offset_row_begin=0, bias_input_value=0, rram_input=rram_input, batch_size=batch_size, num_col=10)
            if batch_idx == 0:
                (scales, biases) = scatter_plot_with_fit_channel_wise(output.detach().numpy(), rram_out, tile=tile, xb=xb[0], epoch=0, batch=batch_idx)
                print()
            rram_out_to_pt = (rram_out - biases) / scales
            (_, predicted) = torch.max(torch.tensor(rram_out_to_pt), 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def evaluate_last2layer_rram(model, data_loader, tile, xb, bias_input, offset_row_begin, batch_size, all_num):
    (num_class, last_channel) = model.fc4.weight.shape
    model.eval()
    correct = 0
    total = 0
    fc2_xb = [xb[0]]
    fc3_xb = [xb[1]]
    fc2_bias_input = bias_input[0]
    fc3_bias_input = bias_input[1]
    input_quantizer = uniform_quantizer(symmetric=False, bit=1, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
    with torch.no_grad():
        for (ith, (data, target)) in enumerate(data_loader):
            print()
            (data, target) = (data.to(device), target.to(device))
            (output, fc2_input, _) = model(data)
            (fc2_input_int, fc2_input_scale) = input_quantizer(fc2_input)
            fc2_rram_input = trans_pt_data_2_rram(fc2_input_int.detach().numpy(), voltage=136)
            fc2_rram_out = cal_one_layer_fc_output_and_offset(tile, fc2_xb, adc_range=32, shift_num=4, offset_row_begin=offset_row_begin, bias_input_value=fc2_bias_input, rram_input=fc2_rram_input, batch_size=8, num_col=last_channel, offset_sub_flag=True)
            fc2_rram_out_to_pt = fc2_rram_out
            fc3_input_rram_out_to_pt = np.maximum(0, fc2_rram_out_to_pt)
            (fc3_input_int, fc3_input_scale) = input_quantizer(torch.tensor(fc3_input_rram_out_to_pt))
            fc3_rram_input = trans_pt_data_2_rram(fc3_input_int.detach().numpy(), voltage=136)
            fc3_rram_out = cal_one_layer_fc_output_and_offset(tile, fc3_xb, adc_range=32, shift_num=4, offset_row_begin=offset_row_begin, bias_input_value=fc3_bias_input, rram_input=fc3_rram_input, batch_size=8, num_col=num_class, offset_sub_flag=True)
            fc3_rram_out_to_pt = fc3_rram_out
            (_, predicted) = torch.max(torch.tensor(fc3_rram_out_to_pt), 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if ith * batch_size >= all_num:
                break
    print()
    return correct / total

def evaluate_last2layer_rram_batch(model, data, target, tile, xb, bias_input, offset_row_begin, infer_batch_num=32):
    (num_class, last_channel) = model.fc4.weight.shape
    model.eval()
    correct = 0
    total = 0
    fc2_xb = [xb[0]]
    fc3_xb = [xb[1]]
    fc2_bias_input = bias_input[0]
    fc3_bias_input = bias_input[1]
    input_quantizer = uniform_quantizer(symmetric=False, bit=1, clamp_std=0, th_point='max', th_scale=0.5, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
    with torch.no_grad():
        (output, fc2_input, fc3_input) = model(data)
        (fc2_input_int, fc2_input_scale) = input_quantizer(fc2_input)
        batch_size = 8
        fc2_rram_input = trans_pt_data_2_rram(fc2_input_int.detach().numpy(), voltage=136)
        fc2_rram_out = cal_one_layer_fc_output_and_offset(tile, fc2_xb, adc_range=32, shift_num=4, offset_row_begin=offset_row_begin, bias_input_value=fc2_bias_input, rram_input=fc2_rram_input, batch_size=batch_size, num_col=last_channel, offset_sub_flag=True)
        fc2_rram_out_to_pt = fc2_rram_out
        fc3_input_rram_out_to_pt = np.maximum(0, fc2_rram_out_to_pt)
        (fc3_input_int, fc3_input_scale) = input_quantizer(torch.tensor(fc3_input_rram_out_to_pt))
        fc3_rram_input = trans_pt_data_2_rram(fc3_input_int.detach().numpy(), voltage=136)
        fc3_rram_out = cal_one_layer_fc_output_and_offset(tile, fc3_xb, adc_range=32, shift_num=4, offset_row_begin=offset_row_begin, bias_input_value=fc3_bias_input, rram_input=fc3_rram_input, batch_size=batch_size, num_col=num_class, offset_sub_flag=True)
    return (fc3_input_rram_out_to_pt, fc3_rram_out)

def bp_cal_one_fc(tile, xb, input_data, input_val, offset_row_begin, fc_weight_shape, val_bp):
    
    (input_pattern_num, input_len) = input_data.shape
    (rram_w_dim1, rram_w_dim2) = fc_weight_shape
    rram_w_dim1 = rram_w_dim1 * 2
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=24)
    fc_rram_weight_read = a111_read_weight(tile, xb, addr=[0, 0, 640, 128])
    if input_len % 64 != 0:
        input_data = np.concatenate((input_data, np.zeros((input_pattern_num, 64 - input_len % 64))), axis=1)
    input_data = input_data.astype(int) * input_val
    fc_pt_weight = trans_rram_weight_pt_col_trans(fc_rram_weight_read, SA_scale=1)
    fc_pt_weight = fc_pt_weight[2 * offset_row_begin:2 * offset_row_begin + rram_w_dim1, :32]
    sf_fc_bp = np.dot(input_data[:, :32], fc_pt_weight.T)
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=val_bp)
    rram_bp_fc = calculate_BP_one_xb(tile, xb, input_data, output_num=64, WL_index=0)
    fail_sa2 = []
    for i in range(32):
        if np.sum(rram_bp_fc[:, i] >= 30) > int(input_pattern_num * 0.3):
            fail_sa2.append(i)
    sf_fc_bp_3d = np.zeros((32, int(int(rram_w_dim1 / 32) * input_pattern_num)))
    rram_fc_bp_3d = np.zeros((32, int(int(rram_w_dim1 / 32) * input_pattern_num)))
    for i in range(int(rram_w_dim1 / 32)):
        for j in range(input_pattern_num):
            sf_fc_bp_3d[:, i * input_pattern_num + j] = sf_fc_bp[j, 32 * i:32 * (i + 1)]
    for i in range(int(rram_w_dim1 / 32)):
        for j in range(input_pattern_num):
            rram_fc_bp_3d[:, i * input_pattern_num + j] = rram_bp_fc[j, 32 * i:32 * (i + 1)]
    for i in range(32):
        if i in fail_sa2:
            rram_fc_bp_3d[i, :] = sf_fc_bp_3d[i, :]
    rram_bp_fc_sucess_fixed = bp_sa2_fit(sf_fc_bp_3d, rram_fc_bp_3d, irange=24, fixed_flag=True)
    for i in range(32):
        if i not in fail_sa2:
            rram_bp_fc_sucess_fixed[i, :] = 0
    rram_bp_fc_sucess_fixed_full = rram_bp_fc_sucess_fixed
    rram_bp_fc_sucess_fixed_full_reshape = np.zeros((input_pattern_num, int(rram_w_dim1 / 32) * 32))
    for i in range(int(rram_w_dim1 / 32)):
        for j in range(input_pattern_num):
            rram_bp_fc_sucess_fixed_full_reshape[j, 32 * i:32 * (i + 1)] = rram_bp_fc_sucess_fixed_full[:, i * input_pattern_num + j]
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=24)
    print()
    rram_bp_fc_sucess_fixed_full_reshape = rram_bp_fc_sucess_fixed_full_reshape[:, ::2] - rram_bp_fc_sucess_fixed_full_reshape[:, 1::2]
    return rram_bp_fc_sucess_fixed_full_reshape

def bp_cal_one_fc_sw(tile, xb, input_data, input_val, offset_row_begin, fc_weight_shape, val_bp):
    
    (input_pattern_num, input_len) = input_data.shape
    (rram_w_dim1, rram_w_dim2) = fc_weight_shape
    rram_w_dim1 = rram_w_dim1 * 2
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=24)
    fc_rram_weight_read = a111_read_weight(tile, xb, addr=[0, 0, 640, 128])
    if input_len % 64 != 0:
        input_data = np.concatenate((input_data, np.zeros((input_pattern_num, 64 - input_len % 64))), axis=1)
    input_data = input_data.astype(int) * input_val
    fc_pt_weight = trans_rram_weight_pt_col_trans(fc_rram_weight_read, SA_scale=1)
    fc_pt_weight = fc_pt_weight[2 * offset_row_begin:2 * offset_row_begin + rram_w_dim1, :32]
    sf_fc_bp = np.dot(input_data[:, :32], fc_pt_weight.T)
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=val_bp)
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=24)
    sf_fc_bp = sf_fc_bp / input_val
    sf_fc_bp_reshape = sf_fc_bp[:, ::2] - sf_fc_bp[:, 1::2]
    return sf_fc_bp_reshape

def trans_pt_weight_2_rram_row_direct_scale(pt_weight):
    (row, colum) = pt_weight.shape
    rram_weight = np.zeros([row * 2, colum])
    pos_weight = np.zeros_like(pt_weight)
    neg_weight = np.zeros_like(pt_weight)
    max_value = np.max(np.abs(pt_weight))
    min_value = np.min(np.abs(pt_weight))
    flag = pt_weight > 0
    pos_weight[flag] = max_value
    flag = pt_weight < 0
    neg_weight[flag] = min_value
    rram_weight[::2, :] = pos_weight
    rram_weight[1::2, :] = neg_weight
    return rram_weight

def bp_cal_one_fc_v3(tile, xb, fc_weight_fixed, input_data, input_val, offset_row_begin, fc_weight_shape, val_bp):
    
    (input_pattern_num, input_len) = input_data.shape
    (rram_w_dim1, rram_w_dim2) = fc_weight_shape
    rram_w_dim1_origin = rram_w_dim1
    rram_w_dim1 = rram_w_dim1 * 2 + 32
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=24)
    if input_len % 64 != 0:
        input_data = np.concatenate((input_data, np.zeros((input_pattern_num, 64 - input_len % 64))), axis=1)
    input_data = input_data.astype(int) * input_val
    fc_weight_fixed = fc_weight_fixed.detach().numpy()
    fc_pt_weight = trans_pt_weight_2_rram_row_direct_scale(fc_weight_fixed.T)
    fc_pt_weight_with_offset = np.zeros((rram_w_dim1, rram_w_dim2))
    fc_pt_weight_with_offset[offset_row_begin * 2:offset_row_begin * 2 + fc_pt_weight.shape[0], :fc_pt_weight.shape[1]] = fc_pt_weight
    fc_pt_weight = fc_pt_weight_with_offset[:rram_w_dim1, :32]
    sf_fc_bp = np.dot(input_data[:, :32], fc_pt_weight.T)
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=val_bp)
    rram_bp_fc = calculate_BP_one_xb(tile, xb, input_data, output_num=64, WL_index=0)
    fail_sa2 = []
    for i in range(32):
        if np.sum(rram_bp_fc[:, i] >= 30) > int(input_pattern_num * 0.3):
            fail_sa2.append(i)
    sf_fc_bp_3d = np.zeros((32, int(int(rram_w_dim1 / 32) * input_pattern_num)))
    rram_fc_bp_3d = np.zeros((32, int(int(rram_w_dim1 / 32) * input_pattern_num)))
    for i in range(int(rram_w_dim1 / 32)):
        for j in range(input_pattern_num):
            sf_fc_bp_3d[:, i * input_pattern_num + j] = sf_fc_bp[j, 32 * i:32 * (i + 1)]
    for i in range(int(rram_w_dim1 / 32)):
        for j in range(input_pattern_num):
            rram_fc_bp_3d[:, i * input_pattern_num + j] = rram_bp_fc[j, 32 * i:32 * (i + 1)]
    for i in range(32):
        if i in fail_sa2:
            rram_fc_bp_3d[i, :] = sf_fc_bp_3d[i, :]
    rram_bp_fc_sucess_fixed = bp_sa2_fit(sf_fc_bp_3d, rram_fc_bp_3d, irange=24, fixed_flag=True)
    for i in range(32):
        if i not in fail_sa2:
            rram_bp_fc_sucess_fixed[i, :] = 0
    rram_bp_fc_sucess_fixed_full = rram_bp_fc_sucess_fixed
    rram_bp_fc_sucess_fixed_full_reshape = np.zeros((input_pattern_num, int(rram_w_dim1 / 32) * 32))
    for i in range(int(rram_w_dim1 / 32)):
        for j in range(input_pattern_num):
            rram_bp_fc_sucess_fixed_full_reshape[j, 32 * i:32 * (i + 1)] = rram_bp_fc_sucess_fixed_full[:, i * input_pattern_num + j]
    a111_isource_set(tile_id=tile, iref=IREF_IDAC_SA, val=24)
    print()
    rram_bp_fc_sucess_fixed_full_reshape = rram_bp_fc_sucess_fixed_full_reshape[:, ::2] - rram_bp_fc_sucess_fixed_full_reshape[:, 1::2]
    return rram_bp_fc_sucess_fixed_full_reshape[:, offset_row_begin:rram_w_dim1_origin + offset_row_begin]

def trans_delta_w_2_operationFlag(delta_w_3value):
    (dim1, dim2) = delta_w_3value.shape
    operationFlag = np.zeros((dim1 * 2, dim2))
    positive_indices = delta_w_3value > 0
    negative_indices = delta_w_3value < 0
    operationFlag[::2, :] = positive_indices.astype(int) - negative_indices.astype(int)
    operationFlag[1::2, :] = -operationFlag[::2, :]
    operationFlag_rram = np.zeros((dim1 * 2, 128))
    for i in range(dim2):
        col_bias = i // 32
        col_k = i % 32
        operationFlag_rram[:, 4 * col_k + col_bias] = operationFlag[:, i]
    return operationFlag_rram

def trans_delta_w_2_operationFlag_for_m2(delta_w_3value, last_read_array):
    (dim1, dim2) = delta_w_3value.shape
    last_read_array_2_pt = np.zeros((dim1 * 2, dim2))
    for kk in range(dim2):
        index_k = kk % 32
        index_b = kk // 32
        last_read_array_2_pt[:, kk] = last_read_array[:dim1 * 2, 4 * index_k + index_b]
    delta_w_sign = np.sign(delta_w_3value)
    last_read_array_sign = np.sign(last_read_array_2_pt)
    operationFlag = np.zeros((dim1 * 2, dim2))
    for row in range(dim1):
        for col in range(dim2):
            if last_read_array_sign[2 * row + 1, col] == 0 and delta_w_sign[row, col] == 1:
                operationFlag[row * 2, col] = 1
                operationFlag[row * 2 + 1, col] = 0
            elif last_read_array_sign[2 * row, col] == 0 and delta_w_sign[row, col] == -1:
                operationFlag[row * 2, col] = 0
                operationFlag[row * 2 + 1, col] = 1
            elif last_read_array_sign[2 * row + 1, col] == 1 and delta_w_sign[row, col] == 1:
                operationFlag[row * 2, col] = 0
                operationFlag[row * 2 + 1, col] = -1
            elif last_read_array_sign[2 * row, col] == 1 and delta_w_sign[row, col] == -1:
                operationFlag[row * 2, col] = -1
                operationFlag[row * 2 + 1, col] = 0
    operationFlag_rram = np.zeros((dim1 * 2, 128))
    for i in range(dim2):
        col_bias = i // 32
        col_k = i % 32
        operationFlag_rram[:, 4 * col_k + col_bias] = operationFlag[:, i]
    return operationFlag_rram

def mapping_sbp_one_batch(operationFlag, tile, xb, row_start=0, col_start=0):
    MIN_STEP = 0.03125
    vbl_form_num = int((4.5 - 4.3) // MIN_STEP)
    vwl_form_num = int((3.73 - 3.3) // MIN_STEP)
    (row_len, col_len) = operationFlag.shape
    for row in range(row_len):
        for col in range(col_len):
            if operationFlag[row][col] == 1:
                origin_val = read_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col)
                set_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=1.7 * 1000, vWL=1.2 * 1000, pulse=90)
                set_aft_val = read_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col)
                diff_set = set_aft_val - origin_val
                if origin_val >= 2 or diff_set > 0:
                    pass
                else:
                    form_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=4.5 * 1000, vWL=3.6 * 1000, pulse=100000)
                    reset_val = reset_cell(tid=tile, xid=xb, row=row, col=col, vSL=4.0 * 1000, vWL=5.0 * 1000, pulse=1000)
                    set_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=1.7 * 1000, vWL=1.2 * 1000, pulse=90)
            elif operationFlag[row][col] == -1:
                reset_val = reset_cell(tid=tile, xid=xb, row=row, col=col, vSL=2.0 * 1000, vWL=4.8 * 1000, pulse=600)

def mapping_sbp_one_batch_method1_fixed(operationFlag, batch_num, tile, xb, row_start=0, col_start=0):
    MIN_STEP = 0.03125
    vbl_form_num = int((4.5 - 4.3) // MIN_STEP)
    vwl_form_num = int((3.73 - 3.3) // MIN_STEP)
    (row_len, col_len) = operationFlag.shape
    if batch_num % 2 == 0:
        for row in range(row_len):
            for col in range(col_len):
                if operationFlag[row][col] == 1:
                    origin_val = read_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col)
                    set_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=1.7 * 1000, vWL=1.2 * 1000, pulse=90)
                    set_aft_val = read_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col)
                    diff_set = set_aft_val - origin_val
                    if origin_val >= 2 or diff_set > 0:
                        pass
                    else:
                        form_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=4.5 * 1000, vWL=3.6 * 1000, pulse=100000)
                        reset_val = reset_cell(tid=tile, xid=xb, row=row, col=col, vSL=4.0 * 1000, vWL=5.0 * 1000, pulse=1000)
                        set_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=1.7 * 1000, vWL=1.2 * 1000, pulse=90)
    if batch_num % 2 == 1:
        for row in range(row_len):
            for col in range(col_len):
                if operationFlag[row][col] == -1:
                    reset_val = reset_cell(tid=tile, xid=xb, row=row, col=col, vSL=2.0 * 1000, vWL=4.8 * 1000, pulse=600)

def mapping_sbp_one_batch_method2(delta_w_3value, batch_num, tile, xb, row_start=0, col_start=0):
    vbl_set_s = 1.4
    vwl_set_s = 1.1
    set_pulse_width_s = 90
    vsl_reset_s = 1.8
    vwl_reset_s = 4.6
    reset_pulse_width_s = 600
    (dim1_delta_w, _) = delta_w_3value.shape
    (row_len, col_len) = (dim1_delta_w * 2, 128)
    assert row_len % 32 == 0, 'row len must be 32 times'
    rram_read_weight = a111_read_weight(tile, xb, addr=[0, col_start, row_len + 32, col_len], verbose=True)
    rram_read_weight = rram_read_weight[16:row_len + 16, :]
    operationFlag = trans_delta_w_2_operationFlag_for_m2(delta_w_3value, rram_read_weight)
    MIN_STEP = 0.03125
    vbl_form_num = int((4.5 - 4.3) // MIN_STEP)
    vwl_form_num = int((3.73 - 3.3) // MIN_STEP)
    if batch_num % 2 == 0:
        for row in range(row_len):
            for col in range(col_len):
                if operationFlag[row][col] == 1:
                    origin_val = read_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col)
                    set_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=vbl_set_s * 1000, vWL=vwl_set_s * 1000, pulse=set_pulse_width_s)
                    set_aft_val = read_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col)
                    diff_set = set_aft_val - origin_val
                    if origin_val >= 2 or diff_set > 0:
                        pass
                    else:
                        form_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=4.5 * 1000, vWL=3.6 * 1000, pulse=100000)
                        reset_val = reset_cell(tid=tile, xid=xb, row=row, col=col, vSL=vsl_reset_s * 1000, vWL=vwl_reset_s * 1000, pulse=reset_pulse_width_s)
                        set_val = set_cell(tid=tile, xid=xb, row=row_start + row, col=col_start + col, vBL=vbl_set_s * 1000, vWL=vwl_set_s * 1000, pulse=set_pulse_width_s)
    if batch_num % 2 == 1:
        for row in range(row_len):
            for col in range(col_len):
                if operationFlag[row][col] == -1:
                    reset_val = reset_cell(tid=tile, xid=xb, row=row, col=col, vSL=vsl_reset_s * 1000, vWL=vwl_reset_s * 1000, pulse=reset_pulse_width_s)

def form_sbp_check(operationFlag, form_flag_matrix, tile, xb, row_start=0, col_start=0):
    MIN_STEP = 0.03125
    vbl_form_num = int((4.5 - 4.3) // MIN_STEP)
    vwl_form_num = int((3.73 - 3.3) // MIN_STEP)
    (row_len, col_len) = operationFlag.shape
    assert operationFlag.shape == form_flag_matrix.shape, 'form flag matrix应该和operationflag的shape一致'
    for row in range(row_len):
        for col in range(col_len):
            if operationFlag[row][col] == 1:
                if form_flag_matrix[row][col] == 1:
                    continue
                else:
                    origin_val = read_cell(tid=tile, xid=xb, row=row, col=col)
                    if origin_val >= 2:
                        form_flag_matrix[row][col] = 1
                        continue
                    else:
                        for VWL in range(vwl_form_num):
                            for VBL in range(vbl_form_num):
                                form_after_val = set_cell(tid=tile, xid=xb, row=row, col=col, vBL=(4.3 + MIN_STEP * 3 * VBL) * 1000, vWL=(3.3 + MIN_STEP * 2 * VWL) * 1000, pulse=100000)
                                if form_after_val >= 2:
                                    form_flag_matrix[row][col] = 1
                                    form_complete_flag = True
                                    print()
                                    break
                            if form_complete_flag:
                                form_complete_flag = False
                                break
                        _ = reset_cell(tid=tile, xid=xb, row=row, col=col, vSL=4.0 * 1000, vWL=5.0 * 1000, pulse=1000)

def plot_acc(test_acc_9, test_acc_all, save_path):
    acc_len = len(test_acc_9)
    x = np.arange(acc_len)
    plt.plot(x, test_acc_9, label='test_acc_9')
    plt.plot(x, test_acc_all, label='test_acc_9_rram')
    plt.legend()
    plt.title('test_acc')
    plt.xlabel('batch #')
    plt.ylabel('acc')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, target) in data_loader:
            (data, target) = (data.to(device), target.to(device))
            (output, _) = model(data)
            (_, predicted) = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def filter_by_labels(dataset, labels):
    
    indices = []
    for (idx, (_, target)) in enumerate(dataset):
        if target in labels:
            indices.append(idx)
    return indices

def training_bp(num_epochs, model, train_loader, test_loader, optimizer, criterion=nn.CrossEntropyLoss()):
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (batch_idx, (data, target)) in enumerate(train_loader):
            (data, target) = (data.to(device), target.to(device))
            target_onehot = torch.zeros(target.shape[0], 10, device=target.device).scatter_(1, target.unsqueeze(1), 1.0)
            optimizer.zero_grad()
            (output, _, fc3_input) = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for (batch_idx, (data, target)) in enumerate(test_loader):
                (data, target) = (data.to(device), target.to(device))
                target_onehot = torch.zeros(target.shape[0], 10, device=target.device).scatter_(1, target.unsqueeze(1), 1.0)
                (output, _, _) = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_losses.append(test_loss / len(test_loader))
        accuracy = correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)
        print()
    return {'test_losses': test_losses, 'accuracy': accuracy, 'test_accuracies': test_accuracies}

def bp_sa2_fit(ideal_out, bp_out, irange=24, fixed_flag=False):
    (shape_x, shape_y) = ideal_out.shape
    fixed_bp = np.zeros_like(ideal_out)
    scales = []
    biases = []
    for i in range(shape_x):
        x = ideal_out[i]
        y = bp_out[i]
        print()
        if np.sum(x == 0) >= shape_y * 0.9:
            y = x
        else:
            (slope, intercept) = np.polyfit(x, y, 1)
            scales.append(slope)
            biases.append(intercept)
            if fixed_flag:
                y = (y - intercept) / slope
            fixed_bp[i, :] = y
    print()
    print()
    return fixed_bp
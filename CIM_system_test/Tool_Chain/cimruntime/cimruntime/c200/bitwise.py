import numpy as np
import math

def input_bitwise_expansion_fast(input_matrix, dense = True, assign_pulses = None):
    # input 是一个按照 144k 输入数据格式重新排列好的数, 该函数将其每列进行 bitwise 展开
    # input size = [rows, cols]
    # 如果 dense != True:
    #   返回值是一个展开后的稀疏二维数组, 按照 input 中的最大值作为 bitwise 展开的位数, 对每列进行展开
    # output size = [rows, cols * bitlen]
    # 如果 dense == True:
    #   返回值是一个稠密二维数组, 分别按照 input 中每列的最大值作为 bitwise 的展开位数。
    # 即展开后都是 0 的列直接丢弃
    # 在计算时需要知道 bitlen_map 中每个 col 展开了多少次, 根据该值对 144k 的计算结果进行求和处理
    # assign_pulses -> 指定bitwise展开的脉冲次数，如果不是None，则自动判定dense = Fasle

    # 先对输入矩阵进行量化，防止[0,1]浮点数矩阵报错
    input_matrix = input_matrix.round()

    # 如果输入为全 0 矩阵, 直接返回其本身, 在后续的 mvm 中跳过这次计算
    if (input_matrix == 0).all():
        return input_matrix, []

    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int32)
    input_matrix = input_matrix.astype(np.int32)
    input_ori = input_matrix + 0

    max_range = int(np.max(abs(input_matrix)))
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]

    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)

    # 用来存放展开后的矩阵
    # input_expanded 保存了 m 个与 input 相同大小的矩阵(reshape 为 1 列), 作为展开的结果, 每个矩阵元素都是+-1,0。
    input_expanded = np.zeros((rows * cols, max_range), dtype = int)

    # 对 input 矩阵进行并行 bitwise 展开,
    # 当 input 中有任意元素不为零, 则对该元素-1 或+1, 并生成一个与 input 相同大小, 且元素为-1,0,+1 的矩阵, 作为该次展开的值
    # t_expand = time.time()
    for i in range(max_range):
        # bit_cur 为一次并行 bitwise 展开的矩阵, 元素全部为-1,0,+1
        bit_cur = (input_matrix > 0) * 1 + (input_matrix < 0) * -1
        # 将该次展开的 bit_cur 保存到 input_expanded 中
        input_expanded[:, i:i + 1] = bit_cur
        # 在 input 中-1 或+1, 直到所有值为 0
        input_matrix -= bit_cur
    # t_expand = time.time() - t_expand
    # print(f'Time for expand = {t_expand}')

    # t_shift = time.time()
    input_expanded = bitwise_shift(input_expanded)
    # t_shift = time.time() - t_shift
    # print(f'Time for bitwise_shift = {t_shift}')

    # t = time.time()
    input_expanded = input_expanded.reshape(-1, rows, max_range).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)
    # input_expanded = np.split(input_expanded, cols, axis = 0)
    # input_expanded = np.concatenate(input_expanded, axis = 1)
    # t = time.time() - t
    # print(f'Time for split and concate = {t}')
    # 将展开后的稀疏矩阵中的全 0 列删除, 跳过计算
    if dense == True:
        # t = time.time()
        mask = abs(input_expanded).sum(axis = 0)
        input_expanded = input_expanded[:, mask != 0]
        zero_array_num = (mask == 0).astype(int).reshape(1, -1)
        zero_array_num = zero_array_num.reshape(-1, max_range).sum(axis = 1)
        # 获取 input 中, 每一列的最大值, 作为该列 bitwise 展开的位数
        bitlen_map = max_range - zero_array_num

        # t = time.time() - t
        # print(f'Time for dense check = {t}')
    # 如果指定了每列的展开长度，则对input_expanded补0或截取
    # elif assign_pulses:
    #     expanded_bitlen = input_expanded.shape[1]
    #     diff = assign_pulses - expanded_bitlen
    #     if diff > 0:
    #         zero_mat = np.zeros_like(input_expanded[:, 0])
    #         input_expanded = np.concatenate([input_expanded, zero_mat], axis = 1)
    #     else:
    #         input_expanded = input_expanded[:, 0:assign_pulses]
    #     bitlen_map = (np.ones([cols]) * assign_pulses).astype(np.int32)

    else:
        bitlen_map = (np.ones([cols]) * abs(input_ori).max()).astype(np.int32)

    return input_expanded, bitlen_map

def input_multi_bits_shift_expansion(input_matrix, dac_bits = 1):
    input_matrix = input_matrix.round()
    if (input_matrix == 0).all():
        return input_matrix, []

    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int32)

    input_matrix = input_matrix.astype(np.int32)
    input_ori = input_matrix + 0
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]
    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)

    shift_value = 2 ** dac_bits - 1
    input_bits = math.floor(math.log2(max(abs(input_matrix)))) +1
    max_expansion_times = math.ceil(input_bits / dac_bits)

    # 用来存放展开后的矩阵
    # input_expanded 保存了 m 个与 input 相同大小的矩阵(reshape 为 1 列), 作为展开的结果, 每个矩阵元素都是+-1,0。
    input_expanded = np.zeros((rows * cols, max_expansion_times), dtype = int)
    # 保存matrix符号
    input_matrix_sign = np.sign(input_matrix)

    for i in range(max_expansion_times):
        # 忽略符号位，取matrix的原码
        input_matrix = abs(input_matrix)
        # pulse_cur 为一次并行 bitwise 展开的矩阵, 取原始矩阵低 dac_bits 位
        # input_matrix 与 shift_value 进行位运算，并乘以符号
        pulse_cur = (input_matrix & shift_value) * input_matrix_sign

        # 将该次展开的 pulse_cur 保存到 input_expanded 中
        input_expanded[:, i:i + 1] = pulse_cur
        # 对 input_matrix 右移 dac_bits
        input_matrix = input_matrix >> dac_bits

    input_expanded = input_expanded.reshape(-1, rows, max_expansion_times).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)

    pulse_len_map = (np.ones([cols]) * max_expansion_times).astype(np.int32)

    return input_expanded, pulse_len_map


def count_overshoot_percent(output_bitwise, adc_bits = None):
    ADC_half_level = 2 ** adc_bits // 2 - 1
    count_max = (output_bitwise >= ADC_half_level - 1).sum()
    count_min = (output_bitwise <= -ADC_half_level + 1).sum()
    tot_element = output_bitwise.size
    max_percent = count_max / tot_element
    min_percent = count_min / tot_element
    return max_percent, min_percent
  


# 按照 pulse_half_level 指定的DAC等级进行等权脉冲展开
# 最大值为 x 的特征图需要展开为 x / pulse_half_level (+1) 个脉冲
def input_multi_bits_pulse_expansion(input_matrix, pulse_half_level = 7,
                                    ):
    # 参考 input_bitwise_expansion_fast 的说明
    if isinstance(input_matrix,list):
        input_matrix = np.array(input_matrix)
    # 先对输入矩阵进行量化，防止[0,1]浮点数矩阵报错
    # input_matrix_ori = input_matrix
    input_matrix = input_matrix.round()
    assert input_matrix.max() <= 127
    assert input_matrix.min() >= -128
    # 如果输入为全 0 矩阵, 直接返回其本身, 在后续的 mvm 中跳过这次计算
    if (input_matrix == 0).all():
        return input_matrix, []

    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int8)

    input_matrix = input_matrix.astype(np.int8)
    input_ori = input_matrix + 0

    max_expansion_times = math.ceil(np.max(abs(input_matrix)) / pulse_half_level)
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]

    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)

    # 用来存放展开后的矩阵
    # input_expanded 保存了 m 个与 input 相同大小的矩阵(reshape 为 1 列), 作为展开的结果, 每个矩阵元素都是+-1,0。
    input_expanded = np.zeros((rows * cols, max_expansion_times), dtype = np.int8)

    # 对 input 矩阵进行并行 bitwise 展开,
    # 当 input 中有任意元素不为零, 则对该元素-1 或+1, 并生成一个与 input 相同大小, 且元素为-1,0,+1 的矩阵, 作为该次展开的值
    # t_expand = time.time()
    for i in range(max_expansion_times - 1):
        # pulse_cur 为一次并行 bitwise 展开的矩阵, 元素全部为-pulse_half_level,0,+pulse_half_level
        pulse_cur = (input_matrix >= pulse_half_level) * pulse_half_level \
                    + (input_matrix <= -pulse_half_level) * -pulse_half_level
        # 将该次展开的 pulse_cur 保存到 input_expanded 中
        input_expanded[:, i:i + 1] = pulse_cur
        # 在 input 中-1 或+1, 直到所有值为 0
        input_matrix -= pulse_cur

    # t_expand = time.time() - t_expand
    # print(f'Time for expand = {t_expand}')
    input_expanded[:, -1] = input_matrix.reshape(-1)
    # t_shift = time.time()
    input_expanded = bitwise_shift(input_expanded)
    # t_shift = time.time() - t_shift
    # print(f'Time for bitwise_shift = {t_shift}')

    # t = time.time()
    input_expanded = input_expanded.reshape(-1, rows, max_expansion_times).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)

    # t = time.time() - t
    # print(f'Time for split and concate = {t}')

    # 将展开后的稀疏矩阵中的全 0 列删除, 跳过计算

    mask = abs(input_expanded).sum(axis = 0)
    input_expanded = input_expanded[:, mask != 0]
    zero_array_num = (mask == 0).astype(int).reshape(1, -1)
    zero_array_num = zero_array_num.reshape(-1, max_expansion_times).sum(axis = 1)
    # 获取 input 中, 每一列的最大值, 作为该列 bitwise 展开的位数
    pulse_len_map = max_expansion_times - zero_array_num
    
    return input_expanded, pulse_len_map

def expand_GPT(mat_in, pulse_half_level):
    max_val = (np.abs(mat_in) + pulse_half_level - 1) // pulse_half_level
    max_val = max_val.max()
    
    # shape_row = np.prod(mat_in.shape)
    mat_in_expanded = np.zeros((np.prod(mat_in.shape), max_val)).astype(np.int32)

    depth = np.abs(mat_in) // pulse_half_level
    remainder = np.abs(mat_in) % pulse_half_level

    sign = np.sign(mat_in)

    range_mat = np.broadcast_to(
        np.arange(max_val)[None, :],
        (np.prod(mat_in.shape), max_val)
    )

    mat_in_expanded = np.where(range_mat < depth, sign * pulse_half_level, 0)
    mat_in_expanded = np.where(range_mat == depth, sign * remainder, mat_in_expanded)
    return mat_in_expanded

# 按照 pulse_half_level 指定的 DAC 等级进行等权脉冲展开
# 最大值为 x 的特征图需要展开为 x / pulse_half_level (+1) 个脉冲
def input_multi_bits_pulse_expansion_GPT(input_matrix, pulse_half_level = 7,):
    
    # 先对输入矩阵进行量化，防止[0,1]浮点数矩阵报错
    # input_matrix_ori = input_matrix
    input_matrix = input_matrix.round()
    assert input_matrix.max() <= 127
    assert input_matrix.min() >= -128
    # 如果输入为全 0 矩阵, 直接返回其本身, 在后续的 mvm 中跳过这次计算
    if (input_matrix == 0).all():
        return input_matrix, []

    if len(input_matrix.shape) == 1:
        input_matrix = input_matrix.reshape(-1, 1).astype(np.int8)

    input_matrix = input_matrix.astype(np.int8)
    
    rows = input_matrix.shape[0]
    cols = input_matrix.shape[1]
    max_expansion_times = math.ceil(np.max(abs(input_matrix)) / pulse_half_level)

    # t_expand = time.time()
    input_matrix = input_matrix.transpose(1, 0).reshape(-1, 1)
    input_expanded = expand_GPT(input_matrix, pulse_half_level)
    # t_expand = time.time() - t_expand
    # print(f'Time for expand in GPT = {t_expand}')

    # t_shift = time.time()
    # input_expanded = bitwise_shift_GPT(input_expanded, rows)
    input_expanded = bitwise_shift(input_expanded)
    # t_shift = time.time() - t_shift
    # print(f'Time for bitwise_shift in GPT = {t_shift}')

    # t = time.time()
    input_expanded = input_expanded.reshape(-1, rows, max_expansion_times).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)

    # t = time.time() - t
    # print(f'Time for split and concate = {t}')

    # 将展开后的稀疏矩阵中的全 0 列删除, 跳过计算
    # if dense == True:
    # t = time.time()
    mask = abs(input_expanded).sum(axis = 0)
    input_expanded = input_expanded[:, mask != 0]
    zero_array_num = (mask == 0).astype(int).reshape(1, -1)
    zero_array_num = zero_array_num.reshape(-1, max_expansion_times).sum(axis = 1)
    # 获取 input 中, 每一列的最大值, 作为该列 bitwise 展开的位数
    pulse_len_map = max_expansion_times - zero_array_num

    return input_expanded, pulse_len_map


def input_multi_bits_pulse_expansion_batch(input_matrix, pulse_half_level = 7,
                                            ):
    # 获取batch
    batch = input_matrix.shape[0]
    input_expanded = []
    pulse_len_map = []
    batch_expansion_len_list = []
    
    for i in range(batch):
        input_expanded_, bitlen_map_ = input_multi_bits_pulse_expansion(input_matrix[i,:,:],
                                                                    pulse_half_level = pulse_half_level,
                                                                    )

        input_expanded.append(input_expanded_)
        pulse_len_map.append(bitlen_map_)
        batch_expansion_len_list.append(input_expanded_.shape[1])
    input_expanded = np.concatenate(input_expanded, axis=1)
    
    return input_expanded, pulse_len_map, batch_expansion_len_list

def bitwise_shift(input_expanded):
    bitlen = input_expanded.shape[1]
    if bitlen <= 1:
        return input_expanded
    # 复制 input_expanded
    input_shift_count = input_expanded + 0
    # 统计每行有多少个1
    roll = abs(input_shift_count).sum(axis = 1)
    # np.cumsum(roll)对roll中的元素进行累加
    # 例如 roll = [1,2,3]
    # 则返回 roll = [1,3,6]
    # (np.cumsum(roll) - roll) % bitlen 则是计算将每行的1首尾相连，则每行需要位移几列
    roll = (np.cumsum(roll) - roll) % bitlen
    # 对数据进行移位
    rows, column_indices = np.ogrid[:input_expanded.shape[0], :input_expanded.shape[1]]
    roll[roll < 0] += input_expanded.shape[1]
    column_indices = column_indices - roll[:, np.newaxis]
    input_expanded_shifted = input_expanded[rows, column_indices]
    return input_expanded_shifted

def bitwise_shift_batch(input_expanded):
    bitlen = input_expanded.shape[1]
    if bitlen <= 1:
        return input_expanded
    # 复制 input_expanded
    input_shift_count = input_expanded + 0
    # 统计每行有多少个1
    roll = abs(input_shift_count).sum(axis = 1)
    # np.cumsum(roll)对roll中的元素进行累加
    # 例如 roll = [1,2,3]
    # 则返回 roll = [1,3,6]
    # (np.cumsum(roll) - roll) % bitlen 则是计算将每行的1首尾相连，则每行需要位移几列
    roll = (np.cumsum(roll) - roll) % bitlen
    # 对数据进行移位
    rows, column_indices = np.ogrid[:input_expanded.shape[0], :input_expanded.shape[1]]
    roll[roll < 0] += input_expanded.shape[1]
    column_indices = column_indices - roll[:, np.newaxis]
    input_expanded_shifted = input_expanded[rows, column_indices]
    return input_expanded_shifted


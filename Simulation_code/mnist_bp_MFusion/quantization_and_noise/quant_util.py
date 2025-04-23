from turtle import forward
import torch
import torch.nn as nn
import math

# Add noise to input data
def add_noise(weight, method = 'add', n_scale = 0.074, n_range = 'max'):
    # weight -> input data, usually a weight
    # method ->
    #   'add' -> add a Gaussian noise to the weight, preferred method
    #   'mul' -> multiply a noise factor to the weight, rarely used
    # n_scale -> noise factor
    # n_range ->
    #   'max' -> use maximum range of weight times the n_scale as the noise std, preferred method
    #   'std' -> use weight std times the n_scale as the noise std, rarely used
    if n_scale == 0:
        return weight
    std = weight.std()

    if n_range == 'max':
        factor = weight.max()
    if n_range == 'std':
        factor = std
    if n_range == 'max_min':
        factor = weight.max() - weight.min()
    if n_range == 'maxabs_2':
        factor = 2 * torch.max(torch.abs(weight))

    if method == 'add':
        w_noise = torch.randn_like(weight, device=weight.device).clamp_(-3.0, 3.0) * factor * n_scale
        weight_noise = weight + w_noise
    if method == 'mul':
        w_noise = torch.randn_like(weight, device=weight.device).clamp_(-3.0, 3.0) * n_scale + 1 ## whether clamp randn to (-3,3)
        weight_noise = weight * w_noise
    weight_noise = (weight_noise - weight).detach() + weight
    return weight_noise

# ********************* 均匀量化 ***********************
# Quantize input data, uniform quantize
def data_quantization(data_float, symmetric = True, bit = 8, clamp_std = None,
                        th_point='max', th_scale=None, all_positive=False):
    # data_float -> Input data needs to be quantized
    # symmetric -> whether use symmetric quantized, int range: [-(2**(bit-1)-1), 2**(bit-1)-1]
    # bit -> quant bits
    # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]
    # th_point -> clamp data_float mode
    # th_scale -> scale the clamp thred, used together with th_point
    # all_positive -> whether data_float is all positive, int range: [0, 2**bit-1]

    std = data_float.std()
    max_data = data_float.max()
    min_data = data_float.min()
    # if min_data.item() >= 0:
    #     all_positive = True

    if clamp_std != None and clamp_std != 0 and th_scale != None:
        raise ValueError("clamp_std and th_scale, only one clamp method can be used. ")
    if clamp_std != None and clamp_std != 0:
        data_float = torch.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)
    else:
        if min_data.item() * max_data.item() < 0. and th_point == 'min':
            th = min(max_data.abs().item(), min_data.abs().item())
        else:
            th = max(max_data.abs().item(), min_data.abs().item())
        if th_scale != None:
            th *= th_scale
        data_float = torch.clamp(data_float, min = -th, max = th)

    if all_positive:
        if data_float.min().item() < 0:
            raise ValueError("all_positive uniform_quantizer's data_float is not all positive. ")
        data_range = data_float.max()
        quant_range = 2**bit-1
        zero_point = 0
    elif symmetric:
        data_range = 2*abs(data_float).max()
        quant_range = 2**bit - 2
        zero_point = 0
    else:
        data_range = data_float.max() - data_float.min()
        quant_range = 2**bit - 1
        zero_point = data_float.min() / data_range * quant_range

    if data_range == 0:
        return data_float

    scale = data_range / quant_range
    data_quantized = (round_pass(data_float / scale - zero_point) + zero_point) * scale

    return data_quantized

def data_quantization_int(data_float, symmetric = True, bit = 8, clamp_std = None,
                        th_point='max', th_scale=None, all_positive=False):
    # data_float -> Input data needs to be quantized
    # symmetric -> whether use symmetric quantized
    # bit -> quant bits
    # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]
    # th_point -> clamp data_float mode
    # th_scale -> scale the clamp thred, used together with th_point
    # all_positive -> whether data_float is all positive

    std = data_float.std()
    max_data = data_float.max()
    min_data = data_float.min()
    
    # print(f'max_data:{max_data.item()}, min data{min_data} th_point {th_point}')
    # exit()
    # if min_data.item() >= 0:
    #     all_positive = True

    if clamp_std != None and clamp_std != 0 and th_scale != None:
        raise ValueError("clamp_std and th_scale, only one clamp method can be used. ")
    if clamp_std != None and clamp_std != 0:
        data_float = torch.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)
    else:
        # print(f'max_data:{max_data.item()}, min data{min_data} th_point {th_point}')
        if min_data.item() * max_data.item() < 0. and th_point == 'min':
            th = min(max_data.abs().item(), min_data.abs().item())
        else:
            th = max(max_data.abs().item(), min_data.abs().item())
        if th_scale != None:
            th *= th_scale
        data_float = torch.clamp(data_float, min = -th, max = th)

    if all_positive:
        if data_float.min().item() < 0:
            raise ValueError("all_positive uniform_quantizer's data_float is not all positive. ")
        data_range = data_float.max()
        quant_range = 2**bit-1
        zero_point = 0
    elif symmetric:
        data_range = 2*abs(data_float).max()
        quant_range = 2**bit - 2
        zero_point = 0
    else:
        data_range = data_float.max() - data_float.min()
        quant_range = 2**bit - 1
        zero_point = data_float.min() / data_range * quant_range

    if data_range == 0:
        return data_float, 1

    scale = data_range / quant_range
    data_quantized = (round_pass(data_float / scale - zero_point) + zero_point) * scale

    if zero_point != 0:
        raise ValueError("asymmetric uniform quantizer can not be valid next step yet. ")

    return round_pass(data_float / scale - zero_point), scale.item()

# 均匀量化+noise quantizer
class uniform_quantizer(nn.Module):
    def __init__(self, symmetric=False, bit=4, clamp_std=0, th_point='max', th_scale=None, all_positive=False, noise_scale=0,
                noise_method='add', noise_range='max', int_flag=False, *args, **kwargs):
        # symmetric -> whether use symmetric quantized
        # bit -> quant bits
        # clamp_std -> Clamp data_float to [- std * clamp_std, std * clamp_std]
        # th_point -> clamp data_float mode
        # th_scale -> scale the clamp thred, used together with th_point
        # all_positive -> whether data_float is all positive
        # noise_scale -> noise scale
        # noise_method -> noise method, {'add', 'mul'}
        # noise_range -> noise range, activated when noise_method is 'add'
        super(uniform_quantizer, self).__init__()
        self.symmetric = symmetric
        self.bit = bit
        self.clamp_std = clamp_std
        self.th_point = th_point
        self.th_scale = th_scale
        self.all_positive = all_positive
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.int_flag = int_flag
    
    def forward(self, weight):
        weight_int, scale = data_quantization_int(weight, self.symmetric, self.bit, self.clamp_std,
                                        self.th_point, self.th_scale, self.all_positive)
        if self.noise_scale != 0:
            weight_int = add_noise(weight_int, self.noise_method, self.noise_scale, self.noise_range)
        if self.int_flag:
            return weight_int, torch.tensor(scale).cuda()
        else:
            return weight_int * scale
    
    def get_int(self, weight): # without noise
        return data_quantization_int(weight, self.symmetric, self.bit, self.clamp_std, 
                                        self.th_point, self.th_scale, self.all_positive)
    
    def get_quant_params(self):
        members= {}
        members['quant_name'] = 'uniform'
        members['bit'] = self.bit
        members['symmetric'] = self.symmetric
        members['clamp_std'] = self.clamp_std
        members['th_point'] = self.th_point
        members['th_scale'] = self.th_scale
        members['all_positive'] = self.all_positive
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members

# ********************* 二值(+-1) & 三值(+-1、0) ***********************
def clamp_params(w):
    w.data.clamp_(-1.0, 1.0)  # W截断
    return w

class Binary_weight_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input)
        output[output == 0] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Binary_act_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        if input.min().item() < 0:
            output[output == 0] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        return grad_input

# 三值(+-1、0)
class Ternary_weight_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, per_channel=True):
        # **************** channel级 - E(|W|) ****************
        input_shape = input.shape
        if per_channel:
            if len(input_shape) == 4:
                E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
            elif len(input_shape) == 2:
                E = torch.mean(torch.abs(input), 1, keepdim=True)
            else:
                raise ValueError("unexpected input of Ternary_weight_STE's shape {}".format(input.shape))
        else:
            E = torch.mean(torch.abs(input))
        # **************** 阈值 ****************
        threshold = E * 0.7
        # ************** W —— +-1、0 **************
        output = torch.sign(
            torch.add(
                torch.sign(torch.add(input, threshold)),
                torch.sign(torch.add(input, -threshold)),
            )
        )
        output[input.abs() == threshold] = 0
        return output, threshold

    @staticmethod
    def backward(ctx, grad_output, grad_threshold):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input, None

# 二/三值量化+noise weight quantizer
class Binary_weight_quantizer(nn.Module):
    def __init__(self, alpha=True, W = 2, w_clamp=True, per_channel=False, noise_scale=0, noise_method='add',
                    noise_range='max', int_flag=False, *args, **kwargs):
        # alphe -> whether rescale weight to weight_mean after sign function
        # W -> W=2, 二值化, W=3, 三值化 
        # center_clamp -> weight clamp to [-1,1] when W = 2
        # per_channel -> used when W is 3, true: channel wise, false: layer wise
        # noise_scale -> noise scale
        # noise_method -> noise method, {'add', 'mul'}
        # noise_range -> noise range, activated when noise_method is 'add'
        super(Binary_weight_quantizer, self).__init__()
        self.alpha = alpha
        self.W = W
        self.w_clamp = w_clamp
        self.per_channel = per_channel
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.int_flag = int_flag

    def forward(self, weight):
        if self.W == 2:
            if self.w_clamp:
                clamp_params(weight)
            w_shape = weight.shape  # [out_c, in_c, k, k] or [out_c, in_c]
            if self.alpha:
                alpha = weight.abs().view(w_shape[0], -1).mean(1, keepdim=True)
            else:
                alpha = torch.ones((w_shape[0], 1), device=weight.device)
            if len(w_shape) == 4:
                alpha = alpha.view(-1, 1, 1, 1)
            elif len(w_shape) != 2:
                raise ValueError("unexpected weight of Binary_weight_quantizer's shape {}".format(weight.shape))
            weight_int = Binary_weight_STE.apply(weight)
        elif self.W == 3:
            weight_cp = weight.clone()
            weight_int, threshold = Ternary_weight_STE.apply(weight, self.per_channel)
            # **************** α(缩放因子) ****************
            if self.alpha:
                weight_abs = torch.abs(weight_cp)
                mask_le = weight_abs.le(threshold) # 小于等于
                mask_gt = weight_abs.gt(threshold) # 大于
                weight_abs[mask_le] = 0
                weight_abs_th = weight_abs.clone()
                w_shape = weight_cp.shape
                if self.per_channel:
                    if len(w_shape) == 4:
                        weight_abs_th_sum = torch.sum(weight_abs_th, (3, 2, 1), keepdim=True)
                        mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                    elif len(w_shape) == 2:
                        weight_abs_th_sum = torch.sum(weight_abs_th, 1, keepdim=True)
                        mask_gt_sum = torch.sum(mask_gt, 1, keepdim=True).float()
                    else:
                        raise ValueError("unexpected weight of Binary_weight_quantizer's shape {}".format(weight_cp.shape))
                else:
                    weight_abs_th_sum = torch.sum(weight_abs_th)
                    mask_gt_sum = torch.sum(mask_gt).float()
                alpha = weight_abs_th_sum / mask_gt_sum  # α(缩放因子)
            else:
                alpha = torch.ones((weight_cp.shape[0], 1), device=weight_int.device)
                if len(weight_cp.shape) == 4:
                    alpha = alpha.view(-1, 1, 1, 1)
        else:
            raise ValueError("binary weight quantizer's W must set 2 or 3. ")

        if self.noise_scale != 0:
            weight_int = add_noise(weight_int, self.noise_method, self.noise_scale, self.noise_range)
        if self.int_flag:
            return weight_int, alpha
        else:
            return weight_int * alpha

    def get_int(self, weight):
        if self.W == 2:
            if self.w_clamp:
                clamp_params(weight)
            w_shape = weight.shape  # [out_c, in_c, k, k] or [out_c, in_c]
            if self.alpha:
                alpha = weight.abs().view(w_shape[0], -1).mean(1, keepdim=True)
            else:
                alpha = torch.ones((w_shape[0], 1), device=weight.device)
            if len(w_shape) == 4:
                alpha = alpha.view(-1, 1, 1, 1)
            elif len(w_shape) != 2:
                raise ValueError("unexpected weight of Binary_weight_quantizer's shape {}".format(weight.shape))
            weight_q = Binary_weight_STE.apply(weight)
            return weight_q, alpha.view(-1)[0].data.item()
        elif self.W == 3:
            weight_cp = weight.clone()
            weight_q, threshold = Ternary_weight_STE.apply(weight, self.per_channel)
            # **************** α(缩放因子) ****************
            if self.alpha:
                weight_abs = torch.abs(weight_cp)
                mask_le = weight_abs.le(threshold) # 小于等于
                mask_gt = weight_abs.gt(threshold) # 大于
                weight_abs[mask_le] = 0
                weight_abs_th = weight_abs.clone()
                w_shape = weight_cp.shape
                if self.per_channel:
                    if len(w_shape) == 4:
                        weight_abs_th_sum = torch.sum(weight_abs_th, (3, 2, 1), keepdim=True)
                        mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                    elif len(w_shape) == 2:
                        weight_abs_th_sum = torch.sum(weight_abs_th, 1, keepdim=True)
                        mask_gt_sum = torch.sum(mask_gt, 1, keepdim=True).float()
                    else:
                        raise ValueError("unexpected weight of Binary_weight_quantizer's shape {}".format(weight_cp.shape))
                else:
                    weight_abs_th_sum = torch.sum(weight_abs_th)
                    mask_gt_sum = torch.sum(mask_gt).float()
                alpha = weight_abs_th_sum / mask_gt_sum  # α(缩放因子)
            else:
                alpha = torch.ones((weight_cp.shape[0], 1), device=weight_q.device)
                if len(weight_cp.shape) == 4:
                    alpha = alpha.view(-1, 1, 1, 1)
            # *************** W * α ****************
            return weight_q, alpha.view(-1)[0].data.item()
        else:
            raise ValueError("binary weight quantizer's W must set 2 or 3. ")
    
    def get_quant_params(self):
        members= {}
        members['quant_name'] = 'binary'
        members['W'] = self.W
        members['alpha'] = self.alpha
        members['w_clamp'] = self.w_clamp
        members['per_channel'] = self.per_channel
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members

# 二值量化 act quantizer             
class Binary_act_quantizer(nn.Module):
    def __init__(self, W = 2, noise_scale=0, noise_method='add', noise_range='max', *args, **kwargs):
        super(Binary_act_quantizer, self).__init__()
        self.W = W
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range

    def forward(self, input):
        if self.W == 2:
            output = Binary_act_STE.apply(input)
            if self.noise_scale != 0:
                output = add_noise(output, self.noise_method, self.noise_scale, self.noise_range)
        else:
            raise ValueError("binary activation quantizer's W must set 2. ")
        return output
    
    def get_quant_params(self):
        members= {}
        members['quant_name'] = 'binary'
        members['W'] = self.W
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members
    
class Binary_act_STE_std(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, act_std):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        if input.min().item() < 0:
            raise ValueError("Binary_act_STE_std's input has data below 0! ")
        else:
            output[input < act_std * input.data.std()] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        return grad_input, None

# 二值量化 act quantizer add            
class Binary_act_quantizer_std(nn.Module):
    def __init__(self, W = 2, act_std=None, noise_scale=0, noise_method='add', noise_range='max', *args, **kwargs):
        super(Binary_act_quantizer_std, self).__init__()
        self.W = W
        self.act_std = act_std
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range

    def forward(self, input):
        if self.W == 2:
            output = Binary_act_STE_std.apply(input, self.act_std)
            if self.noise_scale != 0:
                output = add_noise(output, self.noise_method, self.noise_scale, self.noise_range)
        else:
            raise ValueError("binary activation quantizer's W must set 2. ")
        return output

class Binary_act_STE_th(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        if input.min().item() < 0:
            raise ValueError("Binary_act_STE_std's input has data below 0! ")
        else:
            output[input <= th] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        return grad_input, None

# 二值量化 act quantizer add            
class Binary_act_quantizer_th(nn.Module):
    def __init__(self, W = 2, th=None, noise_scale=0, noise_method='add', noise_range='max', *args, **kwargs):
        super(Binary_act_quantizer_th, self).__init__()
        self.W = W
        self.th = th
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range

    def forward(self, input):
        if self.W == 2:
            output = Binary_act_STE_th.apply(input, self.th)
            if self.noise_scale != 0:
                output = add_noise(output, self.noise_method, self.noise_scale, self.noise_range)
        else:
            raise ValueError("binary activation quantizer's W must set 2. ")
        return output

# ********************* LSQ（scale可学习，对称均匀量化&unsigned 量化） ***********************
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LSQ_weight_quantizer(nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=False, noise_scale=0,
                    noise_method='add', noise_range='max', s_init=2,
                    init_mode='origin', init_percent=0.95, int_flag=False, *args, **kwargs):
        # bit -> quant bits
        # all_positive -> set int_quant range to [0, 2**bit-1]
        # symmetric -> True: set int_quant range to [-(2**(bit-1)-1), 2**(bit-1)-1], False: set int_quant range to [-2**(bit-1), 2**(bit-1)-1]
        # per_channel -> channel wise quantizer or tensor wise
        # init_mode -> choice of {'origin', 'percent'}
        super(LSQ_weight_quantizer, self).__init__()

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric: # not full_range
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.tensor(1.0))
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.init_mode = init_mode
        self.s_init = s_init
        self.init_percent = init_percent
        self.int_flag = int_flag
        self.min_s = torch.tensor(1e-7)  # s > 0
    
    def init_scale(self, x, *args, **kwargs):
        if self.init_mode == 'origin':
            if self.per_channel:
                self.s = nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * self.s_init / (self.thd_pos ** 0.5))
            else:
                self.s = nn.Parameter(x.detach().abs().mean() * self.s_init / (self.thd_pos ** 0.5))
        elif self.init_mode == 'percent':
            if self.per_channel:
                raise ValueError('per_channel weight quant not supported yet. ')
            else:
                val, ind = torch.sort(x.detach().view(-1).abs())
                self.s = nn.Parameter(val[math.ceil(len(val)*self.init_percent) - 1] / self.thd_pos)
        else:
            raise ValueError('Unknown s init_mode {}. '.format(self.init_mode))

    def get_thred(self):
        return self.s.data.detach() * self.thd_pos

    def forward(self, x):
        if self.s < self.min_s:
            self.s.data = self.min_s.to(self.s.device)
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x_int = round_pass(x)
        if self.noise_scale != 0:
            x_int = add_noise(x_int, self.noise_method, self.noise_scale, self.noise_range)
        if self.int_flag:
            return x_int, s_scale
        else:
            return x_int * s_scale

    def get_int(self, x):
        x = x / self.s
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        return x, self.s.item()

class LSQ_weight_quantizer_1(nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=False, noise_scale = 0,
                    noise_method='add', noise_range='max', s_init=2,
                    init_mode='origin', init_percent=0.95, int_flag=False, *args, **kwargs):
        # bit -> quant bits
        # all_positive -> set int_quant range to [0, 2**bit-1]
        # symmetric -> True: set int_quant range to [-(2**(bit-1)-1), 2**(bit-1)-1], False: set int_quant range to [-2**(bit-1), 2**(bit-1)-1]
        # per_channel -> channel wise quantizer or tensor wise
        super(LSQ_weight_quantizer_1, self).__init__()

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric: # not full_range
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 2
                self.thd_pos = 2 ** (bit - 1) - 2
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.tensor(1.0))
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.init_mode =init_mode
        self.s_init = s_init
        self.init_percent = init_percent
        self.int_flag = int_flag
        self.min_s = torch.tensor(1e-7)
    
    def init_scale(self, x, *args, **kwargs):
        if self.init_mode == 'origin':
            if self.per_channel:
                self.s = nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * self.s_init / (self.thd_pos ** 0.5))
            else:
                self.s = nn.Parameter(x.detach().abs().mean() * self.s_init / (self.thd_pos ** 0.5))
        elif self.init_mode == 'percent':
            if self.per_channel:
                raise ValueError('per_channel weight quant not supported yet. ')
            else:
                val, ind = torch.sort(x.detach().view(-1).abs())
                self.s = nn.Parameter(val[math.ceil(len(val)*self.init_percent) - 1] / self.thd_pos)
        else:
            raise ValueError('Unknown s init_mode {}. '.format(self.init_mode))
    
    def get_thred(self):
        return self.s.data.detach() * self.thd_pos

    def forward(self, x):
        if self.s < self.min_s:
            self.s.data = self.min_s.to(self.s.device)
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x_int = round_pass(x)
        if self.noise_scale != 0:
            x_int = add_noise(x_int, self.noise_method, self.noise_scale, self.noise_range)
        if self.int_flag:
            return x_int, s_scale
        else:
            return x_int * s_scale

    def get_int(self, x):
        x = x / self.s
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        return x, self.s.item()

class LSQ_act_quantizer(nn.Module):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=False,
                 noise_scale=0, noise_method='add', noise_range='max', s_init=2,
                 init_mode='origin', init_percent=0.95, int_flag=False, *args, **kwargs):
        super(LSQ_act_quantizer, self).__init__()

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.bit = bit
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.tensor(1.0))
        self.init_batch_mode = False
        self.init_batch_num = 0
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.init_mode = init_mode
        self.s_init = s_init
        self.init_percent = init_percent
        self.int_flag = int_flag
        self.min_s = torch.tensor(1e-7)

    def init_scale(self, x):
        if self.init_mode == 'origin':
            if self.per_channel:
                self.s = nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * self.s_init / (self.thd_pos ** 0.5))
            else:
                self.s = nn.Parameter(x.detach().abs().mean() * self.s_init / (self.thd_pos ** 0.5))
        elif self.init_mode == 'percent':
            if self.per_channel:
                raise ValueError('per_channel weight quant not supported yet. ')
            else:
                val, ind = torch.sort(x.detach().view(-1).abs())
                self.s = nn.Parameter(val[math.ceil(len(val)*self.init_percent) - 1] / self.thd_pos)
        else:
            raise ValueError('Unknown s init_mode {}. '.format(self.init_mode))
        
    def get_scale(self):
        return self.s.data.detach()

    def get_thred(self):
        return self.s.data.detach() * self.thd_pos
        
    def forward(self, x):
        if self.init_batch_mode:
            self.init_batch_num += 1
            self.init_scale(x.clone())
        if self.s < self.min_s:
            self.s.data = self.min_s.to(self.s.device)
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x_int = round_pass(x)
        if self.noise_scale != 0:
            x_int = add_noise(x_int, self.noise_method, self.noise_scale, self.noise_range)
        if self.int_flag:
            return x_int, s_scale
        else:
            return x_int * s_scale
    
    def get_quant_params(self):
        members= {}
        members['quant_name'] = 'lsq'
        members['bit'] = self.bit
        members['thd_neg'] = self.thd_neg
        members['thd_pos'] = self.thd_pos
        members['s'] = self.s.data.cpu().numpy()
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members
    
# this module only have noise function
class NoQuan(nn.Module):
    def __init__(self, bit=None, noise_scale = 0, noise_method='add', noise_range='max',
                 int_flag=False, *args, **kwargs):
        super(NoQuan, self).__init__()
        # assert bit is None, 'The bit-width of identity quantizer must be None'
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
        self.int_flag = int_flag

    def forward(self, x):
        if self.noise_scale != 0:
            x = add_noise(x, self.noise_method, self.noise_scale, self.noise_range)
        if self.int_flag:
            # return x, 0
            raise ValueError("NoQuan cannot support int_flag:True. ")
        else:
            return x

    def get_int(self, x):
        return x, 1
    
    def get_quant_params(self):
        members = {}
        members['quant_name'] = 'None'
        members['noise_scale'] = self.noise_scale
        members['noise_method'] = self.noise_method
        members['noise_range'] = self.noise_range
        return members

# *********************** bias quan ***********************
def add_noise_bias_rows(weight, b_max=0, rows=1, n_scale=0):
    if n_scale == 0 or b_max == 0:
        return weight
    
    factor = b_max
    w_noise = torch.randn_like(weight, device=weight.device).clamp_(-3.0, 3.0) * factor * n_scale * (rows ** 0.5)
    weight_noise = weight + w_noise
    
    weight_noise = (weight_noise - weight).detach() + weight
    return weight_noise
class NoQuan_bias(nn.Module):
    def __init__(self, bit=None, noise_scale=0, noise_method='add', noise_range='max', *args, **kwargs):
        super(NoQuan_bias, self).__init__()
        assert bit is None, 'The bit-width of identity quantizer must be None'
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
    
    def forward(self, x, s):
        if self.noise_scale != 0:
            x = add_noise(x, self.noise_method, self.noise_scale, self.noise_range)
        return x
    
    def get_int(self, x):
        return x, 1

class Bias_quantizer(nn.Module):
    def __init__(self, bit=None, bias_input=1, all_positive=False, symmetric=True, noise_scale=0,
                 noise_method='add', noise_range='max'):
        super(Bias_quantizer, self).__init__()
        self.bit = bit
        self.bias_input = bias_input
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
        
        self.noise_scale = noise_scale
        self.noise_method = noise_method
        self.noise_range = noise_range
    
    def forward(self, x, scale):
        scale = scale * self.bias_input
        x = round_pass(torch.clamp(x / scale, self.thd_neg, self.thd_pos))
        x = x * scale
        if self.noise_scale != 0:
            x = add_noise(x, self.noise_method, self.noise_scale, self.noise_range)
        return x
    
    def get_int(self, x, scale):
        scale = scale * self.bias_input
        x = round_pass(torch.clamp(x / scale, self.thd_neg, self.thd_pos))
        return x, scale.item()

class Bias_quantizer_rows(nn.Module):
    def __init__(self, bit=None, rows=1, bias_input=1, all_positive=False, symmetric=True, int_flag=False, noise_scale=0):
        super(Bias_quantizer_rows, self).__init__()
        self.bit = bit
        self.rows = rows
        self.bias_input = bias_input
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
        
        self.noise_scale = noise_scale
        # self.noise_method = noise_method
        # self.noise_range = noise_range
        self.int_flag = int_flag
    
    def forward(self, x, scale):
        scale = scale * self.bias_input
        x_int = round_pass(torch.clamp(x / scale, self.thd_neg * self.rows, self.thd_pos * self.rows))
        if self.noise_scale != 0:
            x_int = add_noise_bias_rows(x_int, b_max=self.thd_pos, rows=self.rows, n_scale=self.noise_scale)
        if self.int_flag:
            return x_int, scale.detach()
        else:
            return x_int * scale.detach()
    
    def get_int(self, x, scale):
        scale = scale * self.bias_input
        x_int = round_pass(torch.clamp(x / scale, self.thd_neg * self.rows, self.thd_pos * self.rows))
        return x_int, scale.item()
    
if __name__ == "__main__":
    x = torch.randn(5, 5)
    x_q = data_quantization(x, 1)
    print(x)
    print(x_q)
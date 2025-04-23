from .quant_layer import *
from .quant_layer_backwonoise import QuanModuleMappingBackWoNoise
import munch
from .quant_util import *
import logging
logger = logging.getLogger()

def get_target_cfg(default_cfg, this_cfg=None): # input output is munch.Munch
    target_cfg = munch.Munch(default_cfg)
    if this_cfg is None:
        return target_cfg
    for k in this_cfg:
        if target_cfg.get(k, None) is not None and type(this_cfg[k]) == munch.Munch:
            target_cfg[k] = get_target_cfg(default_cfg[k], this_cfg[k])
        else:
            target_cfg[k] = this_cfg[k]
    return target_cfg

def get_weight_quantizer(cfg):
    target_cfg = dict(cfg)
    if not 'quant_name' in target_cfg or target_cfg['quant_name'] is None:
        q = NoQuan
    elif target_cfg['quant_name'] == 'uniform':
        q = uniform_quantizer
    elif target_cfg['quant_name'] == 'binary':
        q = Binary_weight_quantizer
    elif target_cfg['quant_name'] == 'lsq':
        q = LSQ_weight_quantizer
    elif target_cfg['quant_name'] == 'lsq_1':
        q = LSQ_weight_quantizer_1
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['quant_name'])

    target_cfg.pop('quant_name')
    return q(**target_cfg)
def get_bias_quantizer(cfg):
    target_cfg = dict(cfg)
    if not 'quant_name' in target_cfg or target_cfg['quant_name'] is None:
        q = NoQuan
    elif target_cfg['quant_name'] == 'fixed_scale':
        q = Bias_quantizer_rows
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['quant_name'])

    target_cfg.pop('quant_name')
    return q(**target_cfg)
def get_act_quantizer(cfg):
    target_cfg = dict(cfg)
    if not 'quant_name' in target_cfg or target_cfg['quant_name'] is None:
        q = NoQuan
    elif target_cfg['quant_name'] == 'uniform':
        q = uniform_quantizer
    elif target_cfg['quant_name'] == 'binary':
        return Binary_act_quantizer()
    elif target_cfg['quant_name'] == 'binary_std':
        q = Binary_act_quantizer_std
    elif target_cfg['quant_name'] == 'binary_th':
        q = Binary_act_quantizer_th
    elif target_cfg['quant_name'] == 'lsq':
        q = LSQ_act_quantizer
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['quant_name'])
        
    target_cfg.pop('quant_name')
    return q(**target_cfg)

def find_modules_to_quantize(model, quan_scheduler, QuanModuleMapping):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if quan_scheduler.excepts is not None and name in quan_scheduler.excepts:
                target_cfg = get_target_cfg(quan_scheduler, quan_scheduler.excepts[name])
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    w_quantizer=get_weight_quantizer(target_cfg.weight),
                    a_quantizer=get_act_quantizer(target_cfg.act),
                    a_out_quantizer=get_act_quantizer(target_cfg.act_out)
                )
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    w_quantizer=get_weight_quantizer(quan_scheduler.weight),
                    a_quantizer=get_act_quantizer(quan_scheduler.act),
                    a_out_quantizer=get_act_quantizer(quan_scheduler.act_out)
                )
        elif quan_scheduler.excepts is not None and name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace, QuanModuleMapping):
    def helper(child:nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name)) # 模块替换操作
                        break
            else:
                helper(c)

    helper(model)
    return model
def find_modules_to_quantize2(model, quan_args, QuanModuleMapping):
    replaced_modules = dict()
    if (not 'conv' in quan_args) or (not 'fc' in quan_args):
        raise ValueError("can not find 'conv' or 'fc' key for quan_args. ")
    mapping_name = ['avgpool', 'bn', 'other']
    for m_n in mapping_name:
        if not m_n in quan_args:
            logger.warning("can not find '{}' key for qan_args, it will use default setting as "
                          "quant_flag: False. ".format(m_n))
    for name, module in model.named_modules():
        if not type(module) in totalMappingModule:
            continue
        if type(module) in QuanModuleMapping:
            layer_type = 'conv' if isinstance(module, nn.Conv2d) else 'fc'
            if ('excepts' in quan_args) and (quan_args.excepts is not None) and (name in quan_args.excepts):
                target_cfg = get_target_cfg(quan_args[layer_type], quan_args.excepts[name])
            else:
                target_cfg = munch.Munch(quan_args[layer_type])
            quant_module = QuanModuleMapping[type(module)](
                module,
                w_quantizer=get_weight_quantizer(
                    target_cfg.weight if 'weight' in target_cfg else {}),
                a_quantizer=get_act_quantizer(target_cfg.act if 'act' in target_cfg else {}),
                a_out_quantizer=get_act_quantizer(
                    target_cfg.act_out if 'act_out' in target_cfg else {}),
                int_flag = target_cfg.int_flag if 'int_flag' in target_cfg else False
            )
            replaced_modules[name] = quant_module
            logger.info("{} layer {} switched to quant_{} with args {}. ".format(layer_type,
                                                                                 name, layer_type, target_cfg))
        elif type(module) in BnMapping:
            if ('excepts' in quan_args) and (quan_args.excepts is not None) and (name in quan_args.excepts):
                target_cfg = get_target_cfg(quan_args.bn if 'bn' in quan_args else munch.Munch(),
                                            quan_args.excepts[name])
            else:
                target_cfg = munch.Munch(quan_args.bn) if 'bn' in quan_args else {}
            logger.info("BN layer {} switched to quant_BN with args {}. ".format(name, target_cfg))
            if 'quant_flag' in target_cfg and target_cfg.quant_flag:
                assert target_cfg.get('weight', None) is not None and \
                       target_cfg.weight.get('quant_name', None) is not None
                assert target_cfg.get('bias', None) is not None and \
                       target_cfg.bias.get('quant_name', None) is not None
                assert target_cfg.get('act_out', None) is not None and \
                       target_cfg.act_out.get('quant_name', None) is not None
                replaced_modules[name] = BnMapping[type(module)](m=module,
                                                                 w_quantizer=get_weight_quantizer(target_cfg.pop('weight')),
                                                                 bias_quantizer=get_bias_quantizer(target_cfg.pop('bias')),
                                                                 a_out_quantizer=get_act_quantizer(target_cfg.pop('act_out')),
                                                                 **dict(target_cfg))
            else:
                replaced_modules[name] = BnMapping[type(module)](m=module)
        elif type(module) in AvgMapping:
            if ('excepts' in quan_args) and (quan_args.excepts is not None) and (name in quan_args.excepts):
                target_cfg = get_target_cfg(quan_args.avgpool if 'avgpool' in quan_args else munch.Munch(),
                                            quan_args.excepts[name])
            else:
                target_cfg = munch.Munch(quan_args.avgpool) if 'avgpool' in quan_args else {}
            replaced_modules[name] = AvgMapping[type(module)](m=module, **dict(target_cfg))
            logger.info("Avg layer {} switched to quant_Avg with args {}. ".format(name, target_cfg))
        else:  # other module
            replaced_modules[name] = OtherMapping[type(module)](m=module)

    return replaced_modules

def replace_module_by_names2(model, modules_to_replace, QuanModuleMapping):
    def helper(child: nn.Module):
        for n, c in child.named_children():
            if type(c) in totalMappingModule:
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name)) # 模块替换操作
                        break
            else:
                helper(c)

    helper(model)
    return model

def prepare_quant_model(
    model,
    train_loader,
    quan_args,
    ):
    # model: float32 model to be quantized
    # train_loader: train_data may be used to act initial
    # quan_args: quantizer args
    if quan_args.backwonoise:
        modules_to_replace = find_modules_to_quantize(model, quan_args, QuanModuleMappingBackWoNoise)
        model = replace_module_by_names(model, modules_to_replace, QuanModuleMappingBackWoNoise)
    else:
        modules_to_replace = find_modules_to_quantize(model, quan_args, QuanModuleMapping)
        model = replace_module_by_names(model, modules_to_replace, QuanModuleMapping)
    
    if quan_args.init_batch:
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = True
                print(name)
        for batch_idx, (inputs, targets) in enumerate(train_loader): ##
            if batch_idx >= quan_args.init_batch_num:
                break
            output = model(inputs)
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = False
    return model

def prepare_quant_model2(
    model,
    train_loader,
    quan_args,
    ):
    # model: float32 model to be quantized
    # train_loader: train_data may be used to act initial
    # quan_args: quantizer args
    modules_to_replace = find_modules_to_quantize2(model, quan_args)
    model = replace_module_by_names2(model, modules_to_replace)
    # model.eval()
    if quan_args.init_batch:
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = True
                print("Quantizer {} set init_batch_mode True. ".format(name))
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= quan_args.init_batch_num:
                break
            output = model(inputs)
        for name, module in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = False
                print("Quantizer {} set init_batch_mode False. ".format(name))
    return model

# def prepare_quant_model(
#     model,
#     quant_dict,
#     ):
#     # model: float32 model to be quantized
#     # quant_dict: define the layers to be quantized and noised, {'layer_name': {'w_quant_way': {}, 'act_quant_way': {}, 'w_noise_way': {}}}}
#     for name, layer_module in model.named_modules():
#         if name in quant_dict:
#             if isinstance(layer_module, nn.Conv2d):
#                 quant_conv = conv2d_quant_noise(
#                     layer_module,
#                     w_quant_way=quant_dict[name]['w_quant_way'],
#                     a_quant_way=quant_dict[name]['a_quant_way'],
#                 )
#                 model._modules[name] = quant_conv
#             elif isinstance(layer_module, nn.Linear):
#                 quant_linear = linear_quant_noise(
#                     layer_module,
#                     w_quant_way=quant_dict[name]['w_quant_way'],
#                     a_quant_way=quant_dict[name]['a_quant_way'],
#                 )
#                 model._modules[name] = quant_linear
#             else:
#                 raise ValueError("layer {} can not be quantizes yet! ".format(name))
#     return model

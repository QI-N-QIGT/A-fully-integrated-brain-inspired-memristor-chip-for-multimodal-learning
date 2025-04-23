from .quant_layer import *
from .quant_layer_backwonoise import QuanModuleMappingBackWoNoise
from .quant_util import *
import logging
logger = logging.getLogger()

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
    for (name, module) in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if quan_scheduler.excepts is not None and name in quan_scheduler.excepts:
                target_cfg = get_target_cfg(quan_scheduler, quan_scheduler.excepts[name])
                replaced_modules[name] = QuanModuleMapping[type(module)](module, w_quantizer=get_weight_quantizer(target_cfg.weight), a_quantizer=get_act_quantizer(target_cfg.act), a_out_quantizer=get_act_quantizer(target_cfg.act_out))
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](module, w_quantizer=get_weight_quantizer(quan_scheduler.weight), a_quantizer=get_act_quantizer(quan_scheduler.act), a_out_quantizer=get_act_quantizer(quan_scheduler.act_out))
        elif quan_scheduler.excepts is not None and name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)
    return replaced_modules

def replace_module_by_names(model, modules_to_replace, QuanModuleMapping):

    def helper(child: nn.Module):
        for (n, c) in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for (full_name, m) in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)
    helper(model)
    return model

def replace_module_by_names2(model, modules_to_replace, QuanModuleMapping):

    def helper(child: nn.Module):
        for (n, c) in child.named_children():
            if type(c) in totalMappingModule:
                for (full_name, m) in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)
    helper(model)
    return model

def prepare_quant_model(model, train_loader, quan_args):
    if quan_args.backwonoise:
        modules_to_replace = find_modules_to_quantize(model, quan_args, QuanModuleMappingBackWoNoise)
        model = replace_module_by_names(model, modules_to_replace, QuanModuleMappingBackWoNoise)
    else:
        modules_to_replace = find_modules_to_quantize(model, quan_args, QuanModuleMapping)
        model = replace_module_by_names(model, modules_to_replace, QuanModuleMapping)
    if quan_args.init_batch:
        for (name, module) in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = True
                print()
        for (batch_idx, (inputs, targets)) in enumerate(train_loader):
            if batch_idx >= quan_args.init_batch_num:
                break
            output = model(inputs)
        for (name, module) in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = False
    return model

def prepare_quant_model2(model, train_loader, quan_args):
    modules_to_replace = find_modules_to_quantize2(model, quan_args)
    model = replace_module_by_names2(model, modules_to_replace)
    if quan_args.init_batch:
        for (name, module) in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = True
                print()
        for (batch_idx, (inputs, targets)) in enumerate(train_loader):
            if batch_idx >= quan_args.init_batch_num:
                break
            output = model(inputs)
        for (name, module) in model.named_modules():
            if isinstance(module, LSQ_act_quantizer):
                module.init_batch_mode = False
                print()
    return model
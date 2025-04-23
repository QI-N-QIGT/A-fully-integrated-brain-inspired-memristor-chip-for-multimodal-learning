import torch
import torch.nn as nn
import torch.fx as fx
import operator
from .util import get_act_quantizer
from .quant_layer import add_quant
from .quant_function import funcmapping
import logging
logger = logging.getLogger()

def bias_move(m):
    for (name, module) in m.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.bias = None
            logger.info('layer {} bias set None. '.format(name))
    return m

def get_add_function_name(m: nn.Module, tracer_class: type=torch.fx.Tracer):
    graph: fx.Graph = tracer_class().trace(m)
    add_name = []
    for node in graph.nodes:
        if node.op == 'call_function' and node.target == operator.add:
            add_name.append(node.name)
    return add_name

def graph_transform(m: nn.Module, graph_args, tracer_class: type=torch.fx.Tracer):
    graph: fx.Graph = tracer_class().trace(m)
    if 'func_quant' in graph_args and graph_args.func_quant:
        add_name = get_add_function_name(m)
        add_target_d = {}
        i = 1
        for name in add_name:
            m.add_module('add_quant_{}'.format(i), add_quant(**dict(graph_args.add_quant) if 'add_quant' in graph_args else {}))
            add_target_d[name] = 'add_quant_{}'.format(i)
            i += 1
        for node in graph.nodes:
            if node.op == 'call_function':
                if not node.target == operator.add:
                    if not node.target in funcmapping:
                        raise ValueError('function {} for quant model not supported yet. '.format(node.target))
                    target_old = node.target
                    node.target = funcmapping[target_old]
                    logger.info('switch function {} to quant function '.format(target_old))
                else:
                    logger.info('switch add_func {} to add_quant {}'.format(node.name, add_target_d[node.name]))
                    node.op = 'call_module'
                    node.target = add_target_d[node.name]
    if 'insert_quantizer' in graph_args and graph_args.insert_quantizer is not None:
        quant_layer = get_act_quantizer(graph_args.insert_quantizer.quant_args)
        m.add_module(graph_args.insert_quantizer.module_name, quant_layer)
        for node in graph.nodes:
            if node.target == graph_args.insert_quantizer.insert_layer:
                graph.inserting_after(node)
                insert_node = graph.create_node('call_module', target=graph_args.insert_quantizer.module_name, args=(node,), name=graph_args.insert_quantizer.module_name)
                logger.info('Insert {} quantizer after layer {}. '.format(graph_args.insert_quantizer.quant_args.quant_name, graph_args.insert_quantizer.insert_layer))
                for n in graph.nodes:
                    for (ind, name) in enumerate(n.args):
                        if name == node:
                            if n.name != insert_node.name:
                                tmp_args = [n.args]
                                tmp_args[ind] = insert_node
                                n.args = tuple(tmp_args)
                break
    graph.lint()
    return fx.GraphModule(m, graph)
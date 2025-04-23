import copy
import numpy as np
import warnings

def mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info, mesh_height=6, mesh_width=6, alpha=0, pe_bind_direction=False, method='random', mapped_node_history=None):
    if name in layer_xb_mapped_node.keys():
        parent_node.append(layer_xb_mapped_node[name])
    else:
        parent_node_split = []
        for name_ in layer_ref[name]:
            if name_ in layer_xb_mapped_node.keys():
                parent_node_split.append(layer_xb_mapped_node[name_])
            elif 'split' in name_.lower() or 'concat' in name_.lower() or 'add' in name_.lower():
                mapping_noncim_nodes(name_, layer_ref, available_nodes_xb, parent_node_split, layer_xb_mapped_node, mapping_info, method=method, mapped_node_history=mapped_node_history)
            else:
                warnings.warn(f' 当前层{name} 的前一层 {name_} 暂未mapping对应的硬件 !!!')
        parent_core_id = []
        record_io_workload_parent_total = {}
        for node in parent_node_split:
            if isinstance(node, MappedLayerNode):
                if node.record_io_workload_total != None:
                    for (k, v) in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                core_id = '.'.join(node.addr_node.split('.')[:2])
                if core_id not in parent_core_id:
                    parent_core_id.append(core_id)
        parent_available_nodes_xb = []
        for n in available_nodes_xb:
            x = '.'.join(n.split('.')[:2])
            if x not in parent_core_id:
                parent_available_nodes_xb.append(n)
        if parent_available_nodes_xb == []:
            warnings.warn(f'当前层 {name} 的父节点已占用所有可部署的不同core id, 因此需要在输出位置生成中转线程 !!!')
            parent_available_nodes_xb = available_nodes_xb
        len_ = len(parent_available_nodes_xb)
        if method.lower() == 'random':
            rd_index = np.random.randint(0, len_)
            device_ref = parent_available_nodes_xb[rd_index]
        elif method.lower() == 'onebyone':
            assert len_ >= 1
            device_ref = parent_available_nodes_xb[0]
        elif method.lower() == 'a_search':
            assert mapped_node_history != None
            device_ref = None
            possible_mapping_nodes = []
            for nd in parent_available_nodes_xb:
                node_location = '.'.join(nd.split('.')[:2])
                if node_location not in mapped_node_history:
                    possible_mapping_nodes.append(nd)
            if possible_mapping_nodes != []:
                index = 0
                device_ref = possible_mapping_nodes[index]
                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
            if device_ref == None:
                loc_ = [mapped_node_history.index('.'.join(x.split('.')[:2])) for x in parent_available_nodes_xb]
                oldest_index_ = np.argmin(np.array(loc_))
                device_ref = parent_available_nodes_xb[oldest_index_]
                mapped_node_history.pop(loc_[oldest_index_])
                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
            assert device_ref != None
        else:
            raise ValueError(f'暂不支持的方法: {method} !!!')
        addr = [0]
        child_node = MappedLayerNode(name, device_ref, addr, parent=parent_node_split, available_nodes_xb=parent_available_nodes_xb, record_io_workload_total=copy.deepcopy(record_io_workload_parent_total), MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
        (distance_parent, record_io_workload_parent) = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
        (distance_out, record_io_workload_out) = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
        child_node.record_io_workload_parent = record_io_workload_parent
        child_node.record_io_workload_out = record_io_workload_out
        child_node.to_parent_cost = distance_parent
        child_node.to_out_cost = distance_out
        child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]
        for (n, v) in child_node.record_io_workload_parent.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v
        parent_node.append(child_node)
        layer_xb_mapped_node[name] = child_node
        mapping_info[name] = device_ref

def Workload_balance_search(layer_ref, placed_nodes, available_nodes_xb, node_info, mesh_height=6, mesh_width=6, alpha=0, pe_bind_direction=False):
    count = 0
    record_io_workload = {}
    layer_xb_mapped_node = {}
    mapping_info = {}
    mapped_pe_cluster_history = []
    mapped_node_history = []
    record_node_compute_workload = {}
    for n in available_nodes_xb:
        pe_cluster = '.'.join(n.split('.')[:3])
        if pe_cluster not in record_node_compute_workload.keys():
            record_node_compute_workload[pe_cluster] = 0
    record_layer_compute_workload = {}
    for (node_name, split_info) in placed_nodes.items():
        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []
            assert layer_name in node_info.keys(), f'{layer_name} 不在 node info中 !!!'
            layer_info = node_info[layer_name]
            record_layer_compute_workload[layer_name] = layer_info['calc_num']
            if count != 0:
                parent_layer_name = layer_ref[layer_name]
                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=pe_bind_direction, method='A_search', mapped_node_history=mapped_node_history)
            if count == 0 or 'IN' in parent_node:
                if 'IN' in parent_node:
                    assert len(parent_node) == 1
                rd_index = 0
                device_ref = available_nodes_xb[rd_index]
                print()
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, available_nodes_xb=available_nodes_xb, record_io_workload_total=record_io_workload, MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                count += 1
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
                mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
                record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]
            else:
                parent_core_id = []
                record_io_workload_parent_total = {}
                for node in parent_node:
                    for (k, v) in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])
                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)
                device_ref = None
                possible_mapped_pe_cluster_nodes = []
                possible_mapped_nodes = []
                for nd in parent_available_nodes_xb:
                    pe_cluster_location = '.'.join(nd.split('.')[:3])
                    node_location = '.'.join(nd.split('.')[:2])
                    if node_location not in mapped_node_history:
                        possible_mapped_nodes.append(nd)
                    if pe_cluster_location not in mapped_pe_cluster_history:
                        possible_mapped_pe_cluster_nodes.append(nd)
                if possible_mapped_nodes != []:
                    index = 0
                    device_ref = possible_mapped_nodes[index]
                    mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]
                elif possible_mapped_pe_cluster_nodes != []:
                    oldest_access_node = mapped_node_history[0]
                    possible_mapped_pe_cluster_oldest_access = []
                    for nd in possible_mapped_pe_cluster_nodes:
                        if '.'.join(nd.split('.')[:2]) == oldest_access_node:
                            possible_mapped_pe_cluster_oldest_access.append(nd)
                    if possible_mapped_pe_cluster_oldest_access != []:
                        compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in possible_mapped_pe_cluster_oldest_access]
                        workload_minimum_index = np.argmin(np.array(compute_workload_node))
                        device_ref = possible_mapped_pe_cluster_oldest_access[workload_minimum_index]
                    else:
                        compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in possible_mapped_pe_cluster_nodes]
                        workload_minimum_index = np.argmin(np.array(compute_workload_node))
                        device_ref = possible_mapped_pe_cluster_nodes[workload_minimum_index]
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                    mapped_node_history.pop(0)
                    mapped_node_history.append(oldest_access_node)
                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]
                if device_ref == None:
                    compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in parent_available_nodes_xb]
                    workload_minimum_index = np.argmin(np.array(compute_workload_node))
                    device_ref = parent_available_nodes_xb[workload_minimum_index]
                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]
                assert device_ref != None
                available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent=parent_node, available_nodes_xb=parent_available_nodes_xb, record_io_workload_total=copy.deepcopy(record_io_workload_parent_total), MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                (distance_parent, record_io_workload_parent) = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                (distance_out, record_io_workload_out) = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]
                for (n, v) in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node
    node_mapping_info_list = []
    record_io_workload = []
    if child_node.record_io_workload_out != None:
        for (n, v) in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v
    record_io_workload = child_node.record_io_workload_total
    node_mapping_info_list = mapping_info
    return (node_mapping_info_list, record_io_workload)

def A_search(layer_ref, placed_nodes, available_nodes_xb, mesh_height=6, mesh_width=6, alpha=0, pe_bind_direction=False):
    count = 0
    record_io_workload = {}
    layer_xb_mapped_node = {}
    mapping_info = {}
    mapped_pe_cluster_history = []
    mapped_node_history = []
    for (node_name, split_info) in placed_nodes.items():
        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []
            if count != 0:
                parent_layer_name = layer_ref[layer_name]
                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=pe_bind_direction, method='A_search', mapped_node_history=mapped_node_history)
            if count == 0 or 'IN' in parent_node:
                if 'IN' in parent_node:
                    assert len(parent_node) == 1
                rd_index = 0
                device_ref = available_nodes_xb[rd_index]
                print()
                available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, available_nodes_xb=available_nodes_xb, record_io_workload_total=record_io_workload, MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                count += 1
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
                mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
            else:
                parent_core_id = []
                record_io_workload_parent_total = {}
                for node in parent_node:
                    for (k, v) in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])
                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)
                device_ref = None
                possible_mapped_pe_cluster_nodes = []
                possible_mapped_nodes = []
                for nd in parent_available_nodes_xb:
                    pe_cluster_location = '.'.join(nd.split('.')[:3])
                    node_location = '.'.join(nd.split('.')[:2])
                    if node_location not in mapped_node_history:
                        possible_mapped_nodes.append(nd)
                    if pe_cluster_location not in mapped_pe_cluster_history:
                        possible_mapped_pe_cluster_nodes.append(nd)
                if possible_mapped_nodes != []:
                    index = 0
                    device_ref = possible_mapped_nodes[index]
                    mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                elif possible_mapped_pe_cluster_nodes != []:
                    loc_ = [mapped_node_history.index('.'.join(x.split('.')[:2])) for x in possible_mapped_pe_cluster_nodes]
                    oldest_index_ = np.argmin(np.array(loc_))
                    oldest_access_node = mapped_node_history[loc_[oldest_index_]]
                    possible_mapped_pe_cluster_oldest_access = []
                    for nd in possible_mapped_pe_cluster_nodes:
                        if '.'.join(nd.split('.')[:2]) == oldest_access_node:
                            possible_mapped_pe_cluster_oldest_access.append(nd)
                    if possible_mapped_pe_cluster_oldest_access != []:
                        index = np.random.randint(0, len(possible_mapped_pe_cluster_oldest_access))
                        device_ref = possible_mapped_pe_cluster_oldest_access[index]
                        mapped_node_history.pop(0)
                        mapped_node_history.append(oldest_access_node)
                    else:
                        index = np.random.randint(0, len(possible_mapped_pe_cluster_nodes))
                        device_ref = possible_mapped_pe_cluster_nodes[index]
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                if device_ref == None:
                    loc_ = [mapped_pe_cluster_history.index('.'.join(x.split('.')[:3])) for x in parent_available_nodes_xb]
                    oldest_index_ = np.argmin(np.array(loc_))
                    device_ref = parent_available_nodes_xb[oldest_index_]
                    mapped_pe_cluster_history.pop(loc_[oldest_index_])
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                assert device_ref != None
                available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent=parent_node, available_nodes_xb=parent_available_nodes_xb, record_io_workload_total=copy.deepcopy(record_io_workload_parent_total), MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                (distance_parent, record_io_workload_parent) = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                (distance_out, record_io_workload_out) = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]
                for (n, v) in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node
    node_mapping_info_list = []
    record_io_workload = []
    if child_node.record_io_workload_out != None:
        for (n, v) in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v
    record_io_workload = child_node.record_io_workload_total
    node_mapping_info_list = mapping_info
    return (node_mapping_info_list, record_io_workload)

def random_search(layer_ref, placed_nodes, available_nodes_xb, mesh_height=6, mesh_width=6, alpha=0, pe_bind_direction=False):
    count = 0
    record_io_workload = {}
    layer_xb_mapped_node = {}
    mapping_info = {}
    for (node_name, split_info) in placed_nodes.items():
        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []
            if count != 0:
                parent_layer_name = layer_ref[layer_name]
                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=pe_bind_direction)
            if count == 0 or 'IN' in parent_node:
                if 'IN' in parent_node:
                    assert len(parent_node) == 1
                len_ = len(available_nodes_xb)
                rd_index = np.random.randint(0, len_)
                device_ref = available_nodes_xb[rd_index]
                available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, available_nodes_xb=available_nodes_xb, record_io_workload_total=record_io_workload, MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                count += 1
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            else:
                parent_core_id = []
                record_io_workload_parent_total = {}
                for node in parent_node:
                    for (k, v) in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])
                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)
                len_ = len(parent_available_nodes_xb)
                rd_index = np.random.randint(0, len_)
                device_ref = parent_available_nodes_xb[rd_index]
                available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent=parent_node, available_nodes_xb=parent_available_nodes_xb, record_io_workload_total=copy.deepcopy(record_io_workload_parent_total), MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                (distance_parent, record_io_workload_parent) = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                (distance_out, record_io_workload_out) = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]
                for (n, v) in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node
    node_mapping_info_list = []
    record_io_workload = []
    if child_node.record_io_workload_out != None:
        for (n, v) in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v
    record_io_workload = child_node.record_io_workload_total
    node_mapping_info_list = mapping_info
    return (node_mapping_info_list, record_io_workload)

def onebyone_search(layer_ref, placed_nodes, available_nodes_xb, mesh_height=6, mesh_width=6, alpha=0, pe_bind_direction=False):
    count = 0
    record_io_workload = {}
    layer_xb_mapped_node = {}
    mapping_info = {}
    for (node_name, split_info) in placed_nodes.items():
        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []
            if count != 0:
                parent_layer_name = layer_ref[layer_name]
                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=pe_bind_direction, method='onebyone')
            if count == 0 or 'IN' in parent_node:
                if 'IN' in parent_node:
                    assert len(parent_node) == 1
                len_ = len(available_nodes_xb)
                rd_index = np.random.randint(0, len_)
                device_ref = available_nodes_xb[rd_index]
                available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, available_nodes_xb=available_nodes_xb, record_io_workload_total=record_io_workload, MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                count += 1
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            else:
                parent_core_id = []
                record_io_workload_parent_total = {}
                for node in parent_node:
                    for (k, v) in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])
                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)
                len_ = len(parent_available_nodes_xb)
                device_ref = parent_available_nodes_xb[0]
                available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent=parent_node, available_nodes_xb=parent_available_nodes_xb, record_io_workload_total=copy.deepcopy(record_io_workload_parent_total), MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                (distance_parent, record_io_workload_parent) = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                (distance_out, record_io_workload_out) = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]
                for (n, v) in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node
    node_mapping_info_list = []
    record_io_workload = []
    if child_node.record_io_workload_out != None:
        for (n, v) in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v
    record_io_workload = child_node.record_io_workload_total
    node_mapping_info_list = mapping_info
    return (node_mapping_info_list, record_io_workload)

def compare_func(items):
    return list(items[1][0].values())[0][3]

def packaged_random_search(layer_ref, placed_nodes, available_nodes_xb, mesh_height=6, mesh_width=6, alpha=0, pe_bind_direction=False, dmac_layer=None):
    import math
    placed_nodes_original = copy.deepcopy(placed_nodes)
    count = 0
    record_io_workload = {}
    layer_xb_mapped_node = {}
    mapping_info = {}
    layer_node_count = {}
    rram_available_nodes_xb = []
    dmac_available_nodes = []
    for n in available_nodes_xb:
        if 'dmac' in n:
            dmac_available_nodes.append(n)
        else:
            rram_available_nodes_xb.append(n)
    available_nodes_xb = rram_available_nodes_xb
    max_pe_thread_num = 2
    for (node_name, split_info) in placed_nodes.items():
        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []
            if count != 0:
                parent_layer_name = layer_ref[layer_name]
                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    elif 'split' in name.lower() or 'concat' in name.lower() or 'add' in name.lower():
                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info, mesh_height=mesh_height, mesh_width=mesh_width, alpha=alpha, pe_bind_direction=pe_bind_direction)
                    else:
                        warnings.warn(f' 当前层{layer_name} 的前一层 {name} 暂未mapping对应的硬件 !!!')
            IsUseDMAC = False
            for dl in dmac_layer:
                if dl in layer_name:
                    IsUseDMAC = True
                    break
            if IsUseDMAC:
                parent_core_id = []
                for node in parent_node:
                    if isinstance(node, MappedLayerNode):
                        core_id = '.'.join(node.addr_node.split('.')[:2])
                        if core_id not in parent_core_id:
                            parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in dmac_available_nodes:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)
                rd_index = np.random.randint(0, len(parent_available_nodes_xb))
                device_ref = parent_available_nodes_xb[rd_index]
                dmac_available_nodes.remove(device_ref)
                mapping_info[split_layer_name] = device_ref
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, available_nodes_xb=dmac_available_nodes, MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                layer_xb_mapped_node[layer_name] = child_node
                continue
            if count == 0 or 'IN' in parent_node:
                if 'IN' in parent_node:
                    assert len(parent_node) == 1
                len_ = len(available_nodes_xb)
                device_num = int(math.ceil(addr[3] / 128))
                if device_num > 1:
                    ct_ = 0
                    while True:
                        rd_index = np.random.randint(0, len_)
                        device_ref = available_nodes_xb[rd_index]
                        dr_list = device_ref.split(':')
                        device_prefix = ':'.join(dr_list[:-1])
                        current_device_xb_index = int(dr_list[-1])
                        if current_device_xb_index % 2 == 0:
                            break
                        ct_ += 1
                        if ct_ > 100:
                            raise ValueError(f'当前不存在偶数开始的XB空闲 !!! 目前空闲的XB: {available_nodes_xb}')
                    ct = 0
                    while True:
                        CanMapNode = True
                        for dn in range(device_num):
                            if device_prefix + f':{current_device_xb_index + dn}' not in available_nodes_xb:
                                CanMapNode = False
                                break
                        if CanMapNode:
                            break
                        ct_ = 0
                        while True:
                            rd_index = np.random.randint(0, len_)
                            device_ref = available_nodes_xb[rd_index]
                            dr_list = device_ref.split(':')
                            device_prefix = ':'.join(dr_list[:-1])
                            current_device_xb_index = int(dr_list[-1])
                            if current_device_xb_index % 2 == 0:
                                break
                            ct_ += 1
                            if ct_ > 100:
                                raise ValueError(f'当前不存在偶数开始的XB空闲 !!! 目前空闲的XB: {available_nodes_xb}')
                        ct += 1
                        if ct > 100:
                            raise ValueError(f'当前层 {split_layer_name}, 无法找到有效的 mapping XB !!! 目前空闲的XB: {available_nodes_xb}')
                    for dn in range(device_num):
                        available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')
                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'
                    if device_prefix not in layer_node_count.keys():
                        layer_node_count[device_prefix] = 1
                else:
                    rd_index = np.random.randint(0, len_)
                    device_ref = available_nodes_xb[rd_index]
                    available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, available_nodes_xb=available_nodes_xb, record_io_workload_total=record_io_workload, MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                count += 1
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            else:
                parent_core_id = []
                for node in parent_node:
                    if isinstance(node, MappedLayerNode):
                        core_id = '.'.join(node.addr_node.split('.')[:2])
                        if core_id not in parent_core_id:
                            parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)
                len_ = len(parent_available_nodes_xb)
                device_num = int(math.ceil(addr[3] / 128))
                if device_num > 1:
                    ct_ = 0
                    while True:
                        rd_index = np.random.randint(0, len_)
                        device_ref = parent_available_nodes_xb[rd_index]
                        dr_list = device_ref.split(':')
                        device_prefix = ':'.join(dr_list[:-1])
                        current_device_xb_index = int(dr_list[-1])
                        if current_device_xb_index % 2 == 0:
                            break
                        ct_ += 1
                        if ct_ > 100:
                            raise ValueError(f'当前不存在偶数开始的XB空闲 !!! 目前空闲的XB: {parent_available_nodes_xb}')
                    ct = 0
                    while True:
                        CanMapNode = True
                        for dn in range(device_num):
                            if device_prefix + f':{current_device_xb_index + dn}' not in parent_available_nodes_xb:
                                CanMapNode = False
                                break
                        if CanMapNode:
                            break
                        ct_ = 0
                        while True:
                            rd_index = np.random.randint(0, len_)
                            device_ref = parent_available_nodes_xb[rd_index]
                            dr_list = device_ref.split(':')
                            device_prefix = ':'.join(dr_list[:-1])
                            current_device_xb_index = int(dr_list[-1])
                            if current_device_xb_index % 2 == 0:
                                break
                            ct_ += 1
                            if ct_ > 100:
                                raise ValueError(f'当前不存在偶数开始的XB空闲 !!! 目前空闲的XB: {parent_available_nodes_xb}')
                        ct += 1
                        if ct > 100:
                            raise ValueError(f'当前层 {split_layer_name}, 无法找到有效的 mapping XB !!! 目前空闲的XB: {parent_available_nodes_xb}')
                    for dn in range(device_num):
                        available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')
                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'
                    if device_prefix not in layer_node_count.keys():
                        layer_node_count[device_prefix] = 1
                    elif layer_node_count[device_prefix] < max_pe_thread_num:
                        layer_node_count[device_prefix] += 1
                    if layer_node_count[device_prefix] == max_pe_thread_num:
                        available_nodes_xb_ = copy.deepcopy(available_nodes_xb)
                        for n in available_nodes_xb_:
                            if device_prefix in n:
                                available_nodes_xb.remove(n)
                else:
                    rd_index = np.random.randint(0, len_)
                    device_ref = parent_available_nodes_xb[rd_index]
                    available_nodes_xb.remove(device_ref)
                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent=parent_node, available_nodes_xb=parent_available_nodes_xb, MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node
    record_io_workload = []
    for layer_name in layer_ref.keys():
        if layer_name in layer_xb_mapped_node.keys():
            parent_layer_name = layer_ref[layer_name]
            parent_node = []
            for name in parent_layer_name:
                if 'graph_input' in name:
                    parent_node.append('IN')
                    continue
                assert name in layer_xb_mapped_node.keys(), f'{name} not in {layer_xb_mapped_node.keys()}'
                parent_node.append(layer_xb_mapped_node[name])
            record_io_workload_parent_total = {}
            for node in parent_node:
                if 'IN' == node:
                    continue
                if node.record_io_workload_total != None:
                    for (k, v) in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
            child_node = layer_xb_mapped_node[layer_name]
            child_node.parent = parent_node
            child_node.record_io_workload_total = copy.deepcopy(record_io_workload_parent_total)
            if 'IN' in parent_node:
                continue
            (distance_parent, record_io_workload_parent) = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
            (distance_out, record_io_workload_out) = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
            child_node.record_io_workload_parent = record_io_workload_parent
            child_node.record_io_workload_out = record_io_workload_out
            child_node.to_parent_cost = distance_parent
            child_node.to_out_cost = distance_out
            for (n, v) in child_node.record_io_workload_parent.items():
                if n in child_node.record_io_workload_total.keys():
                    child_node.record_io_workload_total[n] += v
                else:
                    child_node.record_io_workload_total[n] = v
    if child_node.record_io_workload_out != None:
        for (n, v) in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v
    record_io_workload = child_node.record_io_workload_total
    return (mapping_info, record_io_workload)

class MappedLayerNode:

    def __init__(self, layer_name, addr_node, addr_xb, parent=None, to_parent_cost=0, to_out_cost=0, record_io_workload_parent=None, record_io_workload_out=None, available_nodes_xb=None, record_io_workload_total=None, MESH_HEIGHT=4, MESH_WIDTH=4):
        self.layer_name = layer_name
        self.addr_node = addr_node
        self.addr_xb = addr_xb
        self.parent = parent
        self.to_parent_cost = to_parent_cost
        self.to_out_cost = to_out_cost
        self.record_io_workload_parent = record_io_workload_parent
        self.record_io_workload_out = record_io_workload_out
        self.available_nodes_xb = available_nodes_xb
        self.record_io_workload_total = record_io_workload_total
        self.mesh_height = MESH_HEIGHT
        self.mesh_width = MESH_WIDTH

    def get_all_parent_layer_list(self, parent_layer_list):
        parent = self.parent
        layer_name = self.layer_name.split('.')[0]
        if layer_name not in parent_layer_list:
            parent_layer_list.append(layer_name)
        if parent != None:
            for parent_node in parent:
                layer_name = parent_node.layer_name.split('.')[0]
                if layer_name not in parent_layer_list:
                    parent_layer_list.append(layer_name)
                parent_node.get_all_parent_layer_list(parent_layer_list)

    def get_all_parent_addr(self, node_addr_list):
        
        parent = self.parent
        addr_node = self.addr_node
        if addr_node not in node_addr_list:
            node_addr_list.append(addr_node)
        if parent != None:
            for parent_node in parent:
                if addr_node not in node_addr_list:
                    node_addr_list.append(parent_node.addr_node)
                parent_node.get_all_parent_addr(node_addr_list)

    def get_specify_parent_node(self, parent_name):
        
        parent = self.parent
        specify_parent_node = None
        if parent != None:
            for parent_node in parent:
                if parent_node.layer_name.split('.')[0] == parent_name:
                    specify_parent_node = parent_node
                    return specify_parent_node
                else:
                    specify_parent_node = parent_node.get_specify_parent_node(parent_name)
                    return specify_parent_node
        elif self.layer_name.split('.')[0] == parent_name:
            return self
        return None

    def get_all_parent_node_mapping_info(self, node_mapping_info):
        
        parent = self.parent
        layer_name = self.layer_name
        addr_node = self.addr_node
        addr_xb = self.addr_xb
        if layer_name not in node_mapping_info.keys():
            node_mapping_info[layer_name] = addr_node + '.' + str(addr_xb)
        if parent != None:
            for parent_node in parent:
                parent_node.get_all_parent_node_mapping_info(node_mapping_info)

    def total_cost(self):
        
        return self.to_parent_cost + self.to_out_cost

    def get_candidate_xb(self, radius=2):
        
        cima_node_name = self.addr_node
        device_name = cima_node_name.split('.')[0]
        MESH_WIDTH = self.mesh_width
        MESH_HEIGHT = self.mesh_height
        node_index = int(cima_node_name.split('.')[1].split(':')[1])
        node_id = [node_index // MESH_WIDTH, node_index % MESH_WIDTH]
        assert node_id[0] <= MESH_HEIGHT
        x_direction = []
        y_direction = []
        for i in range(1, radius + 1):
            x_left = node_id[0] - i
            x_right = node_id[0] + i
            if x_left >= 0 and x_left not in x_direction:
                x_direction.append(x_left)
            if x_right <= MESH_WIDTH - 1 and x_right not in x_direction:
                x_direction.append(x_right)
            y_top = node_id[1] - i
            y_bottom = node_id[1] + i
            if y_top >= 0 and y_top not in y_direction:
                y_direction.append(y_top)
            if y_bottom <= MESH_WIDTH - 1 and y_bottom not in y_direction:
                y_direction.append(y_bottom)
        available_xb_id = []
        for x in x_direction:
            for y in y_direction:
                for j in range(4):
                    xb_name = f'{device_name}.cima-node:{4 * x + y}.cima-xb:{j}'
                    if xb_name in self.available_nodes_xb:
                        available_xb_id.append(xb_name)
        return available_xb_id

    def get_to_parent_distance(self, pe_bind_direction=False, alpha=0.5):
        
        MESH_WIDTH = self.mesh_width
        MESH_HEIGHT = self.mesh_height
        node1 = self.addr_node
        node_index1 = int(node1.split('.')[1].split(':')[1])
        node_id1 = [node_index1 // MESH_WIDTH, node_index1 % MESH_WIDTH]
        assert node_id1[0] <= MESH_HEIGHT
        distance = 0
        record_io_workload_current = {}
        for parent in self.parent:
            node2 = parent.addr_node
            node_index2 = int(node2.split('.')[1].split(':')[1])
            node_id2 = [node_index2 // MESH_WIDTH, node_index2 % MESH_WIDTH]
            assert node_id2[0] <= MESH_HEIGHT
            if pe_bind_direction:
                parent_direction = int(node2.split('.')[2].split(':')[1])
                original_node_id2 = copy.deepcopy(node_id2)
                if parent_direction == 0:
                    if node_id2[0] > 0:
                        node_id2[0] = node_id2[0] - 1
                        distance += 1
                elif parent_direction == 1:
                    if node_id2[1] < MESH_WIDTH - 1:
                        node_id2[1] = node_id2[1] + 1
                        distance += 1
                elif parent_direction == 2:
                    if node_id2[0] < MESH_HEIGHT - 1:
                        node_id2[0] = node_id2[0] + 1
                        distance += 1
                elif parent_direction == 3:
                    if node_id2[1] > 0:
                        node_id2[1] = node_id2[1] - 1
                        distance += 1
                if node_id2 != node_id1 and original_node_id2 != node_id2:
                    pair1 = str(original_node_id2) + '-' + str(node_id2)
                    if pair1 not in record_io_workload_current.keys():
                        record_io_workload_current[pair1] = 1
                    else:
                        record_io_workload_current[pair1] += 1
            walked_node_list = self.get_walked_list(node_id1, node_id2)
            if walked_node_list != []:
                pair1 = str(node_id2) + '-' + str(walked_node_list[0])
                assert node_id2 != walked_node_list[0]
                if pair1 not in record_io_workload_current.keys():
                    record_io_workload_current[pair1] = 1
                else:
                    record_io_workload_current[pair1] += 1
                for index in range(len(walked_node_list) - 1):
                    pair = str(walked_node_list[index]) + '-' + str(walked_node_list[index + 1])
                    assert walked_node_list[index] != walked_node_list[index + 1]
                    if pair not in record_io_workload_current.keys():
                        record_io_workload_current[pair] = 1
                    else:
                        record_io_workload_current[pair] += 1
            for i in range(2):
                distance += abs(node_id1[i] - node_id2[i])
            if walked_node_list != []:
                if self.record_io_workload_total != {}:
                    pair1 = str(node_id2) + '-' + str(walked_node_list[0])
                    if pair1 in self.record_io_workload_total.keys():
                        distance += self.record_io_workload_total[pair1] * alpha
                for index in range(len(walked_node_list) - 1):
                    pair = str(walked_node_list[index]) + '-' + str(walked_node_list[index + 1])
                    if self.record_io_workload_total != {}:
                        if pair in self.record_io_workload_total.keys():
                            distance += self.record_io_workload_total[pair] * alpha
                    if pair in record_io_workload_current.keys():
                        distance += record_io_workload_current[pair] * alpha
        return (distance, record_io_workload_current)

    def get_walked_list(self, node_id1, node_id2):
        
        node_list = []
        x_start = node_id2[1]
        y_start = node_id2[0]
        y_dis = node_id1[0] - node_id2[0]
        x_dis = node_id1[1] - node_id2[1]
        if y_dis == 0 and x_dis == 0:
            return node_list
        if x_dis > 0:
            for x_ in range(abs(x_dis)):
                node_list.append([node_id2[0], x_start + x_ + 1])
        elif x_dis < 0:
            for x_ in range(abs(x_dis)):
                node_list.append([node_id2[0], x_start - x_ - 1])
        if node_list != []:
            new_x = node_list[-1][1]
        else:
            new_x = node_id2[1]
        for y_ in range(abs(y_dis)):
            if y_dis > 0:
                node_list.append([y_start + y_ + 1, new_x])
            elif y_dis < 0:
                node_list.append([y_start - y_ - 1, new_x])
        return node_list

    def get_to_out_distance(self, pe_bind_direction=False, alpha=0.5):
        MESH_WIDTH = self.mesh_width
        MESH_HEIGHT = self.mesh_height
        node1 = self.addr_node
        node_index1 = int(node1.split('.')[1].split(':')[1])
        node_id1 = [node_index1 // MESH_WIDTH, node_index1 % MESH_WIDTH]
        assert node_id1[0] <= MESH_HEIGHT
        walked_node_list = self.get_walked_list((3, 0), node_id1)
        record_io_workload_current = {}
        if walked_node_list != []:
            pair1 = str(node_id1) + '-' + str(walked_node_list[0])
            assert node_id1 != walked_node_list[0]
            if pair1 not in record_io_workload_current.keys():
                record_io_workload_current[pair1] = 1
            else:
                record_io_workload_current[pair1] += 1
            for index in range(len(walked_node_list) - 1):
                pair = str(walked_node_list[index]) + '-' + str(walked_node_list[index + 1])
                assert walked_node_list[index] != walked_node_list[index + 1]
                if pair not in record_io_workload_current.keys():
                    record_io_workload_current[pair] = 1
                else:
                    record_io_workload_current[pair] += 1
        distance = 0
        if pe_bind_direction:
            direction1 = int(node1.split('.')[2].split(':')[1])
            if direction1 == 1:
                node_id1[1] = node_id1[1] + 1
                distance += 1
            elif direction1 in [0, 2]:
                distance += 1
        distance = node_id1[1] + 1
        if walked_node_list != []:
            if self.record_io_workload_total != {}:
                pair1 = str(node_id1) + '-' + str(walked_node_list[0])
                if pair1 in self.record_io_workload_total.keys():
                    distance += self.record_io_workload_total[pair1] * alpha
            for index in range(len(walked_node_list) - 1):
                pair = str(walked_node_list[index]) + '-' + str(walked_node_list[index + 1])
                if self.record_io_workload_total != {}:
                    if pair in self.record_io_workload_total.keys():
                        distance += self.record_io_workload_total[pair] * alpha
                if pair in record_io_workload_current.keys():
                    distance += record_io_workload_current[pair] * alpha
        return (distance, record_io_workload_current)

class CIMAPlacement(object):

    def __init__(self, node_weight, XB_size):
        
        self.node_weight = node_weight
        self.XB_size = XB_size

    def run(self):
        
        all_layer_addr = {}
        for split_layer_name in self.node_weight.keys():
            layer_name = split_layer_name.split('.')[0]
            if layer_name not in all_layer_addr:
                all_layer_addr[layer_name] = []
            node_addr = {split_layer_name: [0, 0, int(self.node_weight[split_layer_name][1]), int(self.node_weight[split_layer_name][0])]}
            all_layer_addr[layer_name].append(node_addr)
        return all_layer_addr
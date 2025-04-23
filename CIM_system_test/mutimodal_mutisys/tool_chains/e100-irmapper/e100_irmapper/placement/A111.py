class A111Placement(object):

    def __init__(self, node_weight, XB_size):
        
        self.node_weight = node_weight
        self.XB_size = XB_size

    def run(self):
        
        all_node_addr = {}
        for split_node_name in self.node_weight.keys():
            node_name = split_node_name.split('.')[0]
            if node_name not in all_node_addr:
                all_node_addr[node_name] = []
            node_addr = {split_node_name: [0, 0, int(self.node_weight[split_node_name][1]), int(self.node_weight[split_node_name][0])]}
            all_node_addr[node_name].append(node_addr)
        return all_node_addr
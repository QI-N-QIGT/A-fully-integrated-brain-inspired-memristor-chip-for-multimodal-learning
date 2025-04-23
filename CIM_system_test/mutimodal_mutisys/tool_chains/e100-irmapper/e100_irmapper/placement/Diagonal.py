import copy

class DiagnanolPlacement(object):

    def __init__(self, node_weight, XB_size):
        
        self.node_weight = node_weight
        self.XB_size = XB_size

    def run(self):
        
        [w, h] = self.XB_size
        tile = []
        keys = list(self.node_weight.keys())
        a = keys
        while a:
            t = a
            XB = []
            node_addr = {}
            w_ = self.node_weight[t[0]][0]
            h_ = self.node_weight[t[0]][1]
            node_addr[t[0]] = [0, 0, h_, w_]
            XB.append(node_addr)
            t.remove(t[0])
            m = copy.deepcopy(t)
            for j in range(1, len(t)):
                w_ = w_ + self.node_weight[t[j]][0]
                h_ = h_ + self.node_weight[t[j]][1]
                if w_ <= w and h_ <= h:
                    node_addr = {}
                    node_addr[t[j]] = [h_ - self.node_weight[t[j]][1], w_ - self.node_weight[t[j]][0], self.node_weight[t[j]][1], self.node_weight[t[j]][0]]
                    XB.append(node_addr)
                    m.remove(t[j])
                else:
                    w_ = w_ - self.node_weight[t[j]][0]
                    h_ = h_ - self.node_weight[t[j]][1]
            tile.append(XB)
            a = m
        return tile
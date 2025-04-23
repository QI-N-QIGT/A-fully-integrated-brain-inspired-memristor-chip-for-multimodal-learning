import copy

class DiagnanolPlacement(object):
    
    def __init__(self,node_weight, XB_size):
        '''
        node_weight:字典形式，所有key即为可以放在 XB上进行计算的节点；
                    含有两个元素的列表，第一个元素为宽，第二个元素为高，（卷积核需要按照片上部署的方式展开）。
        XB_size: 单个计算阵列的大小，第一个元素为宽，第二个元素为高。
        '''
        self.node_weight = node_weight
        self.XB_size = XB_size
    
    def run(self):
        '''
        对于已知大小的方块，按照对角线的方式进行每一个方块的放置，直到将所有的方块都放入在XB中。
        放置过程中不考虑方块之间的数据关系，只用方块的大小作为能否放下的标准。
        return：
            按照对角线的方式进行排布的所有节点的位置信息，列表形式
        '''
        [w,h] = self.XB_size
        tile = []
        keys = list(self.node_weight.keys())
        a = keys
        while a:
            t = a
            XB = []
            node_addr = {}
            w_ = self.node_weight[t[0]][0]
            h_ = self.node_weight[t[0]][1]
            node_addr[t[0]] = [0,0,h_,w_] 
            XB.append(node_addr)
            t.remove(t[0])
            m = copy.deepcopy(t)
            for j in range(1, len(t)):
                w_ = w_ + self.node_weight[t[j]][0]
                h_ = h_ + self.node_weight[t[j]][1]
                if w_ <= w and h_ <= h:
                    node_addr = {}
                    node_addr[t[j]] = [h_ - self.node_weight[t[j]][1], w_ - self.node_weight[t[j]][0], self.node_weight[t[j]][1], self.node_weight[t[j]][0]]
                    # XB.append(t[j])
                    XB.append(node_addr)
                    m.remove(t[j])
                else:
                    w_ = w_ - self.node_weight[t[j]][0]
                    h_ = h_ - self.node_weight[t[j]][1]
            tile.append(XB)
            a = m
        return tile

class OneOnOne(object):
    
    def __init__(self,node_weight,XB_size):
        '''
        node_weight: 字典形式，key为所有需要进行排布的节点名称，
                    value为含有两个元素的列表，第一个元素为宽，第二个元素为高，（卷积核需要按照片上部署的方式展开）。
        XB_size: 单个计算阵列的大小，第一个元素为宽，第二个元素为高。
        '''
        self.node_weight = node_weight
        self.XB_size = XB_size
        
    def run(self):
        '''
        以一对一的方式进行块的放置，一个XB最多放置一层
        return:
            所有节点的位置信息，单个XB列表形式
        '''
        all_node_addr = []
        
        for node_name in self.node_weight.keys():
            node_list_per_XB = []
            node_addr = {node_name:[0,0,self.node_weight[node_name][1],self.node_weight[node_name][0]]}
            node_list_per_XB.append(node_addr)
            all_node_addr.append(node_list_per_XB)
        return all_node_addr
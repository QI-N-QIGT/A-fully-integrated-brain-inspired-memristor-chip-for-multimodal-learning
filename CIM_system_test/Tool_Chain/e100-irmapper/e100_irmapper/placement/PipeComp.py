from .LLA import LowestLevelAlgorithm

class PipelineCompression(object):
    
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
        将原本对角线排布的方块，按照最低水平线的方式进行合并
        '''
        # 获取将当前的对角线区域作为 array size 传给 LLA
        H_FAKE = 0
        W_FAKE = 0
        for node_name in self.node_weight.keys():
            H_FAKE = H_FAKE + self.node_weight[node_name][1]
            W_FAKE = W_FAKE + self.node_weight[node_name][0]
        ARRAY_FAKE = [W_FAKE,H_FAKE]
        lla = LowestLevelAlgorithm(self.node_weight,ARRAY_FAKE)
        
        return lla.run()
    
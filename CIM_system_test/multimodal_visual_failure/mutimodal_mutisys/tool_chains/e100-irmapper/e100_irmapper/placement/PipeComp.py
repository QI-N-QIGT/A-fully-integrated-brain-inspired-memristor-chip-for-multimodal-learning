from .LLA import LowestLevelAlgorithm

class PipelineCompression(object):

    def __init__(self, node_weight, XB_size):
        
        self.node_weight = node_weight
        self.XB_size = XB_size

    def run(self):
        
        H_FAKE = 0
        W_FAKE = 0
        for node_name in self.node_weight.keys():
            H_FAKE = H_FAKE + self.node_weight[node_name][1]
            W_FAKE = W_FAKE + self.node_weight[node_name][0]
        ARRAY_FAKE = [W_FAKE, H_FAKE]
        lla = LowestLevelAlgorithm(self.node_weight, ARRAY_FAKE)
        return lla.run()
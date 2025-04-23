from ..onnx2ir.parser import *

class PytorchModuleParser(OnnxParser):

    def __init__(self, onnx_model):
        super().__init__(onnx_model)
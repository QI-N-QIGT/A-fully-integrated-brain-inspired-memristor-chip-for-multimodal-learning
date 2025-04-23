# import onnx
import onnxruntime as ort
import numpy as np
import torch

input_ = torch.load('data\\layer_input_features_conv_0_maxpool.pth')['conv1'][0].numpy()
re = []
for i in range(1):
    ort_session =  ort.InferenceSession('model_with_fixed_name_new_add_6.onnx')
    ort_inputs = {"modelInput":input_[i:i+2] }
    outputs_val = ort_session.run(None,ort_inputs)
    re.append(outputs_val[1])
re = np.concatenate(re,axis=0)
np.save('onnx_results_add_6',re)

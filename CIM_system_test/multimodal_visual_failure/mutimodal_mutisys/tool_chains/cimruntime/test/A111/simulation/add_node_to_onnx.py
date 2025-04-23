from model2ir.onnx2ir.helper import *

def add_node_to_output():
    model = onnx.load('model_with_fixed_name.onnx')
    value_info = get_model_value_info(model)
    new_output_name = ['/layer4/layer4.0/Add_output_0']
    for name in new_output_name:
        model.graph.output.append(value_info[name])
    model.ir_version = 5
    model.opset_import[0].version = 10
    save_onnx_model(model, 'model_with_fixed_name_new_add_6.onnx')
if __name__ == '__main__':
    add_node_to_output()
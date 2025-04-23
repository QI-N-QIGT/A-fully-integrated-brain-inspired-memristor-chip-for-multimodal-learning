from model2ir.onnx2ir.converter import ConvertONNX
from cimruntime.c200.c200_rt import C200NumpyRT
from cimruntime.cimtensor import CIMNumpyTensor as CNT
from cimruntime.gen_weight import gen_array_weight
from e100_irmapper.mapper import mapper
from e100_irmapper.device.c200 import C200CalcInfo
from e100_irmapper.device.c200 import MappedLayer
from e100_irmapper.placement import LowestLevelAlgorithm
import numpy as np
from e100_irtool.core.ir import load_ir
from dummy_npu.data_io import pickle_save, pickle_load
from e100_iroptimizer.c200.optimizer import AccOptimzier
import time
import multiprocessing
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from e100_irtool.core.ir import BaseIR
from cimruntime.quant import data_quantization_sym

def load_csv(fn, dtype='int32'):
    assert fn
    return np.loadtxt(fn, dtype=dtype, delimiter=',', ndmin=2)

def save_csv(fn, data):
    assert fn
    np.savetxt(fn, data, delimiter=',', fmt='%d')

def txt2numpy(fn):
    with open(fn, 'r') as f:
        data = f.readline()
        data_all = []
        while data:
            data = data.strip()
            data1 = data.split(',')
            if data1[-1] == '':
                data1.remove(data1[-1])
            new_data = []
            for i in data1:
                new_data.append(int(i))
            data = f.readline()
            data_all.append(new_data)
    return np.array(data_all)

def draw_rectangles(H, W, rectangles, save_fig=False):
    n = len(rectangles.keys())
    c = 0
    (fig, ax) = plt.subplots(1, n)
    if n > 1:
        fig.set_size_inches(6 * (n - 1), 5)
    else:
        fig.set_size_inches(4, 5)
    for (key, value) in rectangles.items():
        if n > 1:
            ax[c].set_xlim(0, W)
            ax[c].set_ylim(H, 0)
            ax[c].set_xticks([0, W])
            ax[c].set_yticks([0, H])
            ax[c].set_aspect('auto')
        else:
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_xticks([0, W])
            ax.set_yticks([0, H])
            ax.set_aspect('auto')
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightsalmon', 'lightpink', 'lightgray', 'lightcyan']
        for (i, rect) in enumerate(value):
            (y, x, h, w, text) = rect
            color = colors[i % len(colors)]
            text_x = x + w / 2
            text_y = y + h / 2
            font_size = 16
            if w <= 16:
                font_size = 6
            elif w <= 32:
                font_size = 8
            elif w <= 64:
                font_size = 12
            if w > h:
                rotation = 0
            else:
                rotation = 'vertical'
            if n > 1:
                ax[c].add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black'))
                ax[c].text(text_x, text_y, text, ha='center', va='center', color='black', rotation=rotation, fontsize=font_size)
            else:
                ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black'))
                ax.text(text_x, text_y, text, ha='center', va='center', color='black', rotation=rotation, fontsize=font_size)
        plt.gca().set_aspect('auto', adjustable='box')
        plt.tight_layout()
        if n > 1:
            ax[c].set_title(f'# Chip {key}')
            c += 1
        else:
            ax.set_title(f'# Chip {key}')
    if save_fig:
        plt.savefig(save_fig, transparent=True)
    else:
        plt.show()

def get_rectangles(ir):
    layers = ir.layers
    rectangles = {}
    for (name, layer) in layers.items():
        if layer.type == 'op' and layer.op.op_id in ['conv2d', 'matmul', 'fc', 'conv_transpose2d']:
            mappings = layer.c200_mapping_info.mappings
            len_ = len(mappings.keys())
            for (key, value) in mappings.items():
                loc = list(value.index)
                device = value.device
                index = device.split(':')[-1]
                if index not in rectangles.keys():
                    rectangles[index] = []
                addr = list(value.address)
                if len_ > 1:
                    addr.append(f'{name}:{loc[1:]}')
                else:
                    addr.append(f'{name}')
                rectangles[index].append(addr)
    return rectangles

def draw_ir(ir, save_fig=False):
    if isinstance(ir, str):
        ir = load_ir(file=ir)
    elif isinstance(ir, BaseIR):
        ir = ir
    else:
        raise ValueError(f'暂不支持的格式:{type(ir)} !!!')
    rectangles = get_rectangles(ir)
    rectangles = sorted(rectangles.items(), key=lambda rectangles: int(rectangles[0]))
    rectangles_sorted = {}
    for i in rectangles:
        rectangles_sorted[i[0]] = i[1]
    H = 576
    W = 128
    draw_rectangles(H, W, rectangles_sorted, save_fig=save_fig)
if __name__ == '__main__':
    multiprocessing.freeze_support()
    device = [{'name': 'c200-0', 'kind': 'rram-144k-cluster', 'num': 20, 'ip': '192.168.2.97'}]
    model = 'resnet56_bias=False.onnx'
    onnx_obj = ConvertONNX(model, weight_half_level=7, store_intermediate_model=False)
    onnx_ir = onnx_obj.ir
    onnx_weight_data = onnx_obj.model_parser.weight_numpy
    onnx_weight_data_quant = onnx_obj.model_parser.weight_numpy_quant
    onnx_weight_data_scale = onnx_obj.model_parser.weight_quant_scale
    cpu_layer = None
    calc_info = None
    copy_para = None
    map = mapper(ir=onnx_ir, device=device, cpu_layer=cpu_layer, calc_info=calc_info, place_strategy=LowestLevelAlgorithm, average_copy=copy_para, runtime='simulation')
    map.run()
    mapped_ir = map.ir
    draw_ir(mapped_ir, save_fig=f'layout.svg')
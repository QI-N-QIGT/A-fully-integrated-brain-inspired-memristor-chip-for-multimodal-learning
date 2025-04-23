import sys
import tensorflow as tf
from e100_irtool.cmd.data import load_pickle, save_pickle

def main():
    try:
        (in_f, wt_f, out_f) = sys.argv[1:4]
        dump_all = sys.argv[4:5] == ['-a']
    except:
        sys.exit(f'Usage: {sys.argv[0]} <input> <weight> <output> [-a]')
    inp = load_pickle(in_f)
    wts = load_pickle(wt_f)
    net = tf.keras.Sequential([tf.keras.layers.Conv2D(64, 3, padding='SAME', name='conv'), tf.keras.layers.ReLU(name='relu'), tf.keras.layers.GlobalMaxPool2D(name='pool', keepdims=True), tf.keras.layers.Flatten(name='flat'), tf.keras.layers.Dense(10, use_bias=False, name='fc')])
    net.build(inp.shape)
    net.get_layer('conv').set_weights([wts['conv.weight'], wts['conv.bias']])
    net.get_layer('fc').set_weights([wts['fc.weight']])
    if dump_all:
        oup = dict()
        x = inp
        for layer in net.layers:
            y = layer(x)
            oup[layer.name] = y
            x = y
    else:
        oup = net(inp)
    save_pickle(oup, out_f)
if __name__ == '__main__':
    main()
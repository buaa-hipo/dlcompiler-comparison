import mxnet as mx
import numpy as np
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.contrib import onnx as onnx_mxnet
import os


def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())

    return sym, args, auxs


def save_model(block, model_name, path='./'):
    mx_sym, args, auxs = block2symbol(block)
    os.makedirs(path, exist_ok=True)
    mx.model.save_checkpoint(path + model_name, 0, mx_sym, args, auxs)


def convert_sym_params_to_onnx(model_name, input_shape, path_sym_params, path_onnx='./'):
    model_path = path_onnx + model_name + '.onnx'
    sym = path_sym_params + model_name + '-symbol.json'
    params = path_sym_params + model_name + '-0000.params'
    onnx_mxnet.export_model(sym, params, [input_shape], np.float32, model_path, verbose=True)


def remove_checkpoint(model_name, path='./'):
    sym = path + model_name + '-symbol.json'
    params = path + model_name + '-0000.params'
    os.remove(sym)
    os.remove(params)


def mxnet2onnx(block, model_name, input_shape, path_sym_params='./', path_onnx='./'):
    save_model(block, model_name, path_sym_params)
    convert_sym_params_to_onnx(model_name, input_shape, path_sym_params, path_onnx)
    remove_checkpoint(model_name, path_sym_params)


def get_name2conv(model):
    features = model.features
    name2conv = {}
    name2conv['conv0'] = features._children['0']
    name2conv['conv1'] = features._children['20']
    name2conv['conv2'] = model.output._children['0']

    for bottleneck_number in range(17):
        bottleneck = features._children[str(bottleneck_number + 3)].out._children
        name2conv[f'b{bottleneck_number}_expand'] = bottleneck['0']
        name2conv[f'b{bottleneck_number}_dwise'] = bottleneck['3']
        name2conv[f'b{bottleneck_number}_linear'] = bottleneck['6']

    return name2conv


def get_name2inputShape():
    name2inputShape = {'conv0': (1, 3, 224, 224),
                       'b0_expand': (1, 32, 112, 112),
                       'b0_dwise': (1, 32, 112, 112),
                       'b0_linear': (1, 32, 112, 112),
                       'b1_expand': (1, 16, 112, 112),
                       'b1_dwise': (1, 96, 112, 112),
                       'b1_linear': (1, 96, 56, 56),
                       'b2_expand': (1, 24, 56, 56),
                       'b3_expand': (1, 24, 56, 56),
                       'b2_dwise': (1, 144, 56, 56),
                       'b2_linear': (1, 144, 56, 56),
                       'b3_dwise': (1, 144, 56, 56),
                       'b3_linear': (1, 144, 28, 28),
                       'b4_expand': (1, 32, 28, 28),
                       'b5_expand': (1, 32, 28, 28),
                       'b6_expand': (1, 32, 28, 28),
                       'b4_dwise': (1, 192, 28, 28),
                       'b5_dwise': (1, 192, 28, 28),
                       'b4_linear': (1, 192, 28, 28),
                       'b5_linear': (1, 192, 28, 28),
                       'b6_dwise': (1, 192, 28, 28),
                       'b6_linear': (1, 192, 14, 14),
                       'b7_expand': (1, 64, 14, 14),
                       'b8_expand': (1, 64, 14, 14),
                       'b9_expand': (1, 64, 14, 14),
                       'b10_expand': (1, 64, 14, 14),
                       'b7_dwise': (1, 384, 14, 14),
                       'b8_dwise': (1, 384, 14, 14),
                       'b9_dwise': (1, 384, 14, 14),
                       'b10_dwise': (1, 384, 14, 14),
                       'b7_linear': (1, 384, 14, 14),
                       'b8_linear': (1, 384, 14, 14),
                       'b9_linear': (1, 384, 14, 14),
                       'b10_linear': (1, 384, 14, 14),
                       'b11_expand': (1, 96, 14, 14),
                       'b12_expand': (1, 96, 14, 14),
                       'b13_expand': (1, 96, 14, 14),
                       'b11_dwise': (1, 576, 14, 14),
                       'b11_linear': (1, 576, 14, 14),
                       'b12_dwise': (1, 576, 14, 14),
                       'b12_linear': (1, 576, 14, 14),
                       'b13_dwise': (1, 576, 14, 14),
                       'b13_linear': (1, 576, 7, 7),
                       'b14_expand': (1, 160, 7, 7),
                       'b14_dwise': (1, 960, 7, 7),
                       'b15_expand': (1, 160, 7, 7),
                       'b15_dwise': (1, 960, 7, 7),
                       'b16_expand': (1, 160, 7, 7),
                       'b16_dwise': (1, 960, 7, 7),
                       'b14_linear': (1, 960, 7, 7),
                       'b15_linear': (1, 960, 7, 7),
                       'b16_linear': (1, 960, 7, 7),
                       'conv1': (1, 320, 7, 7),
                       'conv2': (1, 1280, 1, 1)
                       }
    return name2inputShape


if __name__ == '__main__':
    model_name = 'mobilenetv2_1.0'
    onnx_path = './mobilenet/'
    block = get_model(model_name, pretrained=True)
    name2conv = get_name2conv(block)
    name2inputShape = get_name2inputShape()

    for name in name2conv.keys():
        mxnet2onnx(name2conv[name], name, name2inputShape[name], path_onnx=onnx_path)

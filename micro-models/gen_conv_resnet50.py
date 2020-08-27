import torch
import torchvision


def get_name2conv(model):
    name2conv = {}
    name2conv['conv1'] = model.conv1
    for layer_number in range(2, 6):
        layer = getattr(model, 'layer' + str(layer_number - 1))
        for bottleneck_number in range(1, len(layer) + 1):
            bottleneck = layer[bottleneck_number - 1]
            nodes = vars(bottleneck)['_modules']
            for node in nodes.keys():
                if 'conv' in node:
                    conv_name = 'conv' + str(layer_number) + '_x' + str(bottleneck_number) + '_' + node[4:]
                    name2conv[conv_name] = nodes[node]
                elif 'downsample' == node:
                    conv_name = 'conv' + str(layer_number) + '_x' + str(bottleneck_number) + '_shortcut'
                    name2conv[conv_name] = nodes[node][0]

    return name2conv


def get_name2inputSize():
    name2inputSize = {'conv1': (1, 3, 224, 224),
                      'conv2_x1_1': (1, 64, 56, 56),
                      'conv2_x1_2': (1, 64, 56, 56),
                      'conv2_x2_2': (1, 64, 56, 56),
                      'conv2_x3_2': (1, 64, 56, 56),
                      'conv2_x1_3': (1, 64, 56, 56),
                      'conv2_x2_3': (1, 64, 56, 56),
                      'conv2_x3_3': (1, 64, 56, 56),
                      'conv2_x1_shortcut': (1, 64, 56, 56),
                      'conv2_x2_1': (1, 256, 56, 56),
                      'conv2_x3_1': (1, 256, 56, 56),
                      'conv3_x1_1': (1, 256, 56, 56),
                      'conv3_x1_2': (1, 128, 56, 56),
                      'conv3_x1_3': (1, 128, 28, 28),
                      'conv3_x2_3': (1, 128, 28, 28),
                      'conv3_x3_3': (1, 128, 28, 28),
                      'conv3_x4_3': (1, 128, 28, 28),
                      'conv3_x1_shortcut': (1, 256, 56, 56),
                      'conv3_x2_1': (1, 512, 28, 28),
                      'conv3_x3_1': (1, 512, 28, 28),
                      'conv3_x4_1': (1, 512, 28, 28),
                      'conv3_x2_2': (1, 128, 28, 28),
                      'conv3_x3_2': (1, 128, 28, 28),
                      'conv3_x4_2': (1, 128, 28, 28),
                      'conv4_x1_1': (1, 512, 28, 28),
                      'conv4_x1_2': (1, 256, 28, 28),
                      'conv4_x1_3': (1, 256, 14, 14),
                      'conv4_x2_3': (1, 256, 14, 14),
                      'conv4_x3_3': (1, 256, 14, 14),
                      'conv4_x4_3': (1, 256, 14, 14),
                      'conv4_x5_3': (1, 256, 14, 14),
                      'conv4_x6_3': (1, 256, 14, 14),
                      'conv4_x1_shortcut': (1, 512, 28, 28),
                      'conv4_x2_1': (1, 1024, 14, 14),
                      'conv4_x3_1': (1, 1024, 14, 14),
                      'conv4_x4_1': (1, 1024, 14, 14),
                      'conv4_x5_1': (1, 1024, 14, 14),
                      'conv4_x6_1': (1, 1024, 14, 14),
                      'conv4_x2_2': (1, 256, 14, 14),
                      'conv4_x3_2': (1, 256, 14, 14),
                      'conv4_x4_2': (1, 256, 14, 14),
                      'conv4_x5_2': (1, 256, 14, 14),
                      'conv4_x6_2': (1, 256, 14, 14),
                      'conv5_x1_1': (1, 1024, 14, 14),
                      'conv5_x1_2': (1, 512, 14, 14),
                      'conv5_x1_3': (1, 512, 7, 7),
                      'conv5_x2_3': (1, 512, 7, 7),
                      'conv5_x3_3': (1, 512, 7, 7),
                      'conv5_x1_shortcut': (1, 1024, 14, 14),
                      'conv5_x2_1': (1, 2048, 7, 7),
                      'conv5_x2_2': (1, 512, 7, 7),
                      'conv5_x3_1': (1, 2048, 7, 7),
                      'conv5_x3_2': (1, 512, 7, 7)
                      }
    return name2inputSize


def get_models_pytorch(model_name):
    model = getattr(torchvision.models, model_name)(pretrained=True).eval()

    return model


def output_conv(conv, conv_name, input_size, onnx_path='./'):
    print(f'export {conv_name}')
    dummy_input = torch.randn(input_size)
    script = torch.jit.trace(conv, dummy_input)
    torch.onnx.export(script, dummy_input, onnx_path + conv_name + '.onnx', verbose=False, input_names=['data'],
                      output_names=['output'], example_outputs=script(dummy_input))


if __name__ == '__main__':
    model = get_models_pytorch('resnet50')
    name2conv = get_name2conv(model)
    name2inputSize = get_name2inputSize()
    for name in name2conv.keys():
        output_conv(name2conv[name], name, name2inputSize[name], onnx_path='./resnet50/')

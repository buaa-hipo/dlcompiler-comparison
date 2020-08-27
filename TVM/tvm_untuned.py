import tvm
import onnx
import numpy as np
import tvm.relay as relay
import tvm.contrib.graph_runtime as runtime


def get_onnx(path):
    ox = onnx.load(path)
    name = ox.graph.input[0].name
    input_shape = [i.dim_value for i in ox.graph.input[0].type.tensor_type.shape.dim]
    input_shape[0] = 1
    return ox, {name: input_shape}

def create_target(device):
    if device == 'x86' or device == 'CPU':
        target = tvm.target.create('llvm -mcpu=core-avx2')
    elif device == 'gpu':
        target = tvm.target.cuda()
    return target

def create_ctx(device):
    if device == 'x86' or device == 'CPU':
        ctx = tvm.cpu()
    elif device == 'gpu':
        ctx = tvm.gpu()
    return ctx

def speed(graph, lib, params, ctx, input_dict):
    for name, shape in input_dict.items():
        input_name, input_shape = name, shape
    input_data = tvm.nd.array(np.random.uniform(size=input_shape).astype("float32"))
    
    module = runtime.create(graph, lib, ctx)
    module.set_input(input_name, input_data)
    module.set_input(**params)
    
    ftimer = module.module.time_evaluator('run', ctx, number=1, repeat=15)
    prof_res = np.array(ftimer().results)
    return prof_res


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="", type=str)
    parser.add_argument('-d', '--device', default='', type=str)
    args = parser.parse_args()

    model_name = os.path.basename(args.model).replace('.onnx', '')

    ox, input_shape = get_onnx(args.model)
    target = create_target(args.device)
    mod, relay_params = relay.frontend.from_onnx(ox, input_shape)
    func = mod['main']
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=relay_params)
    
    ctx = create_ctx(args.device)
    prof_res = speed(graph, lib, params, ctx, input_shape)
    import time
    print(time.strftime('[localtime] %Y-%m-%d %H:%M:%S', time.localtime()))
    print(model_name)
    print(input_shape)
    for i in range(5, 15):
        print('-- {}, iteration time(s) is {:.6f}'.format(i, prof_res[i]))

    print('@@ {}, average time(s) is {:.6f}'.format(model_name, np.mean(prof_res[5:])))
    print('FINISH')


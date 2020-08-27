import os
import sys
import argparse
from tvm.contrib import util
from tvm import relay
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import numpy as np
import tvm.contrib.graph_runtime as runtime


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', type=str, default='x86')
parser.add_argument('-m', '--model', type=str, default=None)
parser.add_argument('-l', '--log', type=str, default=None)


args = parser.parse_args()

def get_model(model_path):
    import onnx

    ox = onnx.load(model_path)
    name = ox.graph.input[0].name
    input_shape = [i.dim_value for i in ox.graph.input[0].type.tensor_type.shape.dim]
    shape_dict = {name: input_shape}

    return ox, shape_dict

def get_target():
    if args.target == 'x86' or args.target == 'cpu':
        target = tvm.target.create('llvm -mcpu=core-avx2')
    elif args.target == 'gpu':
        target = tvm.target.cuda()

    return target

def get_logfile():
    if args.log:
        return args.log
    model_name = args.model.split('/')[-1][:-5]
    log_filepath = './tvm-log/' + args.target + '/onnx/1batch'
    if not os.path.exists(log_filepath):
        os.mkdir(log_filepath)
    log_file = log_filepath + '/' + '_'.join([args.target, 'onnx', '1batch', model_name]) + '.log'

    return log_file

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=False,
               try_spatial_pack_depthwise=False):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # if we want to use spatial pack for depthwise convolution
    if try_spatial_pack_depthwise:
        tuner = 'xgb_knob'
        for i in range(len(tasks)):
            if tasks[i].name == 'topi_nn_depthwise_conv2d_nchw':
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host,
                                          'contrib_spatial_pack')
                tasks[i] = tsk

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                  target=tsk.target, template_key='direct')
        task.workload = tsk.workload
        tsk = task


        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial_temp = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial_temp,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial_temp, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)




def tuning(tuning_option,
           model_path=None,
           dtype='float32',
           input_name='data',
           device_key=None,
           use_android=False):
    print('Extract tasks...')

    ox, shape_dict = get_model(model_path)
    mod, params = relay.frontend.from_onnx(ox, shape_dict)
    input_shape = shape_dict[input_name]
    target = get_target()
    tasks = autotvm.task.extract_from_program(mod['main'], target=target,
                                             params = params,
                                             ops=(relay.op.nn.conv2d,))
    log_file = tuning_option['log_filename']

    if os.path.exists(log_file):
        print(log_file + " exists, skipping...")
    else:
        print(log_file + " doesn't exist")
        print('Tuning...')
        tune_tasks(tasks, **tuning_option)

    func = mod['main']
    with autotvm.apply_history_best(log_file):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func, target, params=params)

    return graph, lib, params



def tuning_model(model_path):
    dtype='float32'
    ox, shape_dict = get_model(model_path)
    input_name = list(shape_dict.keys())[0]
    device_key = None
    if args.target == 'gpu':
        device_key = 'V100'
    use_android = False

    log_file = get_logfile()

    other_option = {
        'model_path': model_path,
        'dtype': dtype,
        'input_name': input_name,
        'device_key': device_key,
        'use_android': use_android
    }

    if args.target == 'x86' or args.target == 'cpu':
        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(
                    number=10, repeat=1,
                    min_repeat_ms=1000
                )
        )
    elif args.target == 'gpu':
        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.RPCRunner(
                    device_key,
                    '0.0.0.0', 9190,
                    number=20, repeat=3, timeout=4, min_repeat_ms=150)
        )
    n_trial = 200

    tuning_option = {
        'log_filename': log_file,
        'tuner': 'xgb',
        'n_trial': n_trial,
        'early_stopping': 80,
        'measure_option': measure_option
    }

    graph, lib, params = tuning(tuning_option, **other_option)
    return graph, lib, params

def speed(graph, lib, params, shape_dict, dtype='float32'):
    if args.target == 'x86' or args.target == 'cpu':
        ctx = tvm.cpu()
    elif args.target == 'gpu':
        ctx = tvm.gpu()
    input_name = list(shape_dict.keys())[0]
    input_shape = list(shape_dict.values())[0]
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))

    module = runtime.create(graph, lib, ctx)
    module.set_input(input_name, data_tvm)
    module.set_input(**params)

    ftimer = module.module.time_evaluator('run', ctx, number=1, repeat=15)
    prof_res = np.array(ftimer().results)

    return prof_res



if __name__ == '__main__':
    graph, lib, params = tuning_model(args.model)
    _, shape_dict = get_model(args.model)
    prof_res = speed(graph, lib, params, shape_dict)
    import time
    print(time.strftime('[localtime] %Y-%m-%d %H:%M:%S', time.localtime()))
    model_name = args.model.split('/')[-1][:-5]
    print(model_name)
    print(list(shape_dict.values())[0])
    for i in range(5, 15):
        print('-- {}, iteration time(s) is {:.4f}'.format(i, prof_res[i]))

    print('@@ {}, average time(s) is {:.4f}'.format(model_name, np.mean(prof_res[5:])))
    print('FINISH')

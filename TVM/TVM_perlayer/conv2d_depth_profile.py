
from utils import *

def get_conv2d_depthwise(conv_type,
				batch_size,
               IC,
               HW,
               OC,
               KHKW,
               Pad,
               Stride,
               layout="NCHW",
               dtype="float32"):

    from tvm.relay.testing.init import create_workload
    from tvm.relay.testing import layers

    data_layout = layout
    kernel_layout = "OIHW" if layout == "NCHW" else "HWIO"

    data_shape = (batch_size, IC, HW, HW)
    data = relay.var("data", shape=data_shape, dtype=dtype)

    if 'conv' in conv_type:
        net = layers.conv2d(data=data,
                        channels=OC,
                        kernel_size=(KHKW, KHKW),
                        strides=(Stride, Stride),
                        padding=(Pad, Pad),
                        name="conv2d_profile",
                        data_layout=data_layout,
                        kernel_layout=kernel_layout)
    elif 'depthwise' in conv_type:
        print("build depthwise net")
        net = layers.conv2d(data=data,
                        channels=OC,
						groups=OC,
                        kernel_size=(KHKW, KHKW),
                        strides=(Stride, Stride),
                        padding=(Pad, Pad),
                        name="depthwise_profile",
                        data_layout=data_layout,
                        kernel_layout=kernel_layout)
    else:
        print("err : not correct conv type")
	
    return create_workload(net)


from tvm import autotvm

def get_conv2d_workload(mod, target, params, log):
    import os
    if os.path.isfile(log):
        with autotvm.apply_history_best(log):
            with relay.build_config(opt_level=4):
                graph, lib, params = relay.build(mod = mod , target = target , params = params)
    else:
        with relay.build_config(opt_level=4):
            graph, lib, params = relay.build(mod = mod , target = target , params = params)
    return graph, lib , params


def get_tasks(mod, params, target, quantize='No'):
    from tvm import relay
    if arg.quantize == 'fp16' or arg.quantize == 'float16':
        with relay.quantize.qconfig(skip_conv_layers=[0],
                                    nbit_input=16,
                                    nbit_weight=16,
                                    nbit_activation=16,
                                    global_scale=16.0,
                                    dtype_input='float16',
                                    dtype_weight='float16',
                                    dtype_activation='float16'):
            mod = relay.quantize.quantize(mod, params=params)
    elif arg.quantize == 'int8':
        with relay.quantize.qconfig(nbit_activation=8,
                                    dtype_activation='int8'):
            mod = relay.quantize.quantize(mod, params=params)
    func = mod["main"] 
    ops = [
        relay.op.get("nn.conv2d"),
        relay.op.get("nn.batch_matmul"),
        relay.op.get("nn.dense"),
        relay.op.get("nn.conv2d_transpose"),
        ]
    tasks = autotvm.task.extract_from_program(func, target = target,
                                params = params,
                                ops = ops)
    for i in range(len(tasks)):
        try:  
            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                      tasks[i].target, tasks[i].target_host, 'winograd')
            input_channel = tsk.workload[1][1]
            if input_channel >= 64:
                print(tasks[i].name + " goto winograd " + tsk.name , tasks[i].args)
                tasks[i] = tsk
        except Exception:
            pass
	
#print("tasks : ")
#print(tasks)
    '''
    # converting conv2d tasks to conv2d_NCHWc tasks
    for i in range(len(tasks)):
        tsk = tasks[i]kkk
	    op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload
    '''

    return tasks

def create_measure(device, flag = "t4"):
    if device == 'arm' or device == 'aarch64':
        measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(
        build_func='ndk' if use_android else 'default'),
        runner=autotvm.RPCRunner(
        "pi", host='0.0.0.0', port=9190,
        number=5,
        timeout=10,
        ))
    elif ('x86' in device) :
        measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=5, repeat=1,
        min_repeat_ms=1000),
       )
    elif device == 'gpu':
        measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=1000),
        runner=autotvm.RPCRunner(
        flag,  # change the device key to your key
        '0.0.0.0', 9190,
        number=20, repeat=3, timeout=1000, min_repeat_ms=150)
        )
    return measure_option

def tune_task(name, tasks, measure, resume_log_file = "tune.log", n_trial = 10):
    from tvm.autotvm.tuner import XGBTuner
    import os
    dir_name = os.path.dirname(resume_log_file)
    try:
        os.mkdir(dir_name)
    except:
        pass
    for idx , task in enumerate(tasks):
        prefix = "[%s][Task %2d/%2d] " % (name, idx + 1, len(tasks) )
        tuner = XGBTuner(task, loss_type = 'rank')
        if os.path.isfile(resume_log_file):
            print("load log file:" + resume_log_file)
            tuner.load_history(autotvm.record.load_from_file(resume_log_file)) 
        n_try = min(n_trial, len(task.config_space))
        callbacks = [
            autotvm.callback.progress_bar(n_try, prefix = prefix),
            autotvm.callback.log_to_file(resume_log_file)
        ]
        try:
            import tvm.tvmt
            log_db = resume_log_file + ".db"
            callbacks.append(tvm.tvmt.log_to_sqlite(log_db))
        except:
            pass
        tuner.tune(n_trial = n_try,
                    early_stopping = 80,
                    measure_option = measure,
                    callbacks = callbacks
                    )
    return 

def join_name(config):
    res = ''
    for ele in config:
        res=res+str(ele)+'_'
    return res

######################################################################


def conv2d_profile(arg, conv_type, batch_size, IC, HW, OC, KHKW, Pad, Stride):
    # get mod params
    target = create_target(arg.device)
    config_conv2d = (batch_size, IC, HW, OC, KHKW, Pad, Stride)
    mod, params = get_conv2d_depthwise(conv_type, *config_conv2d)
    # set show_meta_data=True if you want to show meta data
    print(mod.astext(show_meta_data=False))

    if arg.log == 'false':
        # tuning
        measure = create_measure(arg.device, arg.flag)
        tasks = get_tasks(mod, params, target, arg.quantize)
        print("Got %d task to tune" % (len(tasks)))
        for i in tasks:
            print(i.name, i.config_space)    
        name_log = os.path.join("logs", arg.device+'-'+arg.flag+'_'+ conv_type+'_'+str(arg.batch)+'batch' +'_'+str(arg.thread)+'thread'+'_' + join_name(config_conv2d) + ".log")
        tune_task("conv2d", tasks, measure, resume_log_file = name_log, n_trial = arg.time)
    else:
        if 'depthwise' in conv_type and arg.usedepthlog == 'false':
            name_log = 'false'
        else:
            name_log = arg.log

    # speed profile
    graph, lib, params = get_conv2d_workload(mod, target, params, log=name_log)
    ctx = create_ctx(arg.device) 
    #
    if arg.profile == 'false':
        time = speed(graph, lib, params, ctx)
    elif arg.profile == 'true':
        time = speed_profile(graph, lib, params, ctx)
    #
    #name = os.path.basename(arg.relay)
    return time

def get_data_and_test(arg):
    import os
    import re
    import csv
    import sys
    def get_raw_data(path):
        with open(path,encoding='utf8') as f:
            f_csv = csv.reader(f)
            data_list =[]
            for row in f_csv:
                data_list.append(row)
            return data_list
    
    def get_conv_cases(path_conv_cases):
        raw_data = get_raw_data(path_conv_cases)
        #print(raw_data)
        conv_cases=[]
        conv_depth_flag =[]
        for line in raw_data:
            tmp=[]
            if( re.search('conv', line[0]) ):
                for ele in line[2:]:
                    tmp.append(int(ele))
                conv_cases.append(tmp)
                conv_depth_flag.append('conv')
            elif( re.search('depthwise', line[0]) ):
                for ele in line[2:]:
                    tmp.append(int(ele))
                conv_cases.append(tmp)
                conv_depth_flag.append('depthwise')
        return conv_cases, conv_depth_flag

    def test_conv2d(arg, conv_type, IC, HW, OC, KHKW, Pad, Stride):
        batch_size = arg.batch

        config_conv2d = (batch_size, IC, HW, OC, KHKW, Pad, Stride)
        time = conv2d_profile (arg, conv_type, *config_conv2d)
        return time
        #name = 'conv2d_profile'
        #print("%s, %.4f" % (name, time))

    def write_csv(path, result_lists):
        with open( path, 'w', encoding='utf8') as f:
            csv_write = csv.writer(f)
            for line in (result_lists):
                csv_write.writerow(line)
    
    def write_res(path_res, config_file, data):
        index =0
        res=[]
        for line in config_file:
            if re.search( 'conv', line[0]):
                line.append(data[index])
                index+=1
            elif re.search( 'depthwise', line[0]):
                line.append(data[index])
                index+=1
            elif re.search( 'id', line[0]):
                line.append(arg.device)
            else:
                line.append('')
            res.append(line)
        #print(res)
        write_csv(path_res, res)



    path_conv_cases = arg.path
    conv_cases, conv_depth_flag = get_conv_cases(path_conv_cases)
    print(conv_cases)
    print(conv_depth_flag)
    assert(len(conv_cases)==len(conv_depth_flag))
#import sys
#sys.exit(0)

    config_file = get_raw_data(path_conv_cases)

    data_time = []
    for i in range(len(conv_cases)):
        case = conv_cases[i]
        conv_type = conv_depth_flag[i]
        time = test_conv2d(arg, conv_type, *case)
        data_time.append(time)

    #
    print("the total time of conv2ds : " + str(sum(data_time)))
    path_res = './data_results/res_'+ arg.model+'_' +arg.device +'-'+arg.flag +'_'+str(arg.batch)+'batch'+'_'+str(arg.thread)+'thread'+'_all_conv_case.csv'
    write_res(path_res, config_file, data_time)



if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description = "run conv2d model")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86", "x86-avx2", "x86-avx512"])
    parser.add_argument("-t", "--time", default=10, type = int)
    parser.add_argument("-b", "--batch", default=1, type = int)
    parser.add_argument("-thread", "--thread", default=1, type = int)
    parser.add_argument("-path", "--path", default='./data_results/perlayer_all_conv_case.csv', type = str)
    parser.add_argument("-model", "--model", default='resnet50', type = str)
    parser.add_argument("-usedepthlog", "--usedepthlog", default='true', type = str)
    parser.add_argument("-l", "--log", default='false', type = str)
    parser.add_argument("-f", "--flag", default="t4", type = str)
    parser.add_argument("-q", "--quantize", default="No", type = str)
    parser.add_argument("-p", "--profile", default="false", type = str)
    arg = parser.parse_args() 

    get_data_and_test(arg)



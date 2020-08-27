if __name__ == "__main__":

    import argparse
    import os

    import onnx
    import ngraph as ng
    from ngraph_onnx.onnx_importer.importer import import_onnx_model
    import numpy as np
    
    import time
    from utils.utils import *
    
    parser = argparse.ArgumentParser(description = "run onnx ngraph model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86", "arm"])
    parser.add_argument("-t", "--thread", default="multiple", choices=["multiple","single"])
    arg = parser.parse_args() 

    if arg.device=='x86':
        backend_name = 'CPU'
        if arg.thread=='single':
            os.environ["OMP_NUM_THREADS"] = '1'
    if arg.device=='gpu':
        backend_name = 'PlaidML'
        if arg.thread=='single':
            os.environ["OMP_NUM_THREADS"] = '1'
        os.environ["PLAIDML_DEVICE_IDS"] = 'opencl_nvidia_tesla_v100-pcie-32gb.1'
        #os.environ["PLAIDML_DEVICE_IDS"] = 'llvm_cpu.0'

    print(time.strftime("[localtime] %Y-%m-%d %H:%M:%S", time.localtime()) )

    on, input_shape = get_onnx(arg.onnx)
    ng_function = import_onnx_model(on)
    print(ng_function)
    
    runtime = ng.runtime(backend_name)
    func = runtime.computation(ng_function)
    
    assert(len(input_shape) == 1)
    for value in input_shape.values():
        shape = value
    
    print(shape)
    #shape=[1,3,224,224]
    picture = np.ones(shape, dtype=np.float32)
   
    nSteps=15
    avg_time=0
    for i in range(0, nSteps):
        time1 = time.time()
        ret = func(picture)
        time2 = time.time()
        if i < 5:
            continue
        avg_time+=float(time2-time1)
        info = '-- %d, iteration time(s) is %.4f' %(i, float(time2-time1))
        print(info)
    avg_time = avg_time/10

    name = os.path.basename(arg.onnx)
    print("@@ %s, average time(s) is %.4f" % (name, avg_time))
    print('FINISH')

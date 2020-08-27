import sys
sys.path.append("../")
from utils.utils import *
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "run xla test")
    parser.add_argument("-d", "--device", choices=["x86", "gpu"])
    parser.add_argument("-t", "--thread", default="multiple", choices=["multiple","single"])
    arg = parser.parse_args() 

    onnx_path='~/hub'

    if arg.device=='x86':
        device='x86'
        log_path='../logs/Appendix_xla2_broadwell_5_15'
        if arg.thread=='single':
            log_path+='_single'
    if arg.device=='gpu':
        device='gpu'
        log_path='../logs/Appendix_xla2_v100_5_15'
        if arg.thread=='single':
            log_path+='_single'

    mkdir(log_path)

    
    f = listdir(onnx_path, '')

    for i, name in enumerate(f):
        on = os.path.join(onnx_path, name.strip())
        cmd     = 'python tf2xla.py %s -d %s -t %s' % (on, device, arg.thread)
        cmd     += ' 2>&1 | tee ' + os.path.join(log_path,name.strip())

        print('##NO %d' % i)
        print(cmd)
        os.system(cmd)

    for i, name in enumerate(['densenet121', 'vgg16', 'vgg19']):
        on = name
        cmd     = 'python tf2xla.py %s -d %s -t %s' % (on, device, arg.thread)
        cmd     += ' 2>&1 | tee ' + os.path.join(log_path,name.strip())

        print('##NO %d' % (len(f)+i))
        print(cmd)
        os.system(cmd)
 
        

import sys, argparse
sys.path.append("../")
from utils.utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "run ngraph test")
    parser.add_argument("-d", "--device", choices=["x86", "gpu"])
    parser.add_argument("-t", "--thread", default="multiple", choices=["multiple","single"])
    arg = parser.parse_args() 

    onnx_path='~/onnx_2ng_0706/'

    if arg.device=='x86':
        device='x86'
        log_path='../logs/Appendix_ngraph_broadwell_5_15'
        if arg.thread=='single':
            log_path+='_single'
    if arg.device=='gpu':
        device='gpu'
        log_path='../logs/Appendix_ngraph_v100_5_15'
        if arg.thread=='single':
            log_path+='_single'

    mkdir(log_path)
    
    f = listdir(onnx_path)
    #f = ['resnet50-v1-7']
    #f = ['resnet50_v2']
    for i, name in enumerate(f):
        on = os.path.join(onnx_path, name.strip()+'.onnx')
        cmd     = 'python ng.py %s -d %s -t %s' % (on, device, arg.thread)
        cmd     += ' 2>&1 | tee ' + os.path.join(log_path,name.strip())

        print('##NO %d' % i)
        print(cmd)
        os.system(cmd)
            
        

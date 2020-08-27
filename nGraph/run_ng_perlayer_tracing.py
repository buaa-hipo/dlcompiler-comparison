import sys, argparse, time
sys.path.append("../")
from utils.utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "run ngraph test")
    parser.add_argument("-d", "--device", choices=["x86", "gpu"])
    parser.add_argument("-t", "--thread", default="multiple", choices=["multiple","single"])
    arg = parser.parse_args() 

    onnx_path='../micro-models/mobilenet/'

    timeline_path='../logs/ng_timelines/mobilenet/'

    mkdir(timeline_path)

    if arg.device=='x86':
        device='x86'
    if arg.device=='gpu':
        device='gpu'

    f = listdir(onnx_path)
    #f = ['resnet50-v1-7']
    #f = ['resnet50_v2']
    os.system('rm Function_0.timeline.json')
    for i, name in enumerate(f):
        on = os.path.join(onnx_path, name.strip()+'.onnx')
        json = os.path.join(timeline_path, name.strip()+'.json')
        cmd     = 'NGRAPH_CPU_TRACING=1 '
        cmd     += 'python ng.py %s -d %s -t %s' % (on, device, arg.thread)

        cmd2     = 'mv Function_0.timeline.json %s' % json

        print('##NO %d' % i)
        print(cmd)
        os.system(cmd)
        time.sleep(0.1)
        print(cmd2)
        os.system(cmd2)
            
        

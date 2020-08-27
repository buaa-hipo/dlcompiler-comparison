import sys, argparse
sys.path.append("../")
from utils.utils import *

if __name__ == "__main__":

    onnx_path   ='./mobilenet/'
    pb_path     ='./mobilenet_pb/'

    f = listdir(onnx_path)
    for i, name in enumerate(f):
        on = os.path.join(onnx_path, name.strip()+'.onnx')
        pb = os.path.join(pb_path, name.strip()+'.pb')
        cmd = 'onnx-tf convert -i %s -o %s' % (on, pb)
        print('##NO %d' % i)
        print(cmd)
        os.system(cmd)
            
        

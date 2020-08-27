import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import os, argparse


def tf2xla(sm_path, nSteps=15):
    if sm_path == 'densenet121':
      model = tf.keras.applications.DenseNet121(weights='imagenet', classes=1000)
    elif sm_path == 'vgg16':
      model = tf.keras.applications.VGG16(weights='imagenet', classes=1000)
    elif sm_path == 'vgg19':
      model = tf.keras.applications.VGG19(weights='imagenet', classes=1000)
    
    else:
      model = tf.keras.Sequential([
          hub.KerasLayer(sm_path)
          ])
    
    shape=[1,224,224,3]
    picture = np.ones(shape, dtype=np.float32)
    
    avg_time=0
    for i in range(0, nSteps):
      time1 = time.time()
      ret = model.predict(picture, batch_size=1)
      time2 = time.time()
      if i < 5:
        continue
      avg_time += float(time2-time1)
      info = '-- %d, iteration time(s) is %.4f' %(i, float(time2-time1))
      print(info)
    
    avg_time = avg_time / (nSteps-5)
    name = os.path.basename(sm_path)
    print("@@ %s, average time(s) is %.4f" % (name, avg_time))
    print('FINISH')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "run onnx xla model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86", "arm"])
    parser.add_argument("-t", "--thread", default="multiple", choices=["multiple","single"])
    arg = parser.parse_args() 

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True) # Enable XLA


    #if arg.device=='x86':
    if arg.thread=='single':
      tf.config.threading.set_inter_op_parallelism_threads(1)
      tf.config.threading.set_intra_op_parallelism_threads(1)

    
    print(time.strftime("[localtime] %Y-%m-%d %H:%M:%S", time.localtime()) )

    tf2xla(arg.onnx) 

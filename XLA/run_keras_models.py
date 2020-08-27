import tensorflow as tf
import numpy as np
import os, argparse


parser = argparse.ArgumentParser(description = "run tf nvtx model")
parser.add_argument("-m", "--model", choices=["resnet", "mobilenet"])
arg = parser.parse_args() 
        
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA

if arg.model == 'resnet':
    from resnet import ResNet50
    model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=(224, 224, 3),
            pooling=None, classes=1000)

if arg.model == 'mobilenet':
    from mobilenet_v2 import MobileNetV2
    model = MobileNetV2(alpha=1.0, include_top=True, weights=None, input_tensor=None, 
            pooling=None, classes=1000, classifier_activation='softmax', input_shape=(224, 224, 3))

model.summary()

shape=[1,224,224,3]
picture = np.ones(shape, dtype=np.float32)

nSteps=50
for i in range(0, nSteps):
    ret = model.predict(picture1, batch_size=1)


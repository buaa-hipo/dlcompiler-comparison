import tensorflow as tf
import numpy as np
import os, argparse
from datetime import datetime


parser = argparse.ArgumentParser(description = "run tf nvtx model")
parser.add_argument("-m", "--model", choices=["resnet", "mobilenet"])
arg = parser.parse_args() 
        
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"


tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA

if arg.model == 'resnet':
    from tensorflow.keras.applications import ResNet50
    model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=(224, 224, 3),
            pooling=None, classes=1000)

if arg.model == 'mobilenet':
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(alpha=1.0, include_top=True, weights=None, input_tensor=None, 
            pooling=None, classes=1000, classifier_activation='softmax', input_shape=(224, 224, 3))

model.summary()

shape=[1,224,224,3]
picture = np.ones(shape, dtype=np.float32)
#picture1 = np.random.rand(1,224,224,3)

nSteps=50
for i in range(0, nSteps):
    #picture = np.random.rand(1,224,224,3)
    if i==nSteps-1:
        tf.profiler.experimental.start('mobilenet_logdir'+str(i))
    ret = model.predict(picture, batch_size=1)
    if i==nSteps-1:
        tf.profiler.experimental.stop()


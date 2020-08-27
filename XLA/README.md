# XLA

## Run end-to-end evaluation

```bash
# run a single onnx model
python tf2xla.py /path/to/onnx/file --device ['x86', 'gpu'] --thread ['single', 'multiple']

# run onnx models in a dictory
# Firstly, modify the onnx_path in the following .py file
python run_tf2xla_end2end.py --device ['x86', 'gpu'] --thread ['single', 'multiple']
```

## Profiling Tensorflow and XLA

### Reference

[NVIDIA DLProf]({https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/)

[tensorflow.profiler.experimental](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental)

### Profiling with DLProf
```bash
# requirements: nvidia-docker

docker pull nvcr.io/nvidia/tensorflow:20.07-tf1-py3
docker run --rm --gpus=1 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it -p6006:6006 -v/path/to/local/dir:/path/to/mapped/dir nvcr.io/nvidia/tensorflow:20.07-tf1-py3

# turn off automatic mixed precision
export TF_ENABLE_AUTO_MIXED_PRECISION=0

# run NN model
dlprof python run_keras_models.py -m ['mobilenet', 'resnet']

# analyze the profiled data 
dlprof reports=detail file_formats =csv
dlprof --reports=detail --nsys_database=nsys_profile.sqlite
# get the dlprof_detailed.csv
```

### Profiling with tensorflow.profiler.experimental
```bash
python run_tfprofiler.py  -m ['mobilenet', 'resnet']

# visualize with tensorboard
tensorboard --logdir=/path/to/your/log --port=6006 --host=0.0.0.0
```

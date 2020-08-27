# nGraph

## Run end-to-end evaluation
```bash
# run a single onnx model
python ng.py /path/to/onnx/file --device ['x86', 'gpu'] --thread ['single', 'multiple']

# run onnx models in a dictory
# Firstly, modify the onnx_path in the following .py file
python run_ng_end2end.py --device ['x86', 'gpu'] --thread ['single', 'multiple']
```
## Run per-layer evaluation
```bash
# generate the per-layer onnx micro-models
cd micro-model
python ['gen_conv_mobilenetv1_1.0.py', 'gen_conv_resnet50.py']

# nGraph can only generate timelines on CPU
# modify the onnx_path in the following .py file
python run_ng_perlayer.py --device ['x86', 'gpu'] --thread ['single', 'multiple']
```


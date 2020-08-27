./bin/image-classifier ./images/cat_285.png -image-mode=0to1 -m mobilenetv2_1.0.onnx -model-input-name=data -backend=$1 --trace-path=mobilenet.json --auto-instrument

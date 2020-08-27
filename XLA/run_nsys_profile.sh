#!/usr/bin/env bash

export CUDA_LAUNCH_BLOCKING=1

nsys profile \
  -d 60 \
  -w true \
  --force-overwrite=true \
  --sample=cpu \
  -t 'nvtx,cuda,cublas,cudnn,openmp' \
  --stop-on-exit=true \
  --kill=sigkill \
  -o $1"_nvtx_more" \
  python run_keras_models.py -m $1

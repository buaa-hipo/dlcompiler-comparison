
#skylake-2080ti
#export TVM_HOME=/root/lib/tvm-0.6

#broadwell-v100
export TVM_HOME=/root/tvm-0.6

#export TVM_HOM=/root/tvm-0.7
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
export TVM_NUM_THREADS=1

## -->> resnet50 <<-- ##

## x86-avx2
#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f broadwell -p false -thread 1  -model resnet50 -l ./logs/broadwell_v100/x86_onnx_1batch_1thread_resnet50.log > res/broadwell_v100/res_x86_broadwell_1batch_1thread_resnet50_tuned_ljj

#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f broadwell -p false -thread 28  -model resnet50 -l ./logs/broadwell_v100/x86_onnx_1batch_multhread_resnet50.log > res/broadwell_v100/res_x86_broadwell_1batch_multhread_resnet50_tuned_ljj


# gpu-v100
#python3 conv2d_depth_profile.py -d gpu -t 200 -b 1 -f v100 -p false -model resnet50 -l ./logs/broadwell_v100/gpu_onnx_1batch_resnet50.log > res/broadwell_v100/res_gpu_1batch_resnet50_tuned_ljj

## -->> mobilenet <<-- ##

## x86-avx2
#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f broadwell -p false -thread 1 -usedepthlog false -path ./data_results/perlayer_conv2d_depthwise_case.csv -model mobilenetv2_1.0 -l ./logs/broadwell_v100/x86_onnx_1batch_1thread_mobilenetv2_1.0.log > res/broadwell_v100/res_x86_broadwell_1batch_1thread_mobilenetv2_1.0_tuned_ljj

#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f broadwell -p false -thread 28 -usedepthlog false -path ./data_results/perlayer_conv2d_depthwise_case.csv -model mobilenetv2_1.0 -l ./logs/broadwell_v100/x86_onnx_1batch_multhread_mobilenetv2_1.0.log > res/broadwell_v100/res_x86_broadwell_1batch_multhread_mobilenetv2_1.0_tuned_ljj

# gpu-v100
python3 conv2d_depth_profile.py -d gpu -t 200 -b 1 -f v100 -p false -path ./data_results/perlayer_conv2d_depthwise_case.csv -model mobilenetv2_1.0 -l ./logs/broadwell_v100/gpu_onnx_1batch_mobilenetv2_1.0.log > res/broadwell_v100/res_gpu_1batch_mobilenet_tuned_ljj


## -->> resnet50 <<-- ##

## x86-avx512
# avx-512 is not set when tuning
#python3 conv2d_depth_profile.py -d x86-avx512 -t 200 -b 1 -f skylake -p false -thread 1 -l ./logs/skylake_2080ti/x86_onnx_1batch_1thread_resnet50.log > res/skylake_2080ti/res_x86_skylake_1batch_1thread_resnet50_tuned_ljj
# use this one
#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f skylake -p false -thread 1 -l ./logs/skylake_2080ti/x86_onnx_1batch_1thread_resnet50.log > res/skylake_2080ti/res_x86_skylake_1batch_1thread_resnet50_tuned_ljj

#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f skylake -p false -thread 16 -l ./logs/skylake_2080ti/x86_onnx_1batch_Multhreads_resnet50.log > res/skylake_2080ti/res_x86_skylake_1batch_Multhreads_resnet50_tuned_ljj

# gpu-2080ti
#python3 conv2d_depth_profile.py -d gpu -t 200 -b 1 -f 2080ti -p false -l ./logs/skylake_2080ti/gpu_onnx_1batch_resnet50.log > res/skylake_2080ti/res_gpu_1batch_resnet50_tuned_ljj


## -->> mobilenet <<-- ##

# x86-avx512
#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f skylake -p false -thread 1 -usedepthlog false -path ./data_results/perlayer_conv2d_depthwise_case.csv -model mobilenetv2_1.0 -l ./logs/skylake_2080ti/x86_onnx_1batch_1thread_mobilenetv2_1.0.log > res/skylake_2080ti/res_x86_skylake_1batch_1thread_mobilenetv2_1.0_tuned_ljj

#python3 conv2d_depth_profile.py -d x86-avx2 -t 200 -b 1 -f skylake -p false -thread 16 -usedepthlog false -path ./data_results/perlayer_conv2d_depthwise_case.csv -model mobilenetv2_1.0 -l ./logs/skylake_2080ti/x86_onnx_1batch_Multhreads_mobilenetv2_1.0.log > res/skylake_2080ti/res_x86_skylake_1batch_Multhreads_mobilenetv2_1.0_tuned_ljj

# gpu-2080ti
#python3 conv2d_depth_profile.py -d gpu -t 200 -b 1 -f 2080ti -p false -path ./data_results/perlayer_conv2d_depthwise_case.csv -model mobilenetv2_1.0 -l ./logs/skylake_2080ti/gpu_onnx_1batch_mobilenetv2_1.0.log > res/skylake_2080ti/res_gpu_1batch_mobilenetv2_1.0_tuned_ljj







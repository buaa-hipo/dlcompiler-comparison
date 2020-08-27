export TVM_NUM_THREADS=1

log_path=./logs/tvm-x86-1thread-V100-tuned
for i in `cat list`
do
    python3 tvm_tuned.py -m models/$i.onnx -t x86 -l ./tvm-log/1thread-log/x86/onnx/1batch/x86_onnx_1batch_$i.log | tee $log_path/$i
done

unset TVM_NUM_THREADS

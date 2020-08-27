log_path=../logs/glow-$1-2080Ti
for i in `cat ../utils/list`
do
	echo $i | tee -a $log_path/$i
	./bin/image-classifier ./images/cat_285.png -image-mode=0to1 -m ./models/${i}.onnx -model-input-name=data -backend=$1 | tee -a $log_path/$i
done

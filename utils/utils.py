import os, sys

def get_onnx(path, batch=1):
    import onnx
    on = onnx.load(open(path, "rb"))
    name = on.graph.input[0].name
    input_shape = [i.dim_value for i in  on.graph.input[0].type.tensor_type.shape.dim]  
    input_shape[0] = batch
    return on, {name : input_shape}

def convert_onnx_tf(onnx_path, pb_path):
    import onnx
    from onnx_tf.backend import prepare
    
    onnx_model = onnx.load(onnx_path)  # load onnx model
    tf_rep = prepare(onnx_model, strict=False)  # prepare tf representation
    
    tf_rep.export_graph(pb_path)  # export the model

def mkdir(path):
    path=path.strip()
    path=path.rstrip("/")
    isExists=os.path.exists(path)
 
    if not isExists:
        os.makedirs(path) 
 
        print(path+' create folder')
        return True
    else:
        print(path+' already exists')
        return False

# 列出文件夹下的所有后缀为suffix的文件（文件夹留空）
def listdir(path, suffix='.onnx'):
    list_name = []
    for file in os.listdir(path):
        #file_path = os.path.join(path, file)
        #if os.path.isdir(file_path):
        #    listdir(file_path, list_name)
        if os.path.splitext(file)[1]==suffix:
            f = os.path.splitext(file)[0]
            f = os.path.basename(f)
            list_name.append(f)
    return list_name



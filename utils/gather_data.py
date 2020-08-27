# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import argparse
# 导入本py文件所在目录下的utils.py
from utils import *

## Log输出格式
'''
[localtime] 2020-07-16 11:51:46
<Function: 'mxnet_converted_model' ([1, 1000])>
[1, 3, 224, 224]
-- 5, iteration time(s) is 0.0149
-- 6, iteration time(s) is 0.0148
-- 7, iteration time(s) is 0.0149
-- 8, iteration time(s) is 0.0151
-- 9, iteration time(s) is 0.0150
-- 10, iteration time(s) is 0.0149
-- 11, iteration time(s) is 0.0149
-- 12, iteration time(s) is 0.0148
-- 13, iteration time(s) is 0.0145
-- 14, iteration time(s) is 0.0118
@@ resnet50_v2.onnx, average time(s) is 0.0146
FINISH
'''

def get_time(fileName=""):

  # 当跑15轮，抛弃前5轮做warmup时
  # 共有10个iteration time， 1个average time
  # 存到len=11的data list中
  data = []
  file = open(fileName) 
  for line in file:
    s='time(s) is '
    fd = line.find(s)
    first = fd+len(s)
    if fd != -1:
        time = float(line[first:])
        data.append(time)
  file.close()    

  try:
    assert(len(data)==11)
  except:
    data = [-1 for i in range(0,11)]

  return data

def get_time_dict(log_path):
    temp = {}
    for file in os.listdir(log_path):
        time = get_time(os.path.join(log_path, file))
        temp.update({file:time})
    return temp
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "collect and get csv")
    parser.add_argument("dir", help = "log path")
    parser.add_argument("-l","--list", help = "list path")
    arg = parser.parse_args()

    log = arg.dir
    csv = os.path.basename(log)
    data = get_time_dict(log)

    f = open(arg.list, 'r')
    s = {}
    a = list(f)
    for i in a:
        key = i.strip()
        if key in data:
            s.update({key:data[key]})
    f.close()
    print(s)

    data = pd.DataFrame.from_dict(data=s, orient='index')

    label = ['it'+str(i) for i in range(0,10)]
    label.append('avg')
    data.columns=label


    data.to_csv('csv/' + csv  +'.csv')

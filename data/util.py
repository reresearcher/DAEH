# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/4/8 9:34
# @Author : Yufeng Shi
# @Email : yufengshi17@hust.edu.cn
# @File : util.py
# @Software: PyCharm


import numpy as np
import pickle as pkl

def range_data(data):
    data = np.array(data).astype(np.int) - 1
    num_data = max(data.shape)
    return np.reshape(data, (num_data,))

def find_names(txt_path):
    data_names = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            all_res = line.split(' ')
            data_names.append(all_res[0])
    return data_names

def read_pkl(file_name, target):

    output = open(file_name, 'rb')
    tmp = pkl.load(output)
    output.close()
    return tmp[target]
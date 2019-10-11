
import os
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def load_mat_func(path):
    """
    :param path:
    :return:  (6*2400, 2048),or (6*1200, 2048)
    """
    data = loadmat(path)
    all_data = [
        data['data0'],
        data['data1'],
        data['data2'],
        data['data3'],
        data['data4'],
        data['data5']]
    all_data = np.concatenate(all_data)
    return all_data


GT_normal = load_mat_func('raw_data\\GT_normal_data')            # (14400,2048)
MT3_normal_max = load_mat_func('raw_data\\MT3_max_norm_data')    # (7200, 2048)
MT3_normal_min = load_mat_func('raw_data\\MT3_min_norm_data')    # (7200, 2048)

GT_Inner = load_mat_func('raw_data\\GT_Inner_Raceway_data')
MT3_Inner_max = load_mat_func('raw_data\\MT3_max_inner_600_data')
MT3_Inner_min = load_mat_func('raw_data\\MT3_min_Inner_Raceway_data')

GT_Outer = load_mat_func('raw_data\\GT_Outer_Raceway_data')
MT3_Outer_max = load_mat_func('raw_data\\MT3_max_outer_600_data')
MT3_Outer_min = load_mat_func('raw_data\\MT3_min_Outer_Raceway_data')

data = np.concatenate((GT_normal, MT3_normal_max, MT3_normal_min,
                       GT_Inner, MT3_Inner_max, MT3_Inner_min,
                       GT_Outer, MT3_Outer_max, MT3_Outer_min),
                      axis=0)
print("data.shape:  ",np.array(data).shape)    # (86400, 2048) = (3*28800, 2048)


normal_label = np.array([1,0,0] * 28800)       # (28800, 3)
normal_label = normal_label.reshape(28800, -1)
print("normal_label.shape",normal_label.shape)

Inner_label = np.array([0,1,0] * 28800)
Inner_label = Inner_label.reshape(28800, -1)
print("Inner_label.shape",Inner_label.shape)

Outer_label = np.array([0,0,1] * 28800)
Outer_label = Outer_label.reshape(28800, -1)
print("Outer_label.shape",Outer_label.shape)

label = np.concatenate((normal_label, Inner_label,Outer_label), axis = 0)
print("label.shape", label.shape)            # (86400, 3)

data_set = {'data' : data,
            'label': label}
io.savemat('preprocessed_data', data_set)




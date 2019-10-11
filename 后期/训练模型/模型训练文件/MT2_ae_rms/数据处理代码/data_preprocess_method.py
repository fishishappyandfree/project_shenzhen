"""
【数据预处理思路】
1 待处理的数据为MxN的二维数组
2 通过reshape函数，将二维数组变成mxTxN的三维数组，其中M=mxT
3 对第一维数组进行处理，得到TxN的二维数组
"""

import numpy
import h5py
from scipy.io import savemat

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        normal_train_data = f['normal_train_data'][:]
        normal_train_label = f['normal_train_label'][:]
        normal_test_data = f['normal_test_data'][:]
        normal_test_label = f['normal_test_label'][:]

        inner_train_data = f['inner_train_data'][:]
        inner_train_label = f['inner_train_label'][:]
        inner_test_data = f['inner_test_data'][:]
        inner_test_label = f['inner_test_label'][:]

        outer_train_data = f['outer_train_data'][:]
        outer_train_label = f['outer_train_label'][:]
        outer_test_data = f['outer_test_data'][:]
        outer_test_label = f['outer_test_label'][:]

    train_data = numpy.concatenate([normal_train_data, inner_train_data, outer_train_data], axis=0)
    train_label = numpy.concatenate([normal_train_label, inner_train_label, outer_train_label], axis=0)
    test_data = numpy.concatenate([normal_test_data, inner_test_data, outer_test_data], axis=0)
    test_label = numpy.concatenate([normal_test_label, inner_test_label, outer_test_label], axis=0)
    return  train_data, train_label, test_data, test_label



def process_method(data, slice_len=78, method='rms'):
    '''
    数据处理方法，待输入数组应该为3D，默认支持rms,输出数组为所需要的2D数组，不需要reshape
    '''
    if data.shape[1] % slice_len != 0:
        raise RuntimeError('slice_len '+ slice_len +  '选择错误')

    data = data.reshape([data.shape[0], slice_len, -1])
    if method == 'rms':
        return numpy.sqrt(numpy.mean(numpy.square(data), axis=1))   # 2D
    elif method == 'mean':
        return numpy.mean(data, axis=1)
    elif method == 'median':
        return numpy.median(data, axis=1)
    else:
        raise NotImplementedError('Method ', method, 'has not been implemented!')

# 定义保存的文件夹
file_path = '/media/hust/f3fb8681-0cb6-4f0d-9e16-02d478f231c4/新建文件夹2/ae_analysis/ae_MT2/MT2_ae_dataset.h5'
# 加载数据
train_data_raw, train_label, test_data_raw, test_label = load_data(file_path)

mat_path_rms = '/media/hust/f3fb8681-0cb6-4f0d-9e16-02d478f231c4/新建文件夹2/ae_analysis/ae_MT2/MT2_ae_dataset_rms'
train_data_rms = process_method(train_data_raw, method='rms')
test_data_rms = process_method(test_data_raw, method='rms')
rms_dict = {
    'train_data'  : train_data_rms,
    'train_label' : train_label,
    'test_data'   : test_data_rms,
    'test_label'  : test_label
}
del train_data_rms
del test_data_rms
savemat(mat_path_rms, rms_dict)
del rms_dict

mat_path_mean = '/media/hust/f3fb8681-0cb6-4f0d-9e16-02d478f231c4/新建文件夹2/ae_analysis/ae_MT2/MT2_ae_dataset_mean'
train_data_mean = process_method(train_data_raw, method='mean')
test_data_mean = process_method(test_data_raw, method='mean')
mean_dict = {
    'train_data'  : train_data_mean,
    'train_label' : train_label,
    'test_data'   : test_data_mean,
    'test_label'  : test_label
}
del train_data_mean
del test_data_mean
savemat(mat_path_mean, mean_dict)
del mean_dict

mat_path_median = '/media/hust/f3fb8681-0cb6-4f0d-9e16-02d478f231c4/新建文件夹2/ae_analysis/ae_MT2/MT2_ae_dataset_median'
train_data_median = process_method(train_data_raw, method='median')
test_data_median = process_method(test_data_raw, method='median')
median_dict = {
    'train_data'  : train_data_median,
    'train_label' : train_label,
    'test_data'   : test_data_median,
    'test_label'  : test_label
}
del train_data_median
del test_data_median
savemat(mat_path_median, median_dict)
del median_dict

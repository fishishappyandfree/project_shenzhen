#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
# import time
import numpy
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.fftpack import fft
import collections
import json
import shutil
import itertools

import tensorflow as tf
from sklearn import metrics

from CNN_model import CNN
from Config import Config

class Model(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.env   # 确定使用哪块GPU
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 程序本身相关信息打印程度

        self.save_dir = 'checkpoints'
        self.save_path = os.path.join(self.save_dir, 'best_validation')  # 最佳验证结果保存路径
        self.save_result = 'results'
        self.gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.gpu_config.gpu_options.allow_growth = True


    # sys.setrecursionlimit(900000)

    # 获取频谱
    def fft_trans(self, array):
        N = len(array)
        return abs(numpy.fft.fft(array))[:, : int(N/2)] / N

    # 获取包络谱
    def enelop_trans(self, array):
        array_h = numpy.imag(hilbert(array))
        array_ene = abs(fft((array ** 2 + array_h ** 2) ** 0.5)) / len(array[0])
        array_ene[:, 0] = 0
        return array_ene[:, :int(len(array[0]) / 2)]


    # 打乱顺序
    def shuffle(self, data, label):
        '''
        打乱下标，打乱数据
        '''
        indices = numpy.random.permutation(numpy.arange(len(label)))
        return data[indices], label[indices]


    # deed dict generator
    def feed_test_data(self, x_batch, dropout_keep_prob_fc, is_training):
        '''
        generate the feed data dict object
        '''
        feed_dict = {self.model.input_x: x_batch,
                    self.model.dropout_keep_prob_fc:dropout_keep_prob_fc,
                    self.model.is_training: is_training }
        return feed_dict


    # 注意函数参数里必须有self
    def run_test(self, x_test):
        #self.json_config = os.path.join(self.save_result, 'config.json')
        print("Predicting data...")
        x_test = numpy.array(x_test).reshape(-1,2048)
        x_test = enelop_trans(x_test)
        print('X_test shape is: ', x_test.shape)
        self.config = Config()
        self.model = CNN(self.config)
        with tf.Session(config=self.gpu_config) as sess_test:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            saver.restore(sess=sess_test, save_path=self.save_path)  # 读取保存的模型
            feed_dict = feed_test_data(x_test, 1.0, False)   # EVAL: keep_prob=1.0, valuation & testing are not during training.
            y_test_pred = sess_test.run(self.model.y_pred, feed_dict=feed_dict)
            print('y_test_pred',y_test_pred)

            
if __name__ == "__main__":
    data = numpy.array(range(2048)).reshape(1,2048)
    model = Model()
    model.run_test(data)
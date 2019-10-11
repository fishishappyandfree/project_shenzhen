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

from models.MT2_spindle_z.CNN_model import CNN
from models.MT2_spindle_z.Config import Config

class Model(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.env   # 确定使用哪块GPU
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 程序本身相关信息打印程度

        #self.save_dir = 'checkpoints'
        #self.save_path = os.path.join(self.save_dir, 'best_validation')  # 最佳验证结果保存路径
        self.save_result = 'results'
        self.gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.gpu_config.gpu_options.allow_growth = True


    # sys.setrecursionlimit(900000)

    def normalize(self, array):
        """
        对所有样本的每一个维度进行归一化  也就是归一化处理次数 = 特征维度 = 2048（每一个点都是一个维度）
        归一化处理 #经过处理的数据符合标准正态分布，即均值为0，标准差为1
        z-score标准化方法适用于属性A的最大值和最小值未知的情况，或有超出取值范围的离群数据的情况。
        该种归一化方式要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。
        """
        # plt.figure(1)
        # plt.subplot(2,1,1)
        # sample_index = 1         # 显示哪个样本的索引
        # plt.plot(array[sample_index,:])
        #print("array.shape before normalize:" , array.shape)  # (14400, 2048)
        #print("array.before normalize", array[1:5,:])
        column_mean = numpy.mean(array, axis = 1)  # 对每列,即所有样本的每个维度求均值
        sigma = numpy.std(array, axis=1)
        #print("column_mean.shape",column_mean.shape) # (2048,) 对每一个维度进行运算，总共2048个维度
        #print("sigma.shape",sigma.shape)             # (2048,)

        norm_array = ((array - column_mean)/sigma)
        # print("array.shape after normalize:" , norm_array.shape)  # (14400, 2048)
        # plt.subplot(2,1,2)
        # plt.plot(norm_array[sample_index,:])
        # print("array.after normalize", norm_array[1:5,:])
        # plt.show()
        return norm_array

    
    def batch_iter(self, data, label, batch_size):  
        data_len = len(label)
        num_batch = int((data_len - 1) / batch_size) + 1

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield data[start_id:end_id], label[start_id:end_id]
        
        
    # deed dict generator
    def feed_data_train(self, x_batch, dropout_keep_prob_fc, is_training):
        '''
        generate the feed data dict object
        '''
        feed_dict = {self.model_train.input_x: x_batch,
                    self.model_train.dropout_keep_prob_fc:dropout_keep_prob_fc,
                    self.model_train.is_training: is_training }
        return feed_dict

        
    def feed_data_test(self, x_batch, dropout_keep_prob_fc, is_training):
        '''
        generate the feed data dict object
        '''
        feed_dict = {self.model_test.input_x: x_batch,
                    self.model_test.dropout_keep_prob_fc:dropout_keep_prob_fc,
                    self.model_test.is_training: is_training }
        return feed_dict
        
    def run_train(self, samples_train, labels_train, checkpoint_path_folder):
        '''
        samples_train: 是一个二维数组,(500,2048)
        labels_train: 是一个二维数组,(500,3)
        '''
        self.save_path_train = os.path.join(checkpoint_path_folder, 'best_validation')  
        samples_train = np.array(samples_train)
        labels_train = np.array(labels_train)
        self.config_train = Config()
        self.config_train.learning_rate = 0.008  
        self.config_train.dropout_keep_prob_fc = 1 
        self.config_train.batch_size = 64
        self.model_train = CNN(self.config_train)

        with tf.Session(config=self.gpu_config) as sess_train:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            # 读取上一次保存的checkpoint文件
            saver.restore(sess=sess_train, save_path=self.save_path_train)  

            num_batch = int(len(samples_train) / (self.config_train.batch_size)) 
            batch_train = self.batch_iter(samples_train, labels_train, self.config_train.batch_size)  
            num_iteration = 0 

            for x_batch, y_batch in batch_train:
                num_iteration = num_iteration + 1

                feed_dict = self.feed_data_train(x_batch, y_batch, self.config_train.dropout_keep_prob_fc, True)  # Training
                optim, loss_train, acc_train = sess_train.run([self.model_train.optim,
                                                         self.model_train.loss,
                                                         self.model_train.acc],
                                                         feed_dict=feed_dict)
        
                #print("batch:  %03d/%03d  train_loss: %.9f train_acc: %.9f" % 
                   # (num_iteration, num_batch, loss_train, acc_train))
            # 保存 checkpoint folder , self.save_path是保存地址
            saver.save(sess=sess_train, save_path=self.save_path_train)
        return 'v1.1'
        
    
    # 注意函数参数里必须有self
    def run_test(self, x_test, checkpoint_path_folder):
        #self.json_config = os.path.join(self.save_result, 'config.json')
        #print("Predicting data...")
        self.save_path_test = os.path.join(checkpoint_path_folder, 'best_validation')  # 最佳验证结果保存路径
        x_test = numpy.array(x_test).reshape(-1,2048)
        #x_test = self.normalize(x_test)
        #print('X_test shape is: ', x_test.shape)
        self.config_test = Config()
        self.model_test = CNN(self.config_test)
        #fault_list = ['normal', 'inner', 'outer']  # 共可能产生的故障类别
        with tf.Session(config=self.gpu_config) as sess_test:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            # 加载保存的模型
            saver.restore(sess=sess_test, save_path=self.save_path_test)  # 读取保存的模型
            feed_dict = self.feed_data_test(x_test, 1.0, False)   # EVAL: keep_prob=1.0, valuation & testing are not during training.
            #y_test_pred = sess_test.run(self.model.result, feed_dict=feed_dict)
            # 由输入的 x_test共2048个数据点 得到故障概率标签 ， 一维数组 （3，）[norm_pro, inner_pro, outer_pro]
            fault_probility_lable = numpy.array(sess_test.run(self.model_test.pred_label, feed_dict=feed_dict)).flatten() #shape:(3，)[norm_pro, inner_pro, outer_pro]
            print(fault_probility_lable)
            # 得到概率最大的故障标签指引
            index = numpy.argmax(fault_probility_lable)
            # 由指引得到故障类别
            #fault_pred_class = fault_list[index]
            # 显示故障概率标签中概率最大的值
            show_pro_fault_pred_class = fault_probility_lable[index]
            #print('y_test_pred',y_test_pred)
            return index, show_pro_fault_pred_class

            
if __name__ == "__main__":
    data = numpy.array(range(2048)).reshape(1,2048)
    model = Model()
    model.run_test(data)

# coding: utf-8
from os import environ
import tensorflow as tf
import Config


class CNN(object):
    '''分类，CNN模型'''

    def __init__(self, Config):
        self.config = Config
        tf.reset_default_graph()
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.dropout_keep_prob_fc = tf.placeholder(tf.float32, name='dropout_keep_prob_fc')
        self.global_step = tf.Variable(0, trainable=False)
        # tf.reset_default_graph()
        self.input_x = tf.placeholder(
            tf.float32, [None, self.config.seq_width * self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, self.config.num_classes], name='input_y')
        self.cnn()

    def cnn_2d_nn(self, input_, output_dim, filter_height=1, filter_width=3, stride_h=1, stride_w=1, initialize_name='xavier_normal', w_para=None,
                b_init=0.0, padding='SAME', a_para=[], name='conv2d'):
        with tf.variable_scope(name):
            w_conv = tf.get_variable('weight', [filter_height, filter_width, input_.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
            b_conv = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(b_init))
            conv = tf.nn.conv2d(input_, w_conv, strides=[1, stride_h, stride_w, 1], padding=padding) + b_conv
        return conv
    
    def full_connect(self, input_, _FC_SIZE=1, initialize_name='xavier_normal', w_para=None, b_init=0.0, name='Full_Connection'):
        with tf.variable_scope(name):
            fc_weight = tf.get_variable(name='weight', shape=[int(input_.get_shape()[-1]), _FC_SIZE], initializer=tf.contrib.layers.xavier_initializer())
            fc_bias = tf.get_variable(name='bias', shape=[_FC_SIZE], initializer=tf.constant_initializer(value=b_init))
            # output
            output_ = tf.nn.xw_plus_b(input_, fc_weight, fc_bias, name=name)
        return output_

    def cnn(self):
        '''build cnn model'''
        with tf.variable_scope('reshape'):
            input_x_reshape = tf.reshape(self.input_x,
                                         [-1, self.config.seq_width,
                                             self.config.seq_length, self.config.seq_height],
                                         name='x-reshape')

        with tf.variable_scope('cnn_layers'):
            self.cnn_0 = self.cnn_2d_nn(input_x_reshape, 
                                        output_dim=self.config.output_dim[0], 
                                        filter_height=self.config.filter_height[0], 
                                        filter_width=self.config.filter_width[0], 
                                        stride_h=self.config.stride_h[0], 
                                        stride_w=self.config.stride_w[0], 
                                        padding='SAME', 
                                        name='conv_0')
            self.activ_0 = tf.nn.elu(self.cnn_0, name='activate_0')
            self.pool_0 = tf.nn.max_pool(self.activ_0, 
                                        ksize=[1, 3, 3, 1], 
                                        strides=[1, 2, 2, 1], 
                                        padding='VALID', 
                                        name='pool_0')
            #print('Pool_0 shape_size is: ', self.pool_0.get_shape().as_list())

            # self.cnn_1 = self.cnn_2d_nn(self.cnn_0, output_dim=self.config.output_dim[1], filter_width=3, stride_h=3, stride_h=2, stride_w=2, padding='SAME', name='conv_1')
            # self.activ_1 = tf.nn.elu(self.cnn_1, name='activate_1')
            # self.pool_1 = tf.nn.max_pool(self.activ_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='VALID', name='pool_1')
            # print('Pool_1 shape_size is: ', self.pool_1.get_shape().as_list())
            self.cnn_1 = self.cnn_2d_nn(self.pool_0, 
                                        output_dim=self.config.output_dim[1], 
                                        filter_height=self.config.filter_height[1], 
                                        filter_width=self.config.filter_width[1], 
                                        stride_h=self.config.stride_h[1], 
                                        stride_w=self.config.stride_w[1], 
                                        padding='SAME', 
                                        name='conv_1')
            self.activ_1 = tf.nn.elu(self.cnn_1, name='activate_1')
            self.pool_1 = tf.nn.max_pool(self.activ_1, 
                                        ksize=[1, 3, 3, 1], 
                                        strides=[1, 2, 2, 1], 
                                        padding='VALID', 
                                        name='pool_1')
            #print('Pool_1 shape_size is: ', self.pool_1.get_shape().as_list())

            self.cnn_2 = self.cnn_2d_nn(self.pool_1, 
                                        output_dim=self.config.output_dim[2], 
                                        filter_height=self.config.filter_height[2], 
                                        filter_width=self.config.filter_width[2], 
                                        stride_h=self.config.stride_h[2], 
                                        stride_w=self.config.stride_w[2], 
                                        padding='SAME', 
                                        name='conv_2')
            self.activ_2 = tf.nn.elu(self.cnn_2, name='activate_2')
            self.pool_2 = tf.nn.avg_pool(self.activ_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='VALID', name='pool_2')
            #print('Pool_2 shape_size is: ', self.pool_2.get_shape().as_list())
        
        with tf.variable_scope('reshape'):
            pool_shape = self.pool_2.get_shape().as_list()   # [batch_size, 1, N]
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            self.cnn_reshape = tf.reshape(self.pool_2, [-1, nodes], name='cnn_reshape')
            #print('cnn_reshape shape_size is: ', self.cnn_reshape.get_shape().as_list())

        with tf.variable_scope("dense"):
            # classifier layer
            self.fc = self.full_connect(self.cnn_reshape, _FC_SIZE=self.config.fc_size, name='fc')
            self.fc_activate = tf.nn.elu(self.fc, name='activate_fc')
            self.fc_dropout = tf.nn.dropout(self.fc_activate, keep_prob=self.dropout_keep_prob_fc)
            #print('fc_layer is: ', self.fc.get_shape().as_list())

        with tf.variable_scope("score"):
            self.logits = self.full_connect(input_=self.fc_dropout, _FC_SIZE=self.config.num_classes, name='logit')
            self.y_pred = tf.nn.softmax(self.logits, name='predict')  # 预测类别

        with tf.name_scope('learning_rate'):
            self.learning_rate_fixed = tf.train.exponential_decay(
                self.config.learning_rate, self.global_step, self.config.decay_steps, self.config.learning_rate_decay, staircase=False)

        with tf.name_scope('loss_function'):
            # Classification
            try:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits, labels=self.input_y), name='cross_entropy_loss')
            except:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.input_y), name='cross_entropy_loss')
            tf.summary.scalar('cross_entropy_loss', self.loss)

        with tf.name_scope('optimize'):
            # 优化器
            ## for short
            # self.optim = tf.train.AdamOptimizer(self.learning_rate_fixed).minimize(self.loss, self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim = tf.train.AdamOptimizer(self.learning_rate_fixed).minimize(self.loss, self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(
                tf.argmax(self.input_y, 1), tf.argmax(self.y_pred, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.acc)

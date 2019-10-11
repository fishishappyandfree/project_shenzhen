"""
【当前模型存在的问题在】
1 在数据预处理的时候  对6小段数据进行数据增强，但是每段数据长度不同，也就会导致它的中间重叠 overlap 的部分不一样长，
  或者说增强的倍数不一样
2 在 run.py 中， 原始代码中， 画图时，横轴其实是 iteration， 而不是 epoch

"""
# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
# import time
import numpy as np
from scipy.io import loadmat
import collections
import json
import shutil
import itertools
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt

from CNN_model import CNN  # 导入CNN类
from Config import Config  # 导入配置文件 Config类

# 另一种操作  with tf.device('/gpu:0')
os.environ['CUDA_VISIBLE_DEVICES'] = Config.env  # 确定使用哪块GPU   env = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 程序本身相关信息打印程度  “0”:INFO   “1”:WARNING;  “2”	ERROR   “3”	FATAL

# train_tensorboard_dir = 'logdir\\train'   # TensorBoard 保存路径  可视化能展示你训练过程中绘制的图像、网络结构等。
# epoch_valid_tensorboard_dir = 'logdir\\epoch_valid'       # 每个epoch valid
# iter_train_tensorboard_dir = 'logdir\\iter_train'   # 每个iteration train

# linux系统文件命名有不同
# epoch_valid_tensorboard_dir = 'logdir/epoch_valid'       # 每个epoch valid
# iter_train_tensorboard_dir = 'logdir/iter_train'   # 每个iteration train
tensorboard_dir = 'logdir'
epoch_valid_tensorboard_dir = 'logdir/epoch_valid'  # 每个epoch valid
iter_train_tensorboard_dir = 'logdir/iter_train'  # 每个iteration train

save_dir = 'checkpoints'  # checkpoint 路径 训练过程中的模型快照。
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
save_result_dir = 'CNN_Model_results'  # 保存文件夹


def normalize(array):
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
    print("array.shape before normalize:", array.shape)  # (14400, 2048)
    # print("array.before normalize", array[1:5,:])
    column_mean = np.mean(array, axis=0)  # 对每列,即所有样本的每个维度求均值
    sigma = np.std(array, axis=0)
    print("column_mean.shape", column_mean.shape)  # (2048,) 对每一个维度进行运算，总共2048个维度
    print("sigma.shape", sigma.shape)  # (2048,)

    norm_array = ((array - column_mean) / sigma)
    # print("array.shape after normalize:" , norm_array.shape)  # (14400, 2048)
    # plt.subplot(2,1,2)
    # plt.plot(norm_array[sample_index,:])
    # print("array.after normalize", norm_array[1:5,:])
    # plt.show()
    return norm_array


# 绘图及保存
def plot_save(array, fig_path=None, xlabel='epochs', ylabel='loss value', dpi=400):
    '''
    Plot and Save Figure
    '''
    # Plot Array
    fig, ax = plt.subplots()
    ax.plot(array)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, (ylabel + '.png')), dpi=dpi)
    # try:
    #     plt.savefig(os.path.join(fig_path, (ylabel + '.svg')))
    #     plt.savefig(os.path.join(fig_path, (ylabel + '.pfg')))
    # except: pass
    plt.close()


def feed_data(x_batch, y_batch, dropout_keep_prob_fc, is_training):
    """
    以字典的形式存储 feed data
    :param x_batch: 模型的输入 input
    :param y_batch: 模型的真实标签
    :param dropout_keep_prob_fc: 神经元保留率
    :param is_training:  判断是训练还是在测试  训练则 True  测试则 FAlse
    :return: feed_dict
    """
    feed_dict = {model.input_x: x_batch,
                 model.real_label: y_batch,
                 model.dropout_keep_prob_fc: dropout_keep_prob_fc,
                 model.is_training: is_training}
    return feed_dict


def batch_iter(data, label, batch_size):  # 类似于mnist数据集中自带的 next_batch  获取下一个batch
    '''
    generating batch data, and return a iteration.
    -------------------------------------------------------------------------------
    '''
    data_len = len(label)
    num_batch = int((data_len - 1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        # len(data)
        yield data[start_id:end_id], label[start_id:end_id]
        # yield 是一个类似 return  的关键字，迭代一次遇到yield时就返回yield后面的值。
        # 重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码开始执行。


def evaluate(sess, merged, x_valid, y_valid):
    """
    经过训练，对验证集数据、或者测试集 进行评估，评估其在所有batch上的平均准确率与平均损失
    :param sess:
    :param x_valid:  验证集的 x_valid (None, 2048)
    :param y_valid:  验证集的 y_valid (None, 3)

    :return: 所有batch的 softmax以后的概率分布 汇总 (8640, 3)
    """
    data_len = len(y_valid)  # batch_size = 1, data_len = num_batch
    total_loss = 0.0
    total_acc = 0.0
    valid_y_pred_results = []
    batch_valid = batch_iter(x_valid, y_valid,
                             1)  # batch_size =1 while valuation & testing  num_batch = data_len = len(y_valid)
    for x_batch, y_batch in batch_valid:
        feed_dict = feed_data(x_batch, y_batch, 1.0,
                              False)  # EVAL: keep_prob=1.0, valuation & testing are not during training.
        summery, valid_loss, valid_acc, valid_y_pred_batch = sess.run([merged, model.loss, model.acc, model.pred_label],
                                                                      feed_dict=feed_dict)
        total_loss += valid_loss
        total_acc += valid_acc
        valid_y_pred_results.append(valid_y_pred_batch.tolist()[0])  # tolist()[0] 多维矩阵(8640, 1, 3) 转化为 列表 (8640, 3)

    # 评估验证集在所有batch上的平均准确率与平均损失
    valid_avg_loss = float(total_loss / data_len)
    valid_avg_acc = float(total_acc / data_len)
    # print("valid_y_pred_batch:",valid_y_pred_batch)
    # softmax以后的概率分布 [[0.85549366 0.1202668  0.02423953]]
    # print("valid_y_pred_batch.shape:",valid_y_pred_batch.shape) # (1, 3)
    print(" valid_y_pred_results.shape:", np.array(valid_y_pred_results).shape)  # (8640, 3)

    return summery, valid_avg_loss, valid_avg_acc, valid_y_pred_results


'''配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖'''


def run_main(x_train_raw, y_train_raw, train_ratio):
    """
    主函数入口

    模型训练函数，训练一段时间后对模型进行验证，如果在一定时间后没有提升，则提前结束训练
    每次调用函数，都先对数据集打乱后划分 训练集和验证集，训练集占比为 train_ratio
    训练集与验证集划分比例 3:1， 综述， 训练集：验证集：测试集 = 6:2:2 = 3： 1: 1
    :param x_train_raw:
    :param y_train_raw:
    :param train_ratio:  0.75 训练集与验证集划分比例 3：1
    :return:
    """
    print("x_train_raw.shape", x_train_raw.shape)  # (34560, 2048)
    print("y_train_raw.shape", y_train_raw.shape)  # (34560, 3)
    split_train = int(len(y_train_raw) * train_ratio)
    x_train = x_train_raw[: split_train, :]  # 取数组 (行：0 ~ split_train，列：全部)
    x_valid = x_train_raw[split_train:, :]  # 取数组 (行：split_train ~ 最后一行，列：全部)
    y_train = y_train_raw[: split_train, :]
    y_valid = y_train_raw[split_train:, :]
    print("x_train.shape", x_train.shape)  # (25920, 2048)
    print("y_train.shape", y_train.shape)  # (25920, 3)
    print("x_valid.shape", x_valid.shape)  # (8640, 2048)
    print("y_valid.shape", y_valid.shape)  # (8640, 3)

    print('''=========================Configuring TensorBoard and Saver...=============================''')
    # =============================================================================================

    # 配置 Saver
    saver = tf.train.Saver()
    # 创建session
    # GPU Configuration
    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的。
    # 为防止收到操作出错， allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。

    gpu_config.gpu_options.allow_growth = True  # 让TensorFlow在运行过程中动态申请显存，需要多少就申请多少;
    # 全局初始化初始化
    init = tf.global_variables_initializer()
    with tf.Session(config=gpu_config) as sess:
        sess.run(init)
        valid_writer.add_graph(sess.graph)
        train_writer.add_graph(sess.graph)
        print('===================================Training and evaluating...================================')
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = Config.require_improvement  # 如果超过require_improvement轮未提升，提前结束训练

        train_loss_list = []  # loss value list while training
        valid_loss_list = []  # loss value list while validating
        train_acc_list = []  # acc value list while training
        valid_acc_list = []  # acc value list while validating

        train_iteration_loss_list = []
        valid_iteration_loss_list = []
        train_iteration_acc_list = []
        valid_iteration_acc_list = []
        y_pred_list = []

        flag = False
        for epoch in range(Config.num_epochs):  # num_epochs = 300
            # print("The length of x_train is :", len(x_train))             # 25920
            num_batch = int(len(x_train) / (Config.batch_size))  # num_batch = num_iterations
            # print("The number of batchs per epoch:", num_batch)           # 810
            # print('Epoch:', epoch + 1)
            # batch_size = 32
            batch_train = batch_iter(x_train, y_train, Config.batch_size)  # 获取 “ next_batch ”
            # loss_train_epoch = []
            # acc_train_epoch = []
            num_iteration = 0  # 记录每个epoch 总共迭代iteration次数 = batch个数(num_batch)
            avg_train_loss = 0
            avg_train_acc = 0
            ''' =================================== Training ================================================='''
            for x_batch, y_batch in batch_train:
                num_iteration = num_iteration + 1

                feed_dict = feed_data(x_batch, y_batch, Config.dropout_keep_prob_fc, True)  # Training
                # 函数 feed_data(x_batch, y_batch, dropout_keep_prob_fc, is_training):
                train_summery, _optim, loss_train, acc_train, y_pred_per_batch = sess.run([merged,
                                                                                           model.optim,
                                                                                           model.loss,
                                                                                           model.acc,
                                                                                           model.pred_label],
                                                                                          feed_dict=feed_dict)
                train_iteration_loss_list.append(float(loss_train))  # loss 拼接
                train_iteration_acc_list.append(float(acc_train))  # acc 拼接
                y_pred_list.append(y_pred_per_batch)

                train_writer.add_summary(train_summery, ((epoch * num_batch) + num_iteration))

                avg_train_loss = avg_train_loss + loss_train
                avg_train_acc = avg_train_acc + acc_train
                print("Epoch： %03d/%03d   batch:  %03d/%03d  train_loss: %.9f train_acc: %.9f" %
                      (epoch, Config.num_epochs, num_iteration, num_batch, loss_train, acc_train))

            print("The number of num_iteration is:", num_iteration)
            avg_train_loss = avg_train_loss / num_batch  # 训练集，每个epoch的平均损失，或者除以 num_iteration
            avg_train_acc = avg_train_acc / num_batch  # 训练集，每个epoch的平均准确率

            # 每1个epoch存入tensorboard
            # train_writer.add_summary(train_summery, epoch)        # 保存最后一个summery，写入本地

            train_loss_list.append(float(avg_train_loss))  # loss 拼接
            train_acc_list.append(float(avg_train_acc))  # acc 拼接

            if epoch % Config.display_epoch == 0:  # 每10个epoch在屏幕上输出训练的结果
                print(
                    " The average training loss and accracy:================================================================= ")
                print("Epoch： %03d/%03d train_loss: %.9f train_acc: %.9f" % (
                epoch, Config.num_epochs, avg_train_loss, avg_train_acc))

            ''' =================================== Valid  ================================================='''
            if epoch % Config.valid_epoch == 0:  # 每10个epoch，用验证集，进行评估，并屏幕打印一次结果

                valid_summery, avg_valid_loss, avg_valid_acc, y_pred_valid = evaluate(sess, merged, x_valid,
                                                                                      y_valid)  # 用验证集评估训练的效果
                valid_loss_list.append(float(avg_valid_loss))  # loss 拼接
                valid_acc_list.append(float(avg_valid_acc))  # acc 拼接
                # 每1个epoch存入tensorboard
                valid_writer.add_summary(valid_summery, epoch)  # 保存最后一个summery，写入本地

                print("Epoch： %03d/%03d valid_loss: %.9f valid_acc: %.9f" % (
                epoch, Config.num_epochs, avg_valid_loss, avg_valid_acc))
            if epoch % Config.save_epoch == 0:
                if avg_valid_acc > best_acc_val:
                    # 保存最好结果
                    best_acc_val = avg_valid_acc
                    last_improved = epoch
                    print('model save in path: ', save_path)
                    saver.save(sess=sess, save_path=save_path)
                    print('===================== Valid best Save success!================================')

            if epoch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("===================No optimization for many epochs, auto-stopping...=============")
                flag = True
                break  # 跳出循环

            if flag:  # 同上
                break

    # 实现了对字典对象中元素的排序
    train_and_valid_result = collections.OrderedDict()
    # train_result ={}
    train_and_valid_result['train_loss'] = train_loss_list
    print(train_and_valid_result['train_loss'])
    print('acc_value_train type and length is:', type(train_acc_list), '   ', len(train_acc_list))
    train_and_valid_result['train_acc'] = train_acc_list
    train_and_valid_result['NA'] = '---------------------------------------------'
    print('loss_value_valid type and length is:', type(valid_loss_list), '   ', len(valid_loss_list))
    train_and_valid_result['valid_loss'] = valid_loss_list
    print('acc_value_valid type and length is:', type(valid_acc_list), '   ', len(valid_acc_list))
    train_and_valid_result['valid_acc'] = valid_acc_list

    # 保存结果到 json 文件
    json_file = os.path.join(save_result_dir, 'train_and_valid_result.json')

    with open(json_file, 'w', encoding='utf-8') as json_file:
        json.dump(train_and_valid_result, json_file, indent=4)

    # 绘图并保存
    # plot_save(train_loss_list, fig_path=save_result_dir, xlabel='epoches', ylabel='train loss value')
    # plot_save(train_acc_list, fig_path=save_result_dir, xlabel='epoches', ylabel='train acc value')
    plot_save(valid_loss_list, fig_path=save_result_dir, xlabel='epoches', ylabel='valid loss value')
    plot_save(valid_acc_list, fig_path=save_result_dir, xlabel='epoches', ylabel='valid acc value')

    plot_save(train_iteration_loss_list, fig_path=save_result_dir, xlabel='iterations', ylabel='train loss value')
    plot_save(train_iteration_acc_list, fig_path=save_result_dir, xlabel='iterations', ylabel='train acc value')
    # plot_save(valid_iteration_loss_list, fig_path=save_result_dir, xlabel='iterations', ylabel='valid loss value')
    # plot_save(valid_iteration_acc_list, fig_path=save_result_dir, xlabel='iterations', ylabel='valid acc value')
    # # try:
    # except: pass


def test(x_test, y_test):
    loss_value_test = []  # loss value list while testing
    acc_value_test = []  # acc value list while testing
    # time_set_test = []   # average time while testing
    msg_list = []  # testing results
    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess_test:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        print('loading model from: ', save_path)
        saver.restore(sess=sess_test, save_path=save_path)  # 读取保存的模型
        print('loaded!')

        print('Testing...')
        test_summery, test_loss, test_acc, test_y_pred = evaluate(sess_test, merged, x_test, y_test)

        test_result = collections.OrderedDict()  # OrderedDict是dict的子类，它记住了内容添加的顺序。
        test_result['test_acc'] = test_acc
        test_result['test_loss'] = test_acc
        test_result['test_y_pred'] = test_y_pred
        test_result['y_test'] = y_test.tolist()

        test_result['learning_rate'] = Config.learning_rate
        test_result['dropout_keep_prob_fc'] = Config.dropout_keep_prob_fc

        test_path = 'test_result' + '.json'
        json_file = os.path.join(save_result_dir, test_path)
        with open(json_file, 'w', encoding='utf-8') as json_file:
            json.dump(test_result, json_file, indent=4)
        return test_acc, test_loss


class js_methods(object):
    # 嵌套类A.__dict,后可用json序列化
    def __init__(self, num, lr, dropout_fc, acc, loss):
        self.num = num
        self.lr = lr
        self.dropout_fc = dropout_fc
        self.acc = acc
        self.loss = loss

    def __repr__(self):
        return repr((self.num, self.lr, self.dropout_fc, self.acc, self.loss))


'''=====================================================================================================================
                                          ======= 主函数 =======
======================================================================================================================='''

# if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
if __name__ == '__main__':
    # json_config = os.path.join(save_result_dir, 'config.json')  # save_result = 'results'
    #
    # print("===============================Loading data...===============================================")
    # # 导入数据，取原始数据的 80% 作为训练集， 20% 作为测试集
    # dataset = loadmat('preprocessed_data.mat')
    # data = dataset['data']
    # label = dataset['label']
    # print("data.shape: ", data.shape)
    # print("label.shape ", label.shape)
    json_config = os.path.join(save_result_dir, 'config.json')     # save_result = 'results'

    print("===============================Loading data...===============================================")
    # 导入数据，取原始数据的 80% 作为训练集， 20% 作为测试集

    dataset_path = os.path.join(Config.PATH, 'fault_bearing_dataset.mat')
    dataset      = loadmat(dataset_path)
    normal_data  = dataset['normal_data']
    normal_label = dataset['normal_label']
    Inner_data   = dataset['Inner_Raceway_data']
    Inner_label  = dataset['Inner_Raceway_label']
    Outer_data   = dataset['Outer_Raceway_data']
    Outer_label  = dataset['Outer_Raceway_label']
    print(normal_data.shape,normal_label.shape,Inner_data.shape,Inner_label.shape,Outer_data.shape,Outer_label.shape)
    # (14400, 2048) (14400, 3) (14400, 2048) (14400, 3) (14400, 2048) (14400, 3)
    data  = np.concatenate((normal_data , Inner_data , Outer_data), axis=0)
    label = np.concatenate((normal_label , Inner_label , Outer_label), axis=0)
    print("data.shape", data.shape, 'label.shape', label.shape)

    num_sample = len(data)
    num_train = int(num_sample * Config.train_ratio)
    num_test = num_sample - num_train
    index_permutation = np.arange(num_sample)
    np.random.shuffle(index_permutation)  # 数据打乱
    train_data = data[index_permutation][: num_train]  # 训练集 随机抽取 80%
    test_data = data[index_permutation][num_train:]  # 测试集 随机抽取 20%
    # 需要分开预处理，以保证均值、方差等不公用，保证测试集的独立性
    #train_data = normalize(train_data)
    #test_data = normalize(test_data)
    train_label = label[index_permutation][: num_train]  # 分段操作的时候，取得到前面，取不到后面
    test_label = label[index_permutation][num_train:]  # 分段操作的时候，取得到前面，取不到后面
    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    # (34560, 2048) (34560, 3) (8640, 2048) (8640, 3)

    results = []
    # itertools.product(list1, list2) 依次取出list1中的每1个元素，与list2中的每1个元素，组成元组，然后将所有的元组组成一个列表，返回。
    # 在每周循环周期repeat内，排列组合循环不同的学习率和 保留的神经元概率（dropout_keep_prob_fc_list），保留50% 或者100%
    # (i,j,k)=(0,0.005,0.5),(0,0.005,1),(0,0.001,0.5),(0,0.005.1),(0,0.001,0.5),(0,0.001,1)...各种排列组合
    for (i, j, k, bat) in itertools.product(range(Config.repeat),  # repeat = 5   # 是epoch对上一级循环
                                            Config.learning_rate_list,
                                            Config.dropout_keep_prob_fc_list,
                                            Config.batch_size_list):

        print('Current working space is: ', os.getcwd())  # 返回当前工作目录。
        print('Configuring and Saving CNN model...')
        # save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
        # 检查路径是否存在，不存在则创建一个

        for path in [epoch_valid_tensorboard_dir, iter_train_tensorboard_dir, save_dir, save_result_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        Config.learning_rate = j  # 改变Config中默认的learning_rate
        Config.dropout_keep_prob_fc = k  # 改变Config中默认的神经元保留率
        Config.batch_size = bat

        config = Config()
        model = CNN(config)  # 调用 CNN 类 实例化，self.acc -> model.acc

        print("===============================CNN Model is Finished=======================")
        print("========================= run_main_function is training =======================")
        # ================================================================================================
        # 保存结果到本地
        '''需要注意的是，tf.summary.merge_all()要写在tf.summary.scalar()
        或是tf.summary.histogram()等方法之后，
        不然会报Fetch argument None has invalid type<class 'NoneType'>的错。
        '''
        merged = tf.summary.merge_all()  # 可以将所有summary全部保存到磁盘，以便tensorboard显示。
        train_writer = tf.summary.FileWriter(iter_train_tensorboard_dir)  # 定义一个写入summary的目标文件，dir为写入文件地址
        valid_writer = tf.summary.FileWriter(epoch_valid_tensorboard_dir)

        # 在训练集上训练， 在验证集上评估
        run_main(train_data, train_label, train_ratio=config.RATIO_train)  # 训练集中，训练集与验证集再划分为 4:1
        print("========================= Testing... =======================")
        # 在测试集上测试结果
        test_acc, test_loss = test(test_data, test_label)
        print("test_acc: ", test_acc, "test_loss: ", test_loss)

        results.append(js_methods(i, j, k, test_acc, test_loss))
        print("results.append is finished")
        shutil.copy('Config.py', save_result_dir)  # 把配置文件保存到本地目录
        shutil.copy('CNN_model.py', save_result_dir)  # 把CNN_Model模型保存到本地目录
        shutil.copy('run.py', save_result_dir)  # 把运行文件 run.py 保存到本地目录
        print("shutil.copy is finished")

        # #以下出错！！！    在c302 实验室服务器跑时不会出错  trail(i) = repeat = 5   # 俗称epoch
        # epoch_valid_tensorboard_dir = 'logdir\\epoch_valid'       # 每个epoch valid
        # iter_train_tensorboard_dir = 'logdir\\iter_train'   # 每个iteration train

        new_fold = 'test_acc_' + str(round(test_acc, 4)) + '_' + 'test_loss' + str(round(test_loss, 4)) \
                   + '_' + 'trail_' + str(i) + '_fc_dropout_' + str(k) + '_lr_' + str(j) + '_batch_size_' + str(bat)

        if not os.path.exists(new_fold):
            os.makedirs(new_fold)

        shutil.move(tensorboard_dir, new_fold)
        # shutil.move(iter_train_tensorboard_dir, new_fold)
        # shutil.move(epoch_valid_tensorboard_dir, new_fold)
        shutil.move(save_dir, new_fold)
        shutil.move(save_result_dir, new_fold)
        # 把 save_result_dir 文件夹里的文件都 被移动到 命名为 "trail 0 fc_drop_out 0.5 lr 0.005" 的文件里

        print(" shutil.move is finished")

    with open('results.json', 'w') as f:
        f.write(json.dumps(results, default=lambda o: o.__dict__, indent=4))

print(" ***************************************All Finished*********************************************")

"""
【关于训练集、验证集、测试集的使用】
1 训练集： 拟合数据样本， 进行学习，更新参数
2 验证集： 用于调试超参数、使用多次，每几个epoch跑一次，！！！必须有！！！
          模型训练过程单独留下的样本集， 用于调整超参数和用于对模型进行初步评估
          验证集可以哟用在训练的过程中， 一般在训练时，几个epoch结束后跑一次验证集看看效果。
          （但是验证的太频繁会影响训练速度）
          优点：
          1) 可以及时发现模型或者参数问题，比如验证集发散、出现奇怪的值（无穷大）、准确率不增长或者很慢，
             此时可以及时终止训练，重新调参或者调整模型，而不需要等到训练结束。就是可以实时监控！
          2）还有就是验证模型的泛化能力， 如果验证集上的效果比训练集上差很多，就可以考虑模型是否过拟合
             一旦 validation_data 的分类精度达到饱和，就停止训练。这种策略叫做提前终止（early stopping）
          3) 可以通过验证集对比不同的模型。在一般的神经网络中，我们用验证集去寻找最优的网络深度（number of hidden layers）
             或者决定反向传播的停止点，或者在神经网络中选择隐藏神经元的个数
          4) 交叉验证（Cross Validation) 就是把训练数据集本身再细分成不同的验证数据集
          缺点：
          1) 模型在一次次手动调参并举行训练后逼近的验证集， 可能只代表一部分非训练集，导致最终的模型泛化还不够好
3 测试集:  ！！！可以没有，可以用验证集来代替！！！
          所有训练、验证、模型调整完毕以后，用整个测试集跑一次，看模型的泛化能力
          不能作为调参、选择特征等算法相关的选择的依据。
4 验证集和测试集相互关系：
          验证集具有足够泛化性（一般来说，如果验证集足够大到包括大部分非训练集时，也等于具有足够泛化性了）
          验证集具有足够泛化性时，测试集就没有存在的必要了
          如果验证集具有足够泛化代表性，测试集是可以没有的，但验证集是必须有的。

PS:       1) test_data是模型出炉的最后一道检测工序，
            test_data 来防止过拟合。如果基于 test_data 的评估结果设置超参数，有可能我们的网络最后是对 test_data 过拟合。
            也就是说，我们或许只是找到了适合 test_data 具体特征的超参数，网络的性能不能推广到其它的数据集。
          2) 普通参数可以通过网络来更新，自动调参（训练集训练），超参数是人工手动"更新"，手动调参（验证集也类似在训练），
             所以测试集有存在的必要！
"""




















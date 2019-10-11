import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cmap = plt.cm.get_cmap('cool')

    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    # print(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].astype('float')
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # ±ê¼ÇÔÚÍ¼Æ¬ÖÐ¶ÔÎÄ×ÖÊÇÕûÊýÐÍ»¹ÊÇ¸¡µãÐÍ
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%.2f" % (c,), color='black', fontsize=14, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            # print(c)
            if c>0.60:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=14, va='center', ha='center')
            elif (c >= 0.01):
                #ÕâÀïÊÇ»æÖÆÊý×Ö£¬¿ÉÒÔ¶ÔÊý×Ö´óÐ¡ºÍÑÕÉ«½øÐÐÐÞ¸Ä
                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=14, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%.2f" % (0,), color='black', fontsize=14, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='gaussian', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontsize=14)
    plt.yticks(xlocations, labels, fontsize=14)
    plt.ylabel('Actual label', labelpad=10,fontsize=18 )
    plt.xlabel('Predict label',labelpad=10, fontsize=18)
    plt.savefig('confusion_matrix.png', dpi=200)
    # plt.show()
    plt.close()

def plot_loss(array, xlabel, ylabel, savefilename):
    plt.figure()
    plt.plot(array)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(savefilename+'.png')
    plt.close()


f = open('train_and_valid_result.json')
train_valid_result = json.load(f)
print(train_valid_result)
# train_loss = train_valid_result['train_loss']
# plot_loss(train_loss, 'epochs', 'train loss value', 'train_loss_value')
#
# train_acc = train_valid_result['train_acc']
# plot_loss(train_acc, 'epochs', 'train acc value', 'train_acc_value')
#
# valid_loss = train_valid_result['valid_loss']
# plot_loss(valid_loss, 'epochs', 'valid_loss value', 'valid_loss_value')
#
# valid_acc = train_valid_result['valid_acc']
# plot_loss(valid_acc, 'epochs', 'valid_acc value', 'valid_acc_value')


c = open('test_result.json')
test_y_pred_result = json.load(c)
test_y_pred = test_y_pred_result['test_y_pred']
test_y_pred = np.array(test_y_pred)
print(test_y_pred.shape)
y_pred = np.argmax(test_y_pred, axis=1)
print(y_pred.shape)

test_y_true = test_y_pred_result['y_test']
test_y_true = np.array(test_y_true)
y_true = np.argmax(test_y_true, axis=1)
print(y_true.shape)
plot_confusion_matrix(y_true, y_pred, [0, 1, 2])
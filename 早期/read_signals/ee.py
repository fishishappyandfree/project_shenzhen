from math import sqrt
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig



def cal_pearson(a,b):
    # n=len(x)
    # #求x_list、y_list元素之和
    # sum_x=sum(x)
    # sum_y=sum(y)
    # #求x_list、y_list元素乘积之和
    # sum_xy=multiply(x,y)
    # #求x_list、y_list的平方和
    # sum_x2 = sum([pow(i,2) for i in x])
    # sum_y2 = sum([pow(j,2) for j in y])
    # molecular=sum_xy-(float(sum_x)*float(sum_y)/n)
    # #计算Pearson相关系数，molecular为分子，denominator为分母
    # denominator=sqrt((sum_x2-float(sum_x**2)/n)*(sum_y2-float(sum_y**2)/n))

    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = np.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq

    return abs(corr_factor)




fx_path1="D:/out/fx600.mat"

fx_path11="D:/out/fx1000.mat"
fx_path12="D:/out/fx1500.mat"

fx_path2="D:/out/fx2000.mat"
fx_path3="D:/out/fx5000.mat"
fx_path4="D:/out/fx10000.mat"
fx_path5="D:/out/fx20000.mat"

fx_path6="D:/in/fx600.mat"

fx_path61="D:/in/fx1000.mat"
fx_path62="D:/in/fx1500.mat"


fx_path7="D:/in/fx2000.mat"

fx_path81="D:/in/fx3000.mat"
fx_path82="D:/in/fx4000.mat"
fx_path8="D:/in/fx5000.mat"
fx_path83="D:/in/fx6000.mat"
fx_path84="D:/in/fx7000.mat"
fx_path85="D:/in/fx8000.mat"


fx_path9="D:/in/fx10000.mat"
fx_path10="D:/in/fx20000.mat"



fx_path600="D:/in2/fx600.mat"

fx_path1000="D:/in2/fx1000.mat"
fx_path1500="D:/in2/fx1500.mat"


fx_path2000="D:/in2/fx2000.mat"
fx_path2500="D:/in2/fx2500.mat"
fx_path3000="D:/in2/fx3000.mat"
fx_path4000="D:/in2/fx4000.mat"
fx_path5000="D:/in2/fx5000.mat"
fx_path6000="D:/in2/fx6000.mat"
fx_path7000="D:/in2/fx7000.mat"
fx_path8000="D:/in2/fx8000.mat"
fx_path9000="D:/in2/fx9000.mat"

fx_path10000="D:/in2/fx10000.mat"

fx_path14400="D:/in2/fx14400.mat"

fx_path20000="D:/in2/fx20000.mat"

def read_data(path, label):

    data = scio.loadmat(path)

    data_list=data[label]#取出字典里的data

    data_array = np.array(data_list)

    return data_array


def getmean2(str, p):

    # p = abs(p)
    ave = np.mean(p)
    max = np.max(p)
    min = np.min(p)
    var = np.var(p)
    print(str + " : " + "ave = %.12f max = %.12f min = %.12f var = %.12f" % (ave,max,min,var))


def getmean(str, p):
    # p = lowpass(p,3, 2*50/25600, "highpass")
    # p = p.flatten()
    p = abs(np.fft.fft(p))[:int(len(p)/2)]
    # p = np.fft.fft(p)[:int(len(p) / 2)]
    # p[0]=0
    # plt.plot(p)
    # plt.show()
    ave = np.mean(p)
    max = np.max(p)
    min = np.min(p)
    var = np.var(p)
    # crol(str, abs(np.fft.fft(fx_data5[5000:10000]))[:int(len(p)/2)], p)
    print(str + " : " + "ave = %.12f max = %.12f min = %.12f var = %.12f" % (ave,max,min,var))



def come(x):
    x = np.array(x)
    i = np.mean(x)
    x = x-i
    j = np.std(x)
    x = x/j
    print(i,j)
    return x
# f1 = fx_data1.flatten()





def pich(data):
    ret = -1

    # normalization_data = come(data)
    # normalization_data = data
    var_time = np.var(data)

    fft_data = abs(np.fft.fft(data))

    var_fre = np.var(fft_data)

    print("var_time : %.12f   var_fre : %.12f" % (var_time, var_fre))
    # 适用于外圈轴承故障数据过滤
    # if var_time>0.01:
    #     if var_fre>20:
    #         ret = 0

    # 适用于内圈轴承故障数据过滤
    if var_time>0.004:
        ret = 0
    elif var_fre>7.5:
        ret = 0
    return ret



if __name__=='__main__':
    # print ("x_list,y_list的Pearson相关系数为："+str(cal_pearson(f,y)))

    # fx_data1 = np.array(read_data(fx_path1, "feed_x")).flatten()
    #
    # fx_data11 = np.array(read_data(fx_path11, "feed_x")).flatten()
    # fx_data12 = np.array(read_data(fx_path12, "feed_x")).flatten()
    #
    # fx_data2 = np.array(read_data(fx_path2, "feed_x")).flatten()
    # fx_data3 = np.array(read_data(fx_path3, "feed_x")).flatten()
    # fx_data4 = np.array(read_data(fx_path4, "feed_x")).flatten()
    # fx_data5 = np.array(read_data(fx_path5, "feed_x")).flatten()
    #
    # fx_data6 = np.array(read_data(fx_path6, "feed_x")).flatten()
    #
    # fx_data61 = np.array(read_data(fx_path61, "feed_x")).flatten()
    # fx_data62 = np.array(read_data(fx_path62, "feed_x")).flatten()
    #
    # fx_data7 = np.array(read_data(fx_path7, "feed_x")).flatten()
    #
    # fx_data81 = np.array(read_data(fx_path81, "feed_x")).flatten()
    # fx_data82 = np.array(read_data(fx_path82, "feed_x")).flatten()
    # fx_data8 = np.array(read_data(fx_path8, "feed_x")).flatten()
    # fx_data83 = np.array(read_data(fx_path83, "feed_x")).flatten()
    # fx_data84 = np.array(read_data(fx_path84, "feed_x")).flatten()
    # fx_data85 = np.array(read_data(fx_path85, "feed_x")).flatten()
    #
    # fx_data9 = np.array(read_data(fx_path9, "feed_x")).flatten()
    # fx_data10 = np.array(read_data(fx_path10, "feed_x")).flatten()

    fx_data600 = np.array(read_data(fx_path600, "feed_x")).flatten()
    fx_data1000 = np.array(read_data(fx_path1000, "feed_x")).flatten()
    fx_data1500 = np.array(read_data(fx_path1500, "feed_x")).flatten()
    fx_data2000 = np.array(read_data(fx_path2000, "feed_x")).flatten()
    fx_data2500 = np.array(read_data(fx_path2500, "feed_x")).flatten()
    fx_data3000 = np.array(read_data(fx_path3000, "feed_x")).flatten()
    fx_data4000 = np.array(read_data(fx_path4000, "feed_x")).flatten()
    fx_data5000 = np.array(read_data(fx_path5000, "feed_x")).flatten()
    fx_data6000 = np.array(read_data(fx_path6000, "feed_x")).flatten()
    fx_data7000 = np.array(read_data(fx_path7000, "feed_x")).flatten()
    fx_data8000 = np.array(read_data(fx_path8000, "feed_x")).flatten()
    fx_data9000 = np.array(read_data(fx_path9000, "feed_x")).flatten()
    fx_data10000 = np.array(read_data(fx_path10000, "feed_x")).flatten()

    fx_data14400 = np.array(read_data(fx_path14400, "feed_x")).flatten()

    fx_data20000 = np.array(read_data(fx_path20000, "feed_x")).flatten()

    # fx_data1 = come(fx_data1)
    #
    # fx_data11 = come(fx_data11)
    # fx_data12 = come(fx_data12)
    #
    # fx_data2 = come(fx_data2)
    # fx_data3 = come(fx_data3)
    # fx_data4 = come(fx_data4)
    # fx_data5 = come(fx_data5)
    # fx_data6 = come(fx_data6)
    #
    # fx_data61 = come(fx_data61)
    # fx_data62 = come(fx_data62)
    #
    # fx_data7 = come(fx_data7)
    #
    # fx_data81 = come(fx_data81)
    # fx_data82 = come(fx_data82)
    # fx_data8 = come(fx_data8)
    # fx_data83 = come(fx_data83)
    # fx_data84 = come(fx_data84)
    # fx_data85 = come(fx_data85)
    #
    # fx_data9 = come(fx_data9)
    # fx_data10 = come(fx_data10)

    # getmean("out_fx600",fx_data1[15000:17048])
    #
    # getmean("out_fx1000", fx_data11[15000:17048])
    # getmean("out_fx1500", fx_data12[15000:17048])
    #
    # getmean("out_fx2000",fx_data2[15000:17048])
    # getmean("out_fx5000",fx_data3[15000:17048])
    # getmean("out_fx10000",fx_data4[15000:17048])
    # getmean("out_fx20000",fx_data5[15000:17048])
    # print("*"*30)
    # getmean("in_fx600",fx_data6[50000:52048])
    #
    # getmean("in_fx1000", fx_data61[10000:12048])
    # getmean("in_fx1500", fx_data62[15000:17048])
    #
    # getmean("in_fx2000",fx_data7[15000:17048])
    #
    # # getmean("in_fx3000",fx_data81[15000:17048])
    # # getmean("in_fx4000",fx_data82[180000:182048])
    # getmean("in_fx5000",fx_data8[20000:22048])
    # # getmean("in_fx6000",fx_data83[15000:17048])
    # # getmean("in_fx7000",fx_data84[15000:17048])
    # # getmean("in_fx8000",fx_data85[15000:17048])
    #
    # getmean("in_fx10000",fx_data9[15000:17048])
    # getmean("in_fx20000",fx_data10[15000:17048])



    # getmean("out_fx600",fx_data1[500000:502048])
    # getmean("out_fx1000", fx_data11[300000:302048])
    # getmean("out_fx2000", fx_data12[300000:302048])
    # getmean("out_fx2000",fx_data2[130000:132048])
    # getmean("out_fx5000",fx_data3[130000:132048])
    # getmean("out_fx10000",fx_data4[130000:132048])
    # getmean("out_fx20000",fx_data5[130000:132048])
    # print("*"*30)
    # getmean("in_fx600",fx_data6[10000:12048])
    #
    # getmean("in_fx1000", fx_data61[150000:152048])
    # getmean("in_fx1500", fx_data62[150000:152048])
    #
    # getmean("in_fx2000",fx_data7[80000:82048])
    # getmean("in_fx5000",fx_data8[250000:252048])
    # getmean("in_fx10000",fx_data9[130000:132048])
    # getmean("in_fx20000",fx_data10[130000:132048])


    # getmean("in_fx3000",fx_data81[100000:102048])
    # getmean("in_fx4000",fx_data82[50000:52048])
    # getmean("in_fx5000",fx_data8[100000:102048])
    # getmean("in_fx6000",fx_data83[100000:102048])
    # getmean("in_fx7000",fx_data84[200000:202048])
    # getmean("in_fx8000",fx_data85[100000:102048])

    # print(pich(fx_data600[6000:8048]))
    # print(pich(fx_data600[36000:38048]))

    # print(pich(fx_data1000[6000:8048]))
    # print(pich(fx_data1000[66000:68048]))

    # print(pich(fx_data1500[6000:8048]))
    # print(pich(fx_data1500[96000:98048]))

    # print(pich(fx_data2500[6000:8048]))
    # print(pich(fx_data2500[166000:168048]))

    print(pich(fx_data2000[6000:8048]))
    print(pich(fx_data2000[66000:68048]))

    datamean=0

    plt.subplot(7,1,1)
    plt.plot(fx_data600-datamean)
    plt.ylabel("out_fx600")

    plt.subplot(7,1,2)
    plt.plot(fx_data1000-datamean)
    plt.ylabel("out_fx1000")

    plt.subplot(7,1,3)
    plt.plot(fx_data1500-datamean)
    plt.ylabel("out_fx1500")

    plt.subplot(7,1,4)
    plt.plot(fx_data2000-datamean)
    plt.ylabel("out_fx2000")

    plt.subplot(7,1,5)
    plt.plot(fx_data2500-datamean)
    plt.ylabel("out_fx2500")

    plt.subplot(7, 1, 6)
    plt.plot(fx_data3000 - datamean)
    plt.ylabel("out_fx3000")

    plt.subplot(7, 1, 7)
    plt.plot(fx_data4000 - datamean)
    plt.ylabel("out_fx4000")

    # plt.subplot(7, 1, 4)
    # plt.plot(fx_data5000 - datamean)
    # plt.ylabel("out_fx5000")
    #
    # plt.subplot(7, 1, 5)
    # plt.plot(fx_data6000 - datamean)
    # plt.ylabel("out_fx6000")
    #
    # plt.subplot(7, 1, 2)
    # plt.plot(fx_data7000 - datamean)
    # plt.ylabel("out_fx7000")
    #
    # plt.subplot(7, 1, 3)
    # plt.plot(fx_data8000 - datamean)
    # plt.ylabel("out_fx8000")
    #
    # plt.subplot(7, 1, 4)
    # plt.plot(fx_data9000 - datamean)
    # plt.ylabel("out_fx9000")
    #
    # plt.subplot(7,1,6)
    # plt.plot(fx_data10000-datamean)
    # plt.ylabel("out_fx10000")
    #
    # plt.subplot(7,1,6)
    # plt.plot(fx_data14400-datamean)
    # plt.ylabel("out_fx14400")
    #
    # plt.subplot(7, 1, 7)
    # plt.plot(fx_data20000-datamean)
    # plt.ylabel("out_fx20000")

    # plt.subplot(7, 1, 1)
    # plt.plot(fx_data6-datamean)
    # plt.ylabel("in_fx600")
    #
    # plt.subplot(7, 1, 2)
    # plt.plot(fx_data61-datamean)
    # plt.ylabel("in_fx1000")
    #
    # plt.subplot(7, 1, 3)
    # plt.plot(fx_data62-datamean)
    # plt.ylabel("in_fx1500")
    #
    # plt.subplot(7, 1, 4)
    # plt.plot(fx_data7-datamean)
    # plt.ylabel("in_fx2000")
    #
    # plt.subplot(7, 1, 5)
    # plt.plot(fx_data8-datamean)
    # plt.ylabel("in_fx5000")
    #
    # plt.subplot(7, 1, 6)
    # plt.plot(fx_data9-datamean)
    # plt.ylabel("in_fx10000")
    #
    # plt.subplot(7, 1, 7)
    # plt.plot(fx_data10-datamean)
    # plt.ylabel("in_fx20000")

    # plt.subplot(6, 1, 1)
    # plt.plot(fx_data81-datamean)
    # plt.ylabel("in_fx3000")
    #
    # plt.subplot(6, 1, 2)
    # plt.plot(fx_data82-datamean)
    # plt.ylabel("in_fx4000")
    #
    # plt.subplot(6, 1, 3)
    # plt.plot(fx_data8-datamean)
    # plt.ylabel("in_fx5000")
    #
    # plt.subplot(6, 1, 4)
    # plt.plot(fx_data83-datamean)
    # plt.ylabel("in_fx6000")
    #
    # plt.subplot(6, 1, 5)
    # plt.plot(fx_data84-datamean)
    # plt.ylabel("in_fx7000")
    #
    # plt.subplot(6, 1, 6)
    # plt.plot(fx_data85-datamean)
    # plt.ylabel("in_fx80000")


    plt.show()







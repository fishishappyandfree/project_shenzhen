import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile


fx_path="G:/js/code/data_mat_analysis/nomal/x_0.6_0.02_num4_feed3000-19-01-34_fx3000.mat"
#fy_path="D:/fy.mat"
#microphone_path="D:/microphone.mat"


#Method 1
def read_data(path, label):

    data = scio.loadmat(path)

    data_list=data[label]#取出字典里的data

    data_array = np.array(data_list)

    return data_array


key = ["/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",   # feed_x_axis channel number
         "/'未命名'/'cDAQ9189-1D71297Mod4/ai1'"]   # feed_y_axis channel number
file = TdmsFile('G:/js/results/x_new_num3_feed3000-14-53-24.tdms')
fx_data = file.objects[key[0]].data  #取出数据
print(type(fx_data))
#fx_data = read_data(fx_path, "feed_x")
#fy_data = read_data(fy_path, "feed_y")
#microphone_data = read_data(microphone_path, "microphone")

#fx_data = fx_data.flatten()
#fx_data1 = f[40000:60000]
#fx_data2 = f[100000:120000]
# fy_data = fy_data.flatten()[100000:120000]
# microphone_data = microphone_data.flatten()[100000:120000]


# fx_data = fx_data[20480*0:20480*1]
# fy_data = fy_data[20480*3:20480*4]
# microphone_data = microphone_data[20480*3:20480*4]

print(fx_data.shape)
print(fx_data)
#
# print('*'*50)
# print(fy_data.shape)
# print(fy_data)
#
# print('*'*50)
# print(microphone_data.shape)
# print(microphone_data)


# fx_data = fx_data[10000:15000]
# plt.plot(fx_data)
# plt.show()

L1 = len(fx_data)  # size of feed_x
# L1 = len(fx_data)  # size of feed_x
# L11 = len(fx_data2)

#L2 = len(fy_data)  # size of feed_y
#L3 = len(microphone_data)  # size of microphone

# print(L1)


plt.subplot(211)
plt.plot(fx_data)
# plt.xticks(range(L1+1024)[::1024],[i*1024 for i in range(21)])
plt.title('feed_x_time')

# plt.subplot(323)
# plt.plot(fx_data2)

# PL1 = np.abs(np.fft.fft(fx_data1/L1))[: int(L1 / 2)]


PL = np.abs(np.fft.fft(fx_data/L1))[: int(L1 / 2)]

PL[0] = 0
# PL2[0] = 0
f1 = np.fft.fftfreq(L1,1/25600)[: int(L1 / 2)]

plt.subplot(212)
plt.plot(f1, PL)
#
# plt.subplot(324)
# plt.plot(f1, PL2)
#
# plt.subplot(325)
# plt.plot(f1, PL2-PL1)
# plt.title('feed_x_fft')


# plt.subplot(323)
# plt.plot(fy_data)
#
# PL2 = abs(np.fft.fft(fy_data/L2))[: int(L2 / 2)]
# PL2[0] = 0
# f2 = np.fft.fftfreq(L2, 1/25600)[: int(L2 / 2)]
# plt.subplot(324)
# plt.plot(f2, PL2)
# plt.title('feed_y_fft')
#
#
# plt.subplot(325)
# plt.plot(microphone_data)
#
# PL3 = abs(np.fft.fft(microphone_data/L3))[: int(L3 / 2)]
# PL3[0] = 0
# f3 = np.fft.fftfreq(L3, 1/25600)[: int(L3 / 2)]
# plt.subplot(326)
# plt.plot(f3, PL3)
# plt.title('microphone_fft')


plt.show()





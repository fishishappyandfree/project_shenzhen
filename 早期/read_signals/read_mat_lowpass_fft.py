import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


fx_path="D:/fx.mat"
fy_path="D:/fy.mat"
microphone_path="D:/microphone.mat"


#Method 1
def read_data(path, label):

    data = scio.loadmat(path)

    data_list=data[label]#取出字典里的data

    data_array = np.array(data_list)

    return data_array


fx_data = read_data(fx_path, "feed_x")
fy_data = read_data(fy_path, "feed_y")
microphone_data = read_data(microphone_path, "microphone")


fx_data = fx_data.flatten()
fy_data = fy_data.flatten()
microphone_data = microphone_data.flatten()


plt.subplot(211)
plt.plot(fx_data)
plt.title('feed_x_normal')

# fx_data = fx_data
# fy_data = fy_data
# microphone_data = microphone_data


# lowpass
def lowpass(signal, n, w, type):
    b, a = sig.butter(n, w, type)
    filtedData = sig.filtfilt(b, a, signal) #data为要过滤的信号
    return filtedData


fx_data = lowpass(fx_data, 3, 2*50/25600, "lowpass")
fy_data = lowpass(fy_data, 3, 2*50/25600, "lowpass")
microphone_data = lowpass(microphone_data, 3, 2*50/25600, "lowpass")

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
L2 = len(fy_data)  # size of feed_y
L3 = len(microphone_data)  # size of microphone

# print(L1)


plt.subplot(212)
plt.plot(fx_data)
plt.title('feed_x_lowpass')

# PL1 = np.abs(np.fft.fft(fx_data/L1))[: int(L1 / 2)]
# PL1[0] = 0
# f1 = np.fft.fftfreq(L1,1/25600)[: int(L1 / 2)]
#
# plt.subplot(212)
# plt.plot(f1, PL1)
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





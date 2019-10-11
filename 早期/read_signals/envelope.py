from scipy import fftpack
import numpy as np
from matplotlib import pyplot as plt
# import scipy.signal as signal
#
# # original singal
# sampling_rate = 51200
# fft_size = 51200
# t = np.arange(0,1.0,1.0/sampling_rate)
# ts = np.array(map(lambda x : x*1000, t))
# x = np.sin(2*np.pi*1e3*t) + 0.1 * np.sin(2*np.pi*980*t) + 0.10 * np.sin(2*np.pi*1020*t)+ 0.01 * np.sin(2*np.pi*960*t) + 0.01 * np.sin(2*np.pi*1040*t)
# xn = x + 0.005*np.random.normal(0.0,1.0,len(x))
# # envelop detecting
# hx = fftpack.hilbert(x)
# hy = np.sqrt(x**2+hx**2)
#
#
# # # parameters of filter
# # a = np.array([1.0, -1.947463016918843, 0.9555873701383931])
# # b = np.array([0.9833716591860479, -1.947463016918843, 0.9722157109523452])
# # # chirp signal
# # t = np.arange(0, 0.5, 1/44100.0)
# # x= signal.chirp(t, f0=10, t1 = 0.5, f1=1000.0)
# # # the chirp signal through the filter
# # y = signal.lfilter(b, a, x)
# # # hilbert transform
# # hy = fftpack.hilbert(y)
#
# L = len(x)
# print(L)
#
# PL = abs(np.fft.fft(hy))
#
#
# # PL[0] = 0
# f = np.fft.fftfreq(L, 1)
#
# plt.plot(f, PL)
# plt.title('envelope')
#
# plt.show()





t = np.arange(0, 0.3, 1/20000.0)
x = np.sin(2*np.pi*1000*t) * (np.sin(2*np.pi*20*t) + np.sin(2*np.pi*8*t) + 3.0)
hx = fftpack.hilbert(x)
# plt.subplot(221)
plt.plot(x, label=u"Carrier")
s = np.sqrt(x**2 + hx**2)

plt.plot(s, "r", linewidth=2, label=u"Envelop")
# plt.plot(abs(np.fft.fft(s)))
plt.title(u"Hilbert Transform")
plt.legend()
plt.show()










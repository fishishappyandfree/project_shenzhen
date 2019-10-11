from pandas import Series,DataFrame
# import pandas as pd
import h5py
import pandas as pd
import numpy as np


fx_path="C:/Users/wzs/Desktop/fx.mat"
fy_path="C:/Users/wzs/Desktop/fy.mat"
microphone_path="C:/Users/wzs/Desktop/microphone.mat"


def read_data(path, label):

    file = h5py.File(path,'r')

    data = file[label][:]

    dfdata = pd.DataFrame(data)

    data_S = pd.Series(dfdata)

    data_array = np.array(data_S)

    return data_array


fx_data = read_data(fx_path, "feed_x")
fy_data = read_data(fy_path, "feed_y")
microphone_data = read_data(microphone_path, "microphone")
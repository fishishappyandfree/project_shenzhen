import os
from nptdms import TdmsFile
from scipy.io import savemat

savepath = '/home/hust/Desktop/wzs/MT2_X_feed/MT2_X_and_y_feed_data/'
# get file path
fault_directory = '/home/hust/Desktop/JiangSu/data_from_MT2/Spindle_Program/Project/MT2/'


folder_normal = 'normal_slightG/'

#folder_inner = 'inner_0.6_0.04/'

folder_outer = 'outer-0.6-0.04_criticalG/'

folder = []
folder.append(folder_normal)
#folder.append(folder_inner)
folder.append(folder_outer)



for file_name_save_read in folder:
    fault_directory_type = fault_directory + file_name_save_read

    tdms_file_directory = fault_directory_type + 'raw_data/'
    filename_list = []

    for file in os.listdir(tdms_file_directory):
        if file.endswith('.tdms'):
            filename_list.append(file)

    for filename_index in range(len(filename_list)):
        tdms_file_path = tdms_file_directory + filename_list[filename_index]
        tdms_content = TdmsFile(tdms_file_path)

        # get group names and cahnnel names for later fetching
        # tdms_groups = tdms_content.groups()
        # print(tdms_groups)
        # for group_name in tdms_groups:
        #     channel_name_lists = tdms_content.group_channels(group_name)
        #     print(channel_name_lists)
        #     exit()

        # ----------------------------------- MT1 channels --------------------------------------------------------
        # spindle_x = tdms_content.object('δ����', 'cDAQ9189-1D71297Mod1/ai0').data  #(group_name, channel_name)
        # spindle_y = tdms_content.object('δ����', 'cDAQ9189-1D71297Mod1/ai1').data
        # spindle_z = tdms_content.object('δ����', 'cDAQ9189-1D71297Mod1/ai2').data

        # ----------------------------------- MT2 channels --------------------------------------------------------
        # spindle_x = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai0').data
        # spindle_y = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai1').data
        # spindle_z = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai2').data
        # MT2_feed_x = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai3').data
        # MT2_feed_y = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod4/ai0').data
        MT2_mic = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod6/ai1').data

        # ----------------------------------- MT3 channels --------------------------------------------------------
        # spindle_x = tdms_content.object('δ����', 'cDAQ9189-1D71297Mod5/ai0').data  # (group_name, channel_name)
        # spindle_y = tdms_content.object('δ����', 'cDAQ9189-1D71297Mod5/ai1').data
        # spindle_z = tdms_content.object('δ����', 'cDAQ9189-1D71297Mod5/ai2').data

        savemat('MT2_micphone_data/' + file_name_save_read + filename_list[filename_index].rstrip('.tdms')+'.mat',
                dict(MT2_mic=MT2_mic))

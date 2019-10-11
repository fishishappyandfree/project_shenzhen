import os
from nptdms import TdmsFile
from scipy.io import savemat

# get file path
#fault_directory = '/home/hust/Desktop/JiangSu/MT3/data_from_machine_tool_3/bearing_x_axis_inner_race_width_0.6mm_depth_0.02mm_num_3_OK/data_not_cutting/'
#fault_directory = "/home/hust/Desktop/wzs/ae_data/MT3_ae_test/with_ouheji/"
#fault_directory = "/home/hust/Desktop/wzs/ae_data/MT3_ae_test/without_ouheji/"
# fault_directory = "/home/hust/Desktop/wzs/Model_for_GT_8_data_and_MT3_8_data/raw_data_MT3/small/"
fault_directory = "/home/hust/Desktop/wzs/Model_for_GT_8_data_and_MT3_8_data/raw_data_MT3/large/"

tdms_file_directory = fault_directory
filename_list = []

for file in os.listdir(tdms_file_directory):
    if file.endswith('.tdms'):
        filename_list.append(file)

for filename_index in range(len(filename_list)):
    print(filename_list[filename_index])
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
    # spindle_x = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod1/ai0').data  #(group_name, channel_name)
    # spindle_y = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod1/ai1').data
    # spindle_z = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod1/ai2').data

    # ----------------------------------- MT2 channels --------------------------------------------------------
    # spindle_x = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai0').data
    # spindle_y = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai1').data
    # spindle_z = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai2').data
    # feed_x = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod3/ai3').data
    # feed_y = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod4/ai0').data
    # mic = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod6/ai1').data

    # ----------------------------------- MT3 channels --------------------------------------------------------
    # spindle_x = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod5/ai0').data  # (group_name, channel_name)
    # spindle_y = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod5/ai1').data
    # spindle_z = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod5/ai2').data
    #micphone = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod6/ai2').data  # (group_name, channel_name)

    micphone_data = tdms_content.object('未命名', 'cDAQ9189-1D71297Mod6/ai2').data

    # savemat(fault_directory + 'mat_data/' + filename_list[filename_index].rstrip('.tdms'),
    #         # dict(spindle_x=spindle_x, spindle_y=spindle_y, spindle_z=spindle_z))
    #         dict(micphone = micphone))

    savemat('micphone_mat/' + filename_list[filename_index].rstrip('.tdms')+'.mat',
    #         # dict(spindle_x=spindle_x, spindle_y=spindle_y, spindle_z=spindle_z))
            dict(micphone_data=micphone_data))
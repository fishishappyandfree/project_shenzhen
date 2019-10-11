% # Norm data ÎÄ¼şÂ·¾¶
% tdms_path_norm_3000_0 = r"G:\js\results\new_feed_x_3000-19-13-03.tdms"
% tdms_path_norm_3000_1 = r"G:\js\results\x_new_num3_feed3000-14-53-24.tdms"
% tdms_path_norm_5000_0 = r"G:\js\results\new_feed_x_5000-19-18-24.tdms"
% tdms_path_norm_5000_1 = r"G:\js\results\x_new_num3_feed5000-14-54-42.tdms"
% 
% # inner 0.6-0.02
% tdms_path_in_600_3000_0 = r"G:\js\results\x_0.6_0.02_num4_feed3000-19-01-34.tdms"
% tdms_path_in_600_3000_1 = r"G:\js\results\x_0.6_0.02_num5_feed3000-15-09-18.tdms"
% tdms_path_in_600_5000_0 = r"G:\js\results\x_0.6_0.02_num4_feed5000-19-02-43.tdms"
% tdms_path_in_600_5000_1 = r"G:\js\results\x_0.6_0.02_num5_feed5000-15-08-38.tdms"
% 
% # outer 0.6-0.02
% tdms_path_out_600_3000_0 = r"G:\js\results\x_out_0.6_0.02_num6_feed3000-15-08-44.tdms"
% tdms_path_out_600_3000_1 = r"G:\js\results\x_out_deg180_0.6_0.02_num6_feed3000-19-54-43.tdms"
% tdms_path_out_600_5000_0 = r"G:\js\results\x_out_0.6_0.02_num6_feed5000-15-07-52.tdms"
% tdms_path_out_600_5000_1 = r"G:\js\results\x_out_deg180_0.6_0.02_num6_feed5000-19-54-00.tdms"

data=convertTDMS(true,'G:\js\results\new_all_feed_x_10000-9-29-06.tdms')
microphone=data.Data.MeasuredData(3).Data(:,1);
feed_x=data.Data.MeasuredData(4).Data(:,1);
feed_y=data.Data.MeasuredData(5).Data(:,1);
save('G:\js\code\data_mat_analysis\nomal\x_0.6_0.02_num4_feed3000-19-01-34_fx3000.mat','feed_x')
save('G:\js\code\data_mat_analysis\nomal\x_0.6_0.02_num4_feed3000-19-01-34_fy3000.mat','feed_y')
save('G:\js\code\data_mat_analysis\nomal\x_0.6_0.02_num4_feed3000-19-01-34_microphone3000.mat','microphone')
# coding=utf-8
from models.MT1_x_feed.run_model_gpu import Model as mt1_x_feed_model
from models.MT2_ae_raw.run_model_gpu import Model as mt2_ae_raw_model
from models.MT2_ae_rms.run_model_gpu import Model as mt2_ae_rms_model
from models.MT2_micphone.run_model_gpu import Model as mt2_micphone_model
from models.MT2_spindle_z.run_model_gpu import Model as mt2_spindle_z_model
from models.MT2_x_feed.run_model_gpu import Model as mt2_x_feed_model
from models.MT2_y_feed.run_model_gpu import Model as mt2_y_feed_model
from models.MT3_micphone.run_model_gpu import Model as mt3_micphone_model
from models.MT3_x_feed.run_model_gpu import Model as mt3_x_feed_model
from models.MT3_y_feed.run_model_gpu import Model as mt3_y_feed_model
from models.TG_x_feed.run_model_gpu import Model as tg_x_feed_model
from models.TG_y_feed.run_model_gpu import Model as tg_y_feed_model

from utils.common_config import get_sensor_checkpoints

"""
    根据sensor_id确定checkpoints路径和所要调用的模型
"""


class Model_Util(object):
    def __init__(self):
        # home 路径
        self.home_path = "/wzs/model/three_model"
        self.sensor_model = {
            "cDAQ9189-1D71297Mod1/ai3": mt1_x_feed_model(),
            "cDAQ9189-1D91958Mod5/ai1": mt2_ae_rms_model(),
            "cDAQ9189-1D71297Mod5/ai1": mt2_micphone_model(),
            "cDAQ9189-1D71297Mod3/ai2": mt2_spindle_z_model(),
            "cDAQ9189-1D71297Mod3/ai3": mt2_x_feed_model(),
            "cDAQ9189-1D71297Mod2/ai1": mt2_y_feed_model(),
            "cDAQ9189-1D71297Mod5/ai2": mt3_micphone_model(),
            "cDAQ9189-1D71297Mod4/ai3": mt3_x_feed_model(),
            "cDAQ9189-1D71297Mod2/ai2": mt3_y_feed_model(),
            "cDAQ2Mod2/ai3": tg_x_feed_model(),
            "cDAQ2Mod3/ai0": tg_y_feed_model()
        }
        self.checkpoints = get_sensor_checkpoints()["one"]

    # 测试
    def call_model_test(self, sensor_id, data, created):
        path, model = self.get_path_and_model(sensor_id)
        if path is None or model is None:
            return None
        else:
            path = self.home_path + path
        # 故障类别, 该故障类别发生的概率
        fault_pred_class, show_pro_fault_pred_class = model.run_test(data, path)
        if fault_pred_class is not None and fault_pred_class != 0:
            breakdownData = {}
            breakdownData["collectInterFaceNo"] = str(sensor_id)
            breakdownData["breakdownType"] = str(fault_pred_class)
            breakdownData["percent"] = str(int(show_pro_fault_pred_class * 100))
            breakdownData["created"] = str(created)
            return breakdownData
        else:
            return None

    # 训练
    def call_model_train(self, sensor_id, samples_train, labels_train):
        path, model = self.get_path_and_model(sensor_id)
        if path is None or model is None:
            return None, None
        version = model.run_train(samples_train, labels_train, path)
        return path, version

    def get_path_and_model(self, sensor_id):
        path = self.checkpoints.get(sensor_id)
        model = self.sensor_model.get(sensor_id)
        return path, model

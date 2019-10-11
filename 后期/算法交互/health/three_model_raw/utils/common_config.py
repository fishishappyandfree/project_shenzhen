import json


def get_common_config():
    f = open("/wzs/model/three_model/utils/esgyn_config.json", "r", encoding="utf-8")
    return json.loads(f.read())


def get_offline_config():
    f = open("/wzs/model/three_model/utils/offline_config.json", "r", encoding="utf-8")
    return json.loads(f.read())


def get_breakdown_type():
    f = open("/wzs/model/three_model/utils/breakdown_type.json", "r", encoding="utf-8")
    return json.loads(f.read())


def update_offline_config(offline_config):
    json_str = json.dumps(offline_config, indent=4)
    with open("/wzs/model/three_model/utils/offline_config.json", "w", encoding="utf-8") as file:
        file.write(json_str)


def get_sensor_checkpoints():
    f = open("/wzs/model/three_model/utils/sensor_checkpoint.json", "r", encoding="utf-8")
    return json.loads(f.read())



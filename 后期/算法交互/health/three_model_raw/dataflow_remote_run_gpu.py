# coding=utf-8
import json
import sys
import time
from utils.model_util import Model_Util as model_util


def dataflow_remote_run(sensor_id):
  try: 
    f = open("/opt/foxconn/mock_data/data.json", "r", encoding="utf-8")
    full_data = json.loads(f.read())
    data = full_data[:2048]
    created = int(time.time()*1000)
    model = model_util()
    print(json.dumps(model.call_model_test(sensor_id, data, created)))
  except:
    sys.stderr.write("error")


if __name__ == '__main__':
    sensor_id = sys.argv[1]
    dataflow_remote_run(sensor_id)


ps -ef|grep OfflineData.py |grep -v grep | awk '{print $2}' |xargs kill -9


#!/bin/bash
pid=`ps -ef|grep schedule_offline_run.py |grep -v grep | awk '{print $2}'`
if [[ $pid -gt 0 ]]; then
  echo 'stop schedule_offline_run.py'
  kill -9 $pid
fi


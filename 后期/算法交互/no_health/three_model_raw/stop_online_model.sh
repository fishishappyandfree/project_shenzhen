#!/bin/bash
pid=`ps -ef|grep dataflow_online_run.py |grep -v grep | awk '{print $2}'`
if [[ $pid -gt 0 ]]; then
  echo 'stop dataflow_online_run.py'
  kill -9 $pid
fi

   

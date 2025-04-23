#!/bin/bash

pid=$(pgrep -f "python3 /home/pi/Rosmaster/rosmaster/rosmaster_main.py")
if [ ! -z "$pid" ]; then
  kill -9 $pid
fi

#gnome-terminal --tab -- bash -i '/home/pi/mycar_ws/lidar_pub.sh'
gnome-terminal --tab -- bash -i '/home/pi/mycar_ws/data_pub.sh'
gnome-terminal --tab -- bash -i '/home/pi/mycar_ws/main_program.sh'
gnome-terminal --tab -- bash -i '/home/pi/mycar_ws/lidar_acquisition.sh'


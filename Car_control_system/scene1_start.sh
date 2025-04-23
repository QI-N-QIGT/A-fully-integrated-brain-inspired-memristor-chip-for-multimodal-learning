#!/bin/bash

gnome-terminal --tab -- bash -i '/home/pi/mycar_ws/data_pub.sh'
gnome-terminal --tab -- bash -i '/home/pi/mycar_ws/scene1_program.sh'
gnome-terminal --tab -- bash -i '/home/pi/mycar_ws/video_acquisition.sh'


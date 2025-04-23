#rosbag record -o video_data  /output_video_frame
rosbag record -o mutil_data /scan /car_speed /audio_data /output_video_frame
#rosbag record -o lidar_data /scan /car_speed /audio_data 
#rosbag record -o lidar_data /scan /car_speed 

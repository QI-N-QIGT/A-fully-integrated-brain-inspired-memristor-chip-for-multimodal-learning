#!/usr/bin/env python
import rospy
from rosbag import Bag

rospy.init_node('lidar_reader')
bag_path = '/home/pi/my_ws/lidar_data_2024-03-20-10-06-52.bag'
#open bag file 
bag = Bag(bag_path)
#traverse the bag 
for topic, msg, t in bag.read_messages(topics=['/scan']):
    #parse msg
    #print topic name and time 
    #print('topic: %s time: %s' % (topic,t))
    rospy.loginfo("Topic: %s Time: %s", topic, t)
#close the bag file 
bag.close()

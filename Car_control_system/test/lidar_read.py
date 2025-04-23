import rospy
from rosbag import Bag
import sys

if len(sys.argv) > 1:
    bag_path = sys.argv[1]
    print("first arg:", bag_path)
else:
    print("usage: python lidar_read.py [file_path].")
    sys.exit()

#set ros node 
#rospy.init_node('lidar_data_reader')
#open bag file 
bag = Bag(bag_path)
#traverse the bag 
#for topic, msg, t in bag.read_messages(topics=['/scan', '/car_speed']):
for topic, msg, t in bag.read_messages(topics=['/scan']):
    #parse msg
    #print topic name and time 
    print('topic: %s time: %s' % (topic,t))
    print(msg)
    #rospy.loginfo("Topic: %s Time: %s", topic, t)
#close the bag file 
bag.close()

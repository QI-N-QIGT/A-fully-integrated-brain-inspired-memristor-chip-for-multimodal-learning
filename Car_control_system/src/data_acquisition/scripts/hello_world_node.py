#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import signal

def sigint_handler(signal, frame):
    print("Shutting down the node...")
    rospy.signal_shutdown('Interrupted')
    raise KeyboardInterrupt


def callback(event):
    rospy.loginfo("Hello World!")

rospy.init_node('hello_world_node')
signal.signal(signal.SIGINT, sigint_handler)
rospy.loginfo("Hello World!")
#pub = rospy.Publisher('chatter', String, queue_size=10)
#rospy.Timer(rospy.Duration(1.0), callback)
if __name__ == '__main__':
    try:
        print("aaaaaaaaaaaa")
        rospy.spin()
        print("bbbbbbbbbbb")
    except KeyboardInterrupt:
        print("Shutting down")

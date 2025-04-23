#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
    cv2.imshow('frame', cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown('User pressed `q`')

rospy.init_node('image_subscriber')
bridge = CvBridge()
sub = rospy.Subscriber("/output_video_frame", Image, callback)
rospy.spin()

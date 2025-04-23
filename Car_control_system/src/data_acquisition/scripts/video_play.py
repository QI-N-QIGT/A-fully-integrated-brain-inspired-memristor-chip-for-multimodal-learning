#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def is_window_alive(name):
    return cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) >=0

def callback(data):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")  # change format to CV format
    #print(cv_image)
    print(cv_image.shape)

    if is_window_alive('frame'):
        cv2.imshow('frame', cv_image)
    else:
        rospy.signal_shutdown('User closed window')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown('User pressed `q`')

def listener():
    rospy.init_node('video_capture_node')
    rospy.Subscriber("/output_video_frame", Image, callback)
    rospy.spin()

def main():
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.setWindowTitle('frame', 'frame')
    #cv2.resizeWindow('frame', 640, 480)
    listener()

if __name__ == '__main__':
    main()

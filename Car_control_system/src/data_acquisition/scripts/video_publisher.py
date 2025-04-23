#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def publish_video_frame(frame):
    bridge = CvBridge()
    ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
    pub = rospy.Publisher("/output_video_frame", Image, queue_size=10)
    pub.publish(ros_image)

def main():
    rospy.init_node('video_frame_publisher') 
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    try:
        frame_size_str = rospy.get_param('frame_size')
        rospy.loginfo("Frame size str is: %s", frame_size_str)
    except KeyError:
        rospy.logerror("Parameter 'frame_size' not found.")
        frame_size = (640,480)
    frame_size = tuple(map(int, frame_size_str.split('x')))
    rospy.loginfo("Frame size is: %s", frame_size)

    try:
        frame_rate = rospy.get_param('frame_rate')
        rospy.loginfo("Frame rate is: %d", frame_rate)
    except KeyError:
        rospy.logerror("Parameter 'frame_rate' not found.")
        frame_rate = fps

    if frame_rate > fps or frame_rate <= 0:
        frame_rate = fps
        print("frame_rate is set to ", frame_rate)
    else:
        print('try to find the nearest supported framerate.') 
        while (fps % frame_rate != 0) and frame_rate != fps :
            frame_rate += 1
        print("frame_rate is set to ", frame_rate)

    modulus = fps//frame_rate
    i = 0

    #rate = rospy.Rate(30)  # 30Hz
    while not rospy.is_shutdown():
        if i == fps:
            i = 0
        ret, frame = cap.read()
        if i % modulus != 0: #skip this frame
            i += 1
            continue
        print("frame index:", i)
        i += 1
        if ret:
            resized_frame = cv2.resize(frame, frame_size)
            publish_video_frame(resized_frame)
            print('video frame publishing')
        #rate.sleep()
    cap.release()

if __name__ == '__main__':
    main()

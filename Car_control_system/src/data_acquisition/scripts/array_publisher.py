#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

def main():
    rospy.init_node('array_publisher', anonymous=True)
    pub = rospy.Publisher('my_array_topic', Float32MultiArray, queue_size=10)
    two_dimensional_array = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    array_msg = Float32MultiArray()
    array_msg.layout.dim.append(MultiArrayDimension())
    array_msg.layout.dim[0].label = "row"
    array_msg.layout.dim[0].size = len(two_dimensional_array)
    array_msg.layout.dim[0].stride = len(two_dimensional_array[0])
    array_msg.layout.dim.append(MultiArrayDimension())
    array_msg.layout.dim[1].label = "column"
    array_msg.layout.dim[1].size = len(two_dimensional_array[0])
    array_msg.layout.dim[1].stride = 1  # Stride for contiguous data

    for row in two_dimensional_array:
        array_msg.data.extend(row)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(array_msg)
        #rate.sleep()
        print('4444')
        
if __name__ == '__main__':
    try:
        main()
    #except rospy.ROSInterruptException:
    except KeyboardInterrupt:
        pass

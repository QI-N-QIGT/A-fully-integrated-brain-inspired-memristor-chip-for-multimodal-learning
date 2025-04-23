#!/usr/bin/env python
import pyaudio
import rospy
import time
from std_msgs.msg import String

pub = rospy.Publisher('audio_data', String, queue_size=10)
rospy.init_node('audio_publisher', anonymous=True)

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    #print(f"Device {i}: {device_info['name']}")
    print("Device %d:%s" % (i, device_info['name']))

chunk = 1600
sample_format = pyaudio.paInt16  
channels = 1 
fs =16000 
i = 0
data = b''

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)


try:
    while not rospy.is_shutdown():
        data += stream.read(chunk, False)
        if (i+1)%10 == 0:
          audio_str = data.encode('base64')
          msg = String()
          msg.data = audio_str
          pub.publish(msg)
          data = b''
        i += 1 
except KeyboardInterrupt:
    print('keyboardInterrupt')
    #stream.stop_stream()
    stream.close()
    p.terminate()
except IOError:
    print('IOError')
    #stream.stop_stream()
    stream.close()
    p.terminate()

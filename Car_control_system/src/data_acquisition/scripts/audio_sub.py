#!/usr/bin/env python
import pyaudio
import rospy
import base64
from std_msgs.msg import String


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True)
 
def callback(msg):
    audio_data_base64 = msg.data
    #print(audio_data_base64)
    audio_data = base64.b64decode(audio_data_base64)
    #print(audio_data)
    stream.write(audio_data)

rospy.init_node('audio_player')
rospy.Subscriber("/audio_data", String, callback)

try:
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()

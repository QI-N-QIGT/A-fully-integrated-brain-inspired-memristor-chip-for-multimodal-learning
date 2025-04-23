#!/usr/bin/env python

import pyaudio
from std_msgs.msg import String
import base64
import time

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    #print(f"Device {i}: {device_info['name']}")
    print("Device %d:%s" % (i, device_info['name']))

chunk = 4410
sample_format = pyaudio.paInt16  
channels = 1 
fs = 44100 
i = 0
data = b''

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                input_device_index=0)

try:
    while True:
        data += stream.read(chunk, False)
        #print(data)
        #print(len(data))
        if (i+1)%10 == 0:
          print(time.time())
          data = b''
        i += 1
        #audio_str = data.encode('base64')
        #msg = String()
        #msg.data = audio_str
        #print(msg)
        #data1 = base64.b64decode(msg.data)
        #print(data1)
        
        
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()


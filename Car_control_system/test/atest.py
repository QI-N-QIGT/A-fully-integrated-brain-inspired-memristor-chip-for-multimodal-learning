#!/usr/bin/env python

import pyaudio
import wave
import time

chunk = 16000 
sample_format = pyaudio.paInt16  
channels = 1  
fs = 16000  
seconds = 10 
filename = "output.wav"  
p = pyaudio.PyAudio()  


stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  


for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk, False)
    frames.append(data)


stream.stop_stream()
stream.close()


p.terminate()

print("rec finished")

wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

print "saved as %s" % filename

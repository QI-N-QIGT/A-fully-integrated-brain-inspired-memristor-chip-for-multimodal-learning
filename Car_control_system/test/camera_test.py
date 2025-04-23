#!/usr/bin/env python
import cv2
import time
import sys

def is_window_alive(name):
    return cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) >=0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
fps = int(cap.get(cv2.CAP_PROP_FPS))

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowTitle('frame', 'frame')
cv2.resizeWindow('frame', 320, 240)

if len(sys.argv) > 1:
    frame_rate = int(sys.argv[1])
    print(f"fps {frame_rate} expected")
else:
    print("default fps expected.")
    frame_rate = fps

if frame_rate > fps or frame_rate <= 0:
    frame_rate = fps
    print(f"frame_rate is set to {frame_rate}.")
else:
    print('try to find the nearest supported framerate.') 
    while (fps % frame_rate != 0) and frame_rate != fps :
        frame_rate += 1
    print(f"frame_rate is set to {frame_rate}.")

modulus = fps//frame_rate
i = 0
while True:
    if i == fps:
        i = 0
    ret, frame = cap.read()
    if not ret:
        print("read frame failed.")
        break

    if i % modulus != 0:
        i += 1
        continue

    print("frame index:", i)
    i += 1
    
    print(time.time())
    resized_frame = cv2.resize(frame, (320, 240))
    if is_window_alive('frame'):
        cv2.imshow('frame', resized_frame)
    else:
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import sys

cap = cv2.VideoCapture(0)
supported_fps = []

for fps in range(1, 60):
    cap.set(cv2.CAP_PROP_FPS, fps)
    if cap.get(cv2.CAP_PROP_FPS) == fps:
        supported_fps.append(fps)
    else:
        print(f"FPS {fps} is not supported.")

print(f"Supported FPS: {supported_fps}")

if __name__ == '__main__' :

    fps = int(cap.get(cv2.CAP_PROP_FPS))

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

    print(f"modulus is {modulus}.")
        
    
    

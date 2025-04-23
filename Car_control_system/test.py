import time
import signal

def sigint_handler(signal, frame):
    print("signnal handler...")
    print(signal)
    #rospy.signal_shutdown('Interrupted')
    #raise KeyboardInterrupt

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
#signal.signal(signal.SIGTERM, sigint_handler)
signal.signal(signal.SIGTSTP, sigint_handler)
#signal.signal(signal.SIGCONT, sigint_handler)

try:
    while True:
        time.sleep(1)
        print("aaa")
except KeyboardInterrupt:
    print("ctr+c")

print('exit ok')

import signal
import sys
import time

def signal_handler(signal, frame):
    print(f"{signal}")
    print("1111")
    sys.exit(0)

signals = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]
for sig in signals:
    signal.signal(sig, signal_handler)

print("sleep...")
while True:
    time.sleep(1)


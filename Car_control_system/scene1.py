#!/usr/bin/env python3
from Rosmaster_Lib import Rosmaster
import time
import signal

car = Rosmaster()


def sigint_handler(signal, frame):
    #car.set_car_motion(0, 0, 0)
    raise KeyboardInterrupt


if __name__ == '__main__':

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    signal.signal(signal.SIGHUP, sigint_handler)
    signal.signal(signal.SIGCONT, sigint_handler)

    try:
        while True:
            car.set_car_motion(0.1, 0, 0)
            time.sleep(3)
            car.set_car_motion(-0.1, 0, 0)
            time.sleep(3)

    except KeyboardInterrupt:
        car.set_car_motion(0, 0, 0)


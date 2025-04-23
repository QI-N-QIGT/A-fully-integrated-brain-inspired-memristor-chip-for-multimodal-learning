#!/usr/bin/python3

from Rosmaster_Lib import Rosmaster
import time

g_bot = Rosmaster(debug=True)
g_bot.create_receive_threading()
gap = 0.1
while True:
  time.sleep(gap)
  g_bot.set_car_motion(0.2, 0.2, 0) 
  time.sleep(gap)
  g_bot.set_car_motion(0.5, 0.5, 0) 

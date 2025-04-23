#!/usr/bin/python3


from Rosmaster_Lib import Rosmaster 
import threading

bot = Rosmaster()

def timer_func():
    bot.set_colorful_effect(0)
    pass

bot.set_colorful_effect(2)
threading.Timer(0.5, timer_func).start()

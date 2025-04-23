import time
import a111sdk
a111sdk.a111_power_on()
a111sdk.a111_hw_sys_init()
sleep_time = 60
for i in range(sleep_time):
    print()
    time.sleep(1)
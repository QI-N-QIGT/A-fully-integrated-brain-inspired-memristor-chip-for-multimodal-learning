import sys
import time
import a111sdk
args = sys.argv
if len(args) == 2:
    chip_id = int(args[1])
else:
    print()
    exit()
a111sdk.a111_power_on(chip_id)
a111sdk.a111_hw_sys_init(chip_id)
sleep_time = 60
for i in range(sleep_time):
    print()
    time.sleep(1)
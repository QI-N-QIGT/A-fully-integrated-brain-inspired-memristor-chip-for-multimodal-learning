import a111sdk
import sys
args = sys.argv
if len(args) == 2:
    chip_id = int(args[1])
else:
    print()
    exit()
a111sdk.a111_power_off(chip_id)
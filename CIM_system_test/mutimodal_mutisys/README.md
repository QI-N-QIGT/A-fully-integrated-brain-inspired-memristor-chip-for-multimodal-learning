# mutimodal_mutisys

## Requirements

A111-sdk installation

```
#clone a copy of A111_LNX_SDK_PYTHON onto the system
git clone http://git.icfc.cc:7383/hardware_system/A111_LNX_SDK_PYTHON.git

# Install the a111sdk package based on README.md in the A111_LNX_SDK_PYTHON directory
cd A111_LNX_SDK_PYTHON/a111sdk-pack
python3 setup.py bdist_wheel
pip3 install ./dist/a111sdk-1.0.0-py3-none-any.whl --force-reinstall


# (optional)
# After each boot need to wait for a few minutes, the equipment is stable before starting to operate,，
If you don't want to wait, you can do 'cat /dev/urandom' and wait 5 seconds to interrupt the command and start doing something else
# Actively reads the random number pool
cat /dev/urandom
#  Check the size of the random number pool. If it is greater than 128, numpy can be imported normally. Otherwise, it can be imported normally after 128 is generated automatically
cat /proc/sys/kernel/random/entropy_avail  
```

A111-sdk-example

```
# Copy a copy of A111-sdk-example to the system, and debug the hardware according to the readme file on the example
git clone http://git.icfc.cc:7383/haoxiaolong/a111_example.git

# Get the latest example package
git pull 
```

A111- Tool Chain Download and  installation

```
cd tool_chains
git clone http://git.icfc.cc:7383/E100/e100-irtool.git
cd e100-irtool && pip  install -e . && cd ..
git clone http://git.icfc.cc:7383/E100/Model2IR.git
cd Model2IR && pip  install -e . && cd ..
git clone http://git.icfc.cc:7383/E100/cimruntime.git
cd cimruntime && pip  install -e . && cd ..
git clone http://git.icfc.cc:7383/E100/e100-irmapper.git
cd e100-irmapper && pip  install -e . && cd ..
```

A111 Install other dependent environments

```
# Install torch torchvision torchaudio
pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# Install other environments
pip3 install -r requirement.txt

```

## Instructions

※ Operation instruction:

```
# Before using the A111-sdk, complete hardware initialization
python  hardware_power_on.py  chip_id[0-8]    # python  hardware_power_on.py  0

# If a hardware error occurs or the chip needs to be replaced, power off the hardware
python3 hardware_power_off.py chip_id[0-8]    # python  hardware_power_off.py  0
```

## Run code
```
python main.py
```

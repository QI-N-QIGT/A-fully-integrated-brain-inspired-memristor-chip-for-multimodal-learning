# Multimodal on-chip learning code for visual failure 

## Requirements

A111-sdk installation

```
#clone a copy of A111_LNX_SDK_PYTHON onto the system
git clone http://git.icfc.cc:7383/hardware_system/A111_LNX_SDK_PYTHON.git

# Install the a111sdk package based on README.md in the A111_LNX_SDK_PYTHON directory
cd A111_LNX_SDK_PYTHON/a111sdk-pack
python3 setup.py bdist_wheel
pip3 install ./dist/a111sdk-1.0.0-py3-none-any.whl --force-reinstall

```
A111-Tool-chain installation

```
cd tool_chains
pip3 install */
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
python3 hardware_power_on.py

# If a hardware error occurs or the chip needs to be replaced, power off the hardware
python3 hardware_power_off.py
```

## Run code
```
python main.py
cd mutimodal_mutisys
python main.py

```

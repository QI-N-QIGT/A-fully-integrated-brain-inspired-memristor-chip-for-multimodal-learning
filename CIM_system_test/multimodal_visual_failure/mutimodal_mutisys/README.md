# Multimodal on-chip learning code for missing images

## Requirements

A111- Tool Chain Download

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

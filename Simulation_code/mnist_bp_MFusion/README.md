This folder contains code for incremental learning experiments on the MNIST dataset using **MFusion** and **Backpropagation (BP)**, with comparisons between **digital environments** and **CIM (Computing-In-Memory) environments** (including quantization and weight noise).


| Script Name                          | Description                                                                                               |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `sc_task_domain_incre_bp.py`         | BP-based incremental learning in **digital environment** (no quantization/noise).                         |
| `sc_task_domain_incre_bp_qn_man.py`  | BP-based incremental learning in **CIM environment** with manual quantization and weight noise.           |
| `sc_task_domain_incre_mfusion.py`    | MFusion-based incremental learning in **digital environment**.                                            |
| `sc_task_domain_incre_mfusion_qn.py` | MFusion-based incremental learning in **CIM environment** with automatic quantization and noise modeling. |


# requirement

pandas==1.1.4
torch==2.0.1
torchvision==0.15.2
numpy==1.19.5
matplotlib==3.3.3

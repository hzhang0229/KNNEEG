## KNN Attention-based EEGViT Model


## Overview
KNN-EEGViT is an adapted version of EEGViT, which is a hybrid Vision Transformer (ViT) incorporated with Depthwise Convolution in patch embedding layers. We add the KNN-attention feature to the EEGVit model and expect the boost of model performance and accuracy on EEGEyeNet benchmarking tasks.

Original EEGVit:https://github.com/ruiqiRichard/EEGViT.git
## Code download
```bash
git clone https://github.com/hzhang0229/KNNEEG.git
```
## Dataset download
Download data for EEGEyeNet absolute position task
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```

## Pip Installation
Just in case you are using VM without pip. 
```bash
sudo apt update
sudo apt install python3-pip
```

### Requirements

First install the general_requirements.txt

```bash
pip3 install -r general_requirements.txt 
```
We do suggest install each item listed in this txt. file individually because our research shows the above instruction is not working. We will fix this issue soon.

### Pytorch Requirements

```bash
pip install torch torchvision torchaudio
```



# KNNEEG

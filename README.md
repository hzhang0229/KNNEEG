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

## Requirements

First install the general_requirements.txt

```bash
pip3 install -r general_requirements.txt 
```
We do suggest install each item listed in this txt. file individually just in case our research shows the above instruction is not working. 

## Pytorch Requirements

```bash
pip install torch torchvision torchaudio
```

## Run the program

```bash
python3 main.py
```
## Environment
```bash
python 3.8.10
NVIDIA T4 GPU
×86/64 Architecture
8 vCPUs with 32GB Memory
32GB is the minimum memory requirement for this program.
```
## Common Problems
1. If you have a NVIDIA GPU but the system prints "Using CPU", Please check if you have the GPU Driver. 
https://cloud.google.com/compute/docs/gpus/install-grid-drivers#install-drivers
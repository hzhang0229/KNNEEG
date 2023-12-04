## KNN Attention-based EEGViT Model


## Overview
KNN-EEGViT is an adapted version of EEGViT, which is a hybrid Vision Transformer (ViT) incorporated with Depthwise Convolution in patch embedding layers. We add the KNN-attention feature to the EEGVit model and expect the boost of model performance and accuracy on EEGEyeNet benchmarking tasks.

Original EEGViT:https://github.com/ruiqiRichard/EEGViT.git
## Code download
```bash
git clone https://github.com/hzhang0229/KNNEEG.git
```
## Dataset download
Download data for EEGEyeNet absolute position task in Linux environment
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

First install the requirements.txt

```bash
pip3 install -r requirements.txt 
```
We do suggest install each item listed in this txt. file individually just in case our research shows the above instruction is not working. 

And then install the pytorch package independently.brew install pandoc
```bash
pip install torch torchvision torchaudio
```

## Run the program

```bash
python3 main.py
```

## Required Environment for Reproducibility
```bash
Linux System
python 3.8.10
Google Cloud VM
NVIDIA T4 GPU
×86/64 Architecture
8 vCPUs with 32GB Memory
32GB System Memory
```
## Common Problems and Troubleshooting
1. If you have a NVIDIA GPU but the system prints "Using CPU", Please check if you have the GPU Driver. 
https://cloud.google.com/compute/docs/gpus/install-grid-drivers#install-drivers
2. Please ensure you have enough memory on your VM to download the dataset, which is around 11 GB. 


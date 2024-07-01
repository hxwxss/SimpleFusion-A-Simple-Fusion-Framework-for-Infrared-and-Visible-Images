# SimpleFusion
Official PyTorch Implementation of [SimpleFusion: A Simple Fusion Framework for
Infrared and Visible Images](https://arxiv.org/abs/2), Accepted by PRCV 2024

# Platform
1. Python 3.10 
2. Pytorch >= 1.8


# Dataset 
The training dataset is KAIST Multispectral Pedestrian Detection Benchmark dataset, which is available at [here](https://soonminhwang.github.io/rgbt-ped-detection/). 

However, to facilitate the use of these image data, we have reorganized the entire dataset as follows:


Dataset/  
├── KAIST/  
├──├── lwir/  
├──├──├──infrand image_1  
├──├──├──infrand image_2  
......  
├──├── vis/  
├──├──├──visible image_1  
├──├──├──visible image_2  
.......


# How to use this project?
## 1. metrics
You can verify our work by downloading the metrics.zip file from the project. The code in metrics.zip is sourced from an open-source project on GitHub, and we express our gratitude to the original authors. When using the code, please remember to modify the file paths accordingly.
## 2. Coming soon.

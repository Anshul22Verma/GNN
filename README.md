# GNN
graph neural network

A colaborative work of :
1. Aalok,  aalok@iitgn.ac.in
1. Anshul, av.vermaans@gmail.com
1. Ashish, ashish.tiwari@iitgn.ac.in
1. Prajwal, singh_prajwal@iitgn.ac.in 

Revision 1.01, 16 October 2020

## Setup 

### Set up conda
1. Create conda environment

```
conda env create -f env.yml
```
or
```
conda create -n shanmuga_iitg -f env.yml
``` 

1. Setup pytorch and pytorch-geometric for the environment

Install pytorch 1.5.0 make sure to install the CPU only version in a PC without CUDA-device (GPUs)

```
conda install -c pytorch pytorch=1.5.0 torchvision=0.6.0 cpuonly
```

build wheels for this version and install essentials for CPU or GPU versions
[ref](https://github.com/rusty1s/pytorch_geometric)

```
pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric==1.5.0
```

Replace cpu with `cu92`, `cu101` or `cu102` for other CUDA-devices GPUs.

If you are using windows you might need additional .dll files to be able to work with pytorch [ref](https://github.com/pytorch/pytorch/issues/37022#issuecomment-618541367)

To resolve the issue download [additional packages](https://drive.google.com/drive/folders/1rAlAVrgh-qCz_WvSxEU-IOCGShKPYaut?usp=sharing) and copy them to `C:\Windows\System32`, this should resolve all the issues and allow you to use PyTorch.

Environment ready.

1. Activate the environment

```
conda activate shanmuga_iitg
```

### Development environment 

#### Pycharm 

#### Jupyter Notebook 

#### Spyder

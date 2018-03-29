# Introduction

This directory contains source codes for slice pooling/unpooling layers. Follow the steps below to compile the source codes before starting training.


1. locate the directory of CUDA in your machine. In my machine, it is `/usr/local/cuda-8.0/`. Set the environmental variable as below:
```bash
$ CPATH=/usr/usc/cuda/8.0/include/
```

2. compile slice pooling layer by commands below:
```bash
$ cd slice_pool_layer/src/cuda
$ nvcc -c -o slice_pool_layer_cuda_kernel.cu.o slice_pool_layer_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
$ cd ../../
$ python build
```

Here `sm_61` denotes the CUDA computing capability of the GPU. Update it for your GPUs if neccessary.

3. compile slice unpooling layer by commands below:
```bash
$ cd slice_unpool_layer/src/cuda
$ nvcc -c -o slice_unpool_layer_cuda_kernel.cu.o slice_unpool_layer_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
$ cd ../../
$ python build
```


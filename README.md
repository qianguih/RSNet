# Introduction

This is the official inplementation of [Recurrent Slice Networks for 3D Segmentation on Point Clouds](https://arxiv.org/abs/1802.04402) (RSNet), which is going to appear in CVPR 2018.

RSNet is a powerful and conceptually simple  network for 3D point cloud segmentation tasks. It is fast and memory-efficient. In this repository, we release codes for training a RSNet on the S3DIS segmentation dataset. Training on other datasets can be easily achieved by following the same process.


# Citation
If you find our work useful in your research, please consider citing:

        @article{huang2018recurrent,
            title={Recurrent Slice Networks for 3D Segmentation on Point Clouds},
            author={Huang, Qiangui and Wang, Weiyue and Neumann, Ulrich},
            journal={arXiv preprint arXiv:1802.04402},
            year={2018}
        }

# Dependencies
- `python` (tested on python2.7)
- `PyTorch` (tested on 0.3.0)
- `cffi`
- `h5py`

# Installation
1. Clone this repository.
2. Compile source codes for slice pooling/unpooling layers by following the readme file in `layers`


# Data Preparation
1. Process the S3DIS dataset by following the readme file in `data`.

# Train
1. Launch training by the command below:
```bash
$ python train.py
```

Type `python train.py --help` for detailed input options. Be default, it will start the training by using `Area 5` as testing set and others as training set.

During training, visualizations (.obj files) of intermediate predictions will be dumped into the folder `results` after each epoch. And they will be evaluated and saved in `test_log.txt`.

# License
Codes in this repository are released under MIT License (see LICENSE file for details).





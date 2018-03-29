# Introduction

This directory contains source codes for preparing the S3DIS dataset for RSNet. A large part of the util functions come from the [PointNet](https://github.com/charlesq34/pointnet) repository. Thanks to [@charlesq34](https://github.com/charlesq34). Follow the steps below to generate data files used for RSNet.

1. Download the raw dataset from the [S3IDS website](http://buildingparser.stanford.edu/dataset.html#Download). Note that the file to download is `Stanford3dDataset_v1.2_Aligned_Version.zip`.  Unzip the file to the folder you want. We note the folder path as `RAW_DATA`

2. Re-organize the raw dataset by the following command:
```bash
$ python collect_indoor3d_data.py --raw_data_dir RAW_DATA
```
By default, processed dataset will be stored in `./stanford_indoor3d`. Set the `--output_folder` flag to overwrite the default output path. We denote the output folder in this step as `INDOOR3D_DATA_DIR`

3. Generate training and testing files for RSNet by the following commands:
```bash
$ python gen_indoor3d_h5.py --indoor3d_data_dir INDOOR3D_DATA_DIR --split train
$ python gen_indoor3d_h5.py --indoor3d_data_dir INDOOR3D_DATA_DIR --split test --stride 1.0
```

By default, these two commands generate files using data in `Area_5` as testing data and others as training data. Type `python gen_indoor3d_h5.py --help` for other generation options.

Be default, there will be two new folders `indoor3d_sem_seg_hdf5_data_Area_5_1.0m_0.5s_train` and `indoor3d_sem_seg_hdf5_data_Area_5_1.0m_1.0s_test` containing training and testing files, respectively.

In the end, update the dataset setting in `load_data.py` if neccessary. Leave it alone if using default settings in all steps.




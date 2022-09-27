# ABCD

This repository contains the code (in PyTorch) for "ABCD: Attentive Bilateral Convolutional Network for Robust Depth Completion" paper (RA-L).

## Requirements

* Python3.6
* PyTorch 1.6
* KITTI Depth Completion dataset
* VirtualKITTI2 dataset

## ENVIRONMENT

```
conda create -n abcd python=3.6
conda activate abcd
pip install -r requirements.txt
```

## Set up
```
cd lib 
python build_khash_cffi.py 
cd ..
```

## Data
```
mkdir KITTI_RAW && cd KITTI_RAW
chmod +x raw_data_downloader.sh
./raw_data_downloader.sh
chmod +x depth_completion_data_downloader.sh
./depth_completion_data_downloader.sh
```
```
.
└── KITTI_RAW
    ├── 2011_09_26
    |   ├── 2011_09_26_drive_0001_sync
    |   └── ...
    ├── 2011_09_28
    ├── 2011_09_29
    ├── 2011_09_30
    ├── 2011_10_03
    |
    ├── train
    |   ├── 2011_09_26_drive_0001_sync
    |   └── ...
    ├── val
    |
    └── depth_selection
        ├── test_depth_completion_anonymous
        ├── test_depth_prediction_anonymous
        └── val_selection_cropped
```

## Train
Set data_root and ckpt_dir in the train_ABCD.yaml file.
```
python main.py configs/train_ABCD.yaml
```

## Test
Set resume in the test_ABCD.yaml file.
```
python main.py configs/test_ABCD.yaml
```

## Acknowledgements
Our BCL implementation is based on https://github.com/laoreja/HPLFlowNet. 

## Citation
If you use our code or method in your work, please cite the following:
```
@article{jeon2021abcd,
  title={ABCD: Attentive Bilateral Convolutional Network for Robust Depth Completion},
  author={Jeon, Yurim and Kim, Hwichang and Seo, Seung-Woo},
  journal={IEEE Robotics and Automation Letters},
  year={2021},
  publisher={IEEE}
}
```
Please direct any questions to Yurim Jeon at yurimjeon1892@gmail.com

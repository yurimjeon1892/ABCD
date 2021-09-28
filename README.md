# ABCD

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

## Data
```
mkdir KITTI_raw && cd KITTI_raw
./raw_data_downloader.sh
```
```
.
└── KITTI_raw
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

## SET UP
```
cd lib 
python build_khash_cffi.py 
cd ..
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

## ACKNOWLEDGMENTS
Our BCL implementation is based on https://github.com/laoreja/HPLFlowNet. 

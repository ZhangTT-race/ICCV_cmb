# ICCV2021

## Prerequisites

`pip install requirements.txt`


## Data pre-progress
crop face and generate label file.

```shell
cd box_process
python face_crop_process.py --path /path/to/img_dir (e.g: /tmp-data/sys/HiFiMask-Challenge)
```

## train
```shell
bash train.sh
```

## generate submit.txt
```shell
python inference_test.py --version 0
```
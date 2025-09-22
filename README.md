# MICAPNet

This repository contains the implementation of the following paper:
> **Mutual Information-Guided Cross-modal Alignment Perception Network for Clinical Multi-modal Fusion**<br>

## Framework
<img width="880" height="480" src="https://github.com/1zhang-yi/MICAPNet/blob/main/assets/MICAPNet.png">

## Prerequisites
- Linux (We trained our codes on Ubuntu 20.04)
- Anaconda
- Python 3.9.0
- Pytorch 1.13.1

## Todo list
1. Please download the dataset [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
2. Follow the data pre-processing procedures and construct the paired dataset in https://github.com/nyuad-cai/MedFuse and https://github.com/dorothy-yao/drfuse.
3. Run the resize.py to resize the X-ray data.

## Train
Run the following command to train and test MICAPNet.
```bash
python main.py --ehr_data_dir <Your ehr folder> --image_dir <Your X-ray folder> --save_checkpoints <Your output path>
```

## Acknowledgement
- We appreciate all the developers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects: [MedFuse](https://github.com/nyuad-cai/MedFuse), [DrFuse](https://github.com/dorothy-yao/drfuse), [CLUB](https://github.com/Linear95/CLUB) and [CMG](https://github.com/haihuangcode/CMG).

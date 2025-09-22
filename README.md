# MICAPNet

This repository contains the implementation of the following paper:
> **Mutual Information-Guided Cross-modal Alignment Perception Network for Clinical Multi-modal Fusion**<br>

## Framework
<img width="880" height="380" src="https://github.com/1zhang-yi/IRSL_2.5DSAM/blob/main/assets/framework.png">

## Prerequisites
- Linux (We trained our codes on Ubuntu 20.04)
- Anaconda
- Python 3.9.0
- Pytorch 1.13.1

## Todo list
1. Please download the dataset [Abdomen](https://www.synapse.org/Synapse:syn3193805/wiki/217789)
2. Please download the pretrain SAM checkpoint [vit-b SAM Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
3. Run the preprocess_data script to preprocess the raw Abdomen dataset.

## Train
Run the following command to train IRSL-2.5DSAM.
```bash
python train.py --root_path <Your folder> --output <Your output path>
```

## Acknowledgement
- We appreciate all the developers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects: [Segment Anything](https://github.com/facebookresearch/segment-anything), [SAMed]( https://github.com/hitachinsk/SAMed), [SAM-Med2D](https://github.com/uni-medical/SAM-Med2D)

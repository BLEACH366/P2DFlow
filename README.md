# P2DFlow

P2DFlow is a protein ensemble generative model with SE(3) flow matching based on ESMFold, the ensembles generated by P2DFlow could aid in understanding protein functions across various scenarios.

Technical details and thorough benchmarking results can be found in our paper:
* Coming soon!

<p align="center">
    <img src="resources/workflow.jpg" width="600"/>
</p>

![P2DFlow](resources/gen_example.gif)


## Table of Contents
1. [Installation](#Installation)
2. [Prepare Dataset](#Prepare-Dataset)
3. [Model weights](#Model-weights)
4. [Training](#Training)
5. [Inference](#Inference)
6. [Evaluation](#Evaluation)
7. [License](#License)
8. [Citation](#Citation)


## Installation
In an environment with cuda 11.7, run:
```
conda env create -f environment.yml
```
To activate the environment, run:
```
conda activate P2DFlow
```

## Prepare Dataset
* Coming soon!


## Model weights
Download the pretrained checkpoint from [Google Drive](https://drive.google.com/drive/folders/11mdVfMi2rpVn7nNG2mQAGA5sNXCKePZj?usp=sharing) whose filename is `pretrained.ckpt`, and put it into `./weights` folder.


## Training
* Coming soon!


## Inference
To infer for specified protein sequence, firstly modify the input .csv file in `./inference/valid_seq.csv` and config file in `./configs/inference.yaml`, then run:
```
python -u experiments/inference_se3_flows.py
```
And you will get the results in `inference_outputs/weights/`

## Evaluation
* Coming soon!


## License
This project is licensed under the terms of the GPL-3.0 license.

## Citation
* Coming soon!

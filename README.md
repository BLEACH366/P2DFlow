# P2DFlow

P2DFlow is a protein ensemble generative model with SE(3) flow matching based on ESMFold, the ensembles generated by P2DFlow could aid in understanding protein functions across various scenarios.

Technical details and thorough benchmarking results can be found in our paper:
* [P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching](https://arxiv.org/abs/2411.17196)

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
#### (tips: If you want to use the data we have preprocessed, please go directly to `3. Download selected datasets`; if you prefer to process the data from scratch or work with your own data, please start from the beginning)
#### 1. Download raw ATLAS datasets
* Coming soon!
#### 2. Calculate the 'approximate energy' and select representative structures
* Coming soon!
#### 3. Download selected datasets

(i) Download the selected dataset (you can find the way that we select the dataset from ATLAS according to part 2.2 and part 3.1 in our papar) from [Google Drive](https://drive.google.com/drive/folders/1wm5_rMbemxqMiTxoBr_V-Vt5NyNtdZT7?usp=share_link) whose filename is `select_dataset.tar`, and decompress it using:
```
tar -xvf select_dataset.tar
```
(ii) Preprocess `.pdb` files to get `.pkl` files:
```
python ./data/process_pdb_files.py --pdb_dir ${pdb_dir} --write_dir ${write_dir}
```
then compute node representation and pair representation using ESM-2:
```
python ./data/preprocess.py
```
then compute predicted static structure using ESMFold:
```
python ./data/preprocess3.py
```
(iii) Download the `csv` files from [Google Drive](https://drive.google.com/drive/folders/1wm5_rMbemxqMiTxoBr_V-Vt5NyNtdZT7?usp=share_link) whose filenames are `train_dataset.csv` and `train_dataset_energy.csv`(they correspond to `csv_path` and `energy_csv_path` in `./configs/base.yaml` during training), and put them into './traininng'



## Model weights
Download the pretrained checkpoint from [Google Drive](https://drive.google.com/drive/folders/11mdVfMi2rpVn7nNG2mQAGA5sNXCKePZj?usp=sharing) whose filename is `pretrained.ckpt`, and put it into `./weights` folder. You can use the pretrained weight for inference.


## Training
To train P2DFlow, firstly make sure you have prepare the dataset according to `Prepare Dataset` and put it in the right folder, then modify the config file in `./configs/base.yaml`(especially for `csv_path` and `energy_csv_path`). After this, you can run:
```
python experiments/train_se3_flows.py
```
And you will get the checkpoints in `./ckpt`


## Inference
To infer for specified protein sequence, firstly modify the input .csv file in `./inference/valid_seq.csv` and config file in `./configs/inference.yaml`(especially for `validset_path`), then run:
```
python experiments/inference_se3_flows.py
```
And you will get the results in `./inference_outputs/weights/`


## Evaluation
* Coming soon!


## License
This project is licensed under the terms of the GPL-3.0 license.


## Citation
```
@article{jin2024p2dflow,
  title={P2DFlow: A Protein Ensemble Generative Model with SE(3) Flow Matching},
  author={Yaowei Jin, Qi Huang, Ziyang Song, Mingyue Zheng, Dan Teng, Qian Shi},
  journal={arXiv preprint arXiv:2411.17196},
  year={2024}
}
```

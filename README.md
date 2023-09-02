# Cross-Modal De-Deviation for Enhancing Few-Shot Classification (CMDD)

PyTorch implementation for the paper: Cross-Modal De-Deviation for Enhancing Few-Shot Classification

## Dependencies
* python 3.6.5
* numpy 1.16.0
* torch 1.8.0
* tqdm 4.57.0
* scipy 1.5.4
* torchvision 0.9.0

## Overview
Few-shot learning poses a critical challenge due to the deviation problem caused by the scarcity of available samples. In this work, we aim to address deviations in both feature representations and prototypes. To achieve this, we propose a cross-modal de-deviation framework that leverages class semantic information to provide robust prior knowledge for the samples. This framework begins with a visual-to-semantic autoencoder trained on the labeled samples to predict semantic features for the unlabeled samples. Then, we devise a binary linear programming model to incorporate the initial prototypes with the cluster centers of the unlabeled samples. To avoid mismatch between the cluster centers and the initial prototypes, we conduct the label assignment process in the semantic space using the class ground truth semantic features as the reference points, with the cluster centers transformed into semantic representations. Moreover, we model a linear classifier with the concatenation of the refined prototypes and the class ground truth semantic features serving as the initial weights. Then we propose a novel optimization strategy based on the alternating least squares (ALS) model. From the ALS model, we can derive two closed-form solutions regarding to the features and weights, facilitating alternative optimization of them. Extensive experiments conducted on three standard benchmarks demonstrate the competitive advantages of our CMDD method over the state-of-the-art few-shot classification methods, confirming its effectiveness in reducing deviation.

![Image text](https://github.com/pmhDL/CMDD/blob/main/Architecture/Figure1.png)

![Image text](https://github.com/pmhDL/CMDD/blob/main/Architecture/Figure2.png)

## Download the Datasets
* [miniImageNet](https://drive.google.com/file/d/1g4wOa0FpWalffXJMN2IZw0K2TM2uxzbk/view) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
* [glove word embedding](https://nlp.stanford.edu/projects/glove/)

## Running Experiments
If you want to train the models from scratch, please run the run_pre.py first to pretrain the backbone. Then specify the path of the pretrained checkpoints to "./checkpoints/[dataname]"
* Run pretrain phase:
```bash
python run_pre.py
```
* Run few-shot training and test phases:
```bash
python run_cmdd.py
```
## LISENCE
* All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode

* The license gives permission for academic use only.

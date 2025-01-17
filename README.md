# Overparameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis

This repository is used to perform the security evaluation of models with increasing size of model parameters. 


## Installation Guide

Before starting the installation process try to obtain the latest version of the `pip` manager by calling:\
`pip install -U pip`

The Python package setuptools manages the setup process. Be sure to obtain the latest version by calling: \
`pip install -U setuptools`

Once the environment is set up, use following commands to install required packages:\
`pip install -r requirements.txt --no-index --find-links file:///tmp/packages`

## Run experiments

### Arguments
Dataset: MNIST and CIFAR10\
Model type: CNN and FC-RELU for MNIST, Resnet for Cifar10\
Attack type: PGD-L2 and AutoAttac

### Command
`python run.py --ds mnist --model cnn --attack pgdl2`\
or \
`python run.py --ds cifar10 --model resnet --attack pgdl2`\

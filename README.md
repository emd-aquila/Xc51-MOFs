# Fine-tuning UniMOF on WS24

## Project Overview
This project focuses on fine-tuning the UniMOF model on the WS24 dataset. The goal is to enhance the model's performance specifically for predicting Metal-Organic Framework (MOF) water stability.

## Description
- **Model**: UniMOF
- **Dataset**: WS24
- **Purpose**: Fine-tuning the UniMOF model to improve its performance on water stability predictions for MOFs

## Project Structure
```
Xc51-MOFs/
├── README.md
└── WS24-UniMOF/
    ├── images/           # Contains model performance
    ├── infer.ipynb       # Script for model inference
    ├── inference_scripts/# Scripts for model inference and predictions
    ├── preprocess.ipynb  # Script for preprocessing WS24 cifs
    ├── plotting.ipynb    # Script for producing model performance plot
    ├── train.ipynb       # Script for fine-tuning Uni-MOF on WS24
    ├── training_scripts/ # Scripts for model training and fine-tuning
    ├── unimat/           # UniMOF model implementation and utilities
    └── WS24/             # WS24 dataset and related processing scripts
```

## Components
- **images/**: Contains visualizations of training results and evaluation
- **inference_scripts/**: Scripts for running inference with the fine-tuned model
- **training_scripts/**: Implementation of the fine-tuning process and training utilities
- **unimat/**: Core UniMOF model implementation and related utilities
- **WS24/**: Training, validation, and test datasets

## Getting Started
[Instructions for setup and usage will be added as the project develops]

## Dependencies
Please refer to the [Uni-MOF repository](https://github.com/dptech-corp/Uni-MOF/blob/main/README.md) for detailed dependency requirements.

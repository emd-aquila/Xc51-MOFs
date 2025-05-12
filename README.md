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
    ├── images/           # Contains visualization of training results and model evaluation
    ├── infer.ipynb       # Script for model inference
    ├── inference_scripts/# Scripts for model inference with the fine-tuned model
    ├── preprocess.ipynb  # Script for preprocessing WS24 cifs
    ├── plotting.ipynb    # Script for producing model performance plot
    ├── train.ipynb       # Script for fine-tuning Uni-MOF on WS24
    ├── training_scripts/ # Scripts for the fine-tuning process and training utilities
    ├── unimat/           # Core Uni-MOF model implementation and related utilities
    └── WS24/             # WS24 training, validation, and test datasets
```

## Dependencies
Please refer to the [Uni-MOF repository](https://github.com/dptech-corp/Uni-MOF/blob/main) for detailed dependency requirements.

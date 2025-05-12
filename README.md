# Fine-tuning UniMOF on WS24

## Project Overview
This project focuses on fine-tuning the UniMOF model on the WS24 dataset. The goal is to utilize transfer learning to make use of UniMOF's latent structural representations of metal-organic frameworks (MOFs) to predict water stability of high CO2 uptake MOFs.

## Description
- **Model**: [Uni-MOF](https://github.com/dptech-corp/Uni-MOF)
- **Dataset**: [WS24](https://chemrxiv.org/engage/chemrxiv/article-details/6627aaa721291e5d1d7a4c59)
- **Purpose**: Fine-tuning the UniMOF model to improve its water stability predictions accuracy for MOFs.

## Project Structure
```
Xc51-MOFs/
├── README.md
├── Uni-MOF_infer.ipynb               # Script for inference using the original Uni_MOF model
└── MOF_screening/
    └── MOFX_cifs/                    # Contains extracted .cif files for MOFX-DB entries
    ├── download_mofx_cifs.ipynb      # Script to download .cif files from MOFX-DB API
    ├── run_unimof_predictions.ipynb  # Script to predict CO2 adsorption with UniMOF for MOFX-DB .cifs (not used, resorted to CoRE MOF due to time constraints)
└── WS24-UniMOF/
    ├── images/                       # Contains visualization of training results and model evaluation
    ├── infer.ipynb                   # Script for model inference
    ├── inference_scripts/            # Scripts for model inference with the fine-tuned model
    ├── preprocess.ipynb              # Script for preprocessing WS24 cifs
    ├── plotting.ipynb                # Script for producing model performance plot
    ├── train.ipynb                   # Script for fine-tuning Uni-MOF on WS24
    ├── training_scripts/             # Scripts for the fine-tuning process and training utilities
    ├── unimat/                       # Core Uni-MOF model implementation and related utilities
    └── WS24/                         # WS24 training, validation, and test datasets
```

## Dependencies
Please refer to the [Uni-MOF repository](https://github.com/dptech-corp/Uni-MOF/blob/main) for detailed dependency requirements.

## Additional Code and Results
Files and data that were too large or cumbersome to add to GitHub are stored in a separate [DropBox](https://www.dropbox.com/scl/fo/ndp04q6qbwtftfdq54154/APjH7BHUiHyUa0jrx22GPNE?rlkey=6u339qp8iyc4rch1j0mnq5isn&e=2&st=qluyyoa0&dl=0).

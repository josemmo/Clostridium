# Automated C. difficile typing

This repository contains the source code for the *Automated web-based typing of Clostridioides difficile ribotypes via MALDI-TOF MS*. The project implements several machine learning models to distinguish between RT027, RT181, and other ribotypes.

## Installation
To use this repository, you need to have Python 3 installed, clone this repository and install the project dependencies:
```sh
git clone https://github.com/aguerrerolopez/Clostridium
cd Clostridium
pip install -r requirements.txt
```

## Project Structure
This repository contains several scripts to ingest, process, train and predict spectra acquired in raw MALDI-TOF MS format.

1. **Data Preparation**:
   - `prepare_data_exp2.py`: Prepares experimental data from CSV files
   - `outliers.py`: Detects and analyzes outliers in the MALDI-TOF data

2. **Model Implementation**:
   - `models.py`: Contains implementations of various classifiers:
     - Random Forest (RF)
     - Decision Tree (DT)
     - Logistic Regression (LR)
     - FAVAE (Factor Analysis Variational AutoEncoder)
     - KSSHIBA (Kernel Sparse SHIBA)
     - DBLFS (Dual Bayesian Linear Feature Selection)

3. **Training Scripts**:
   - `main_trainer.py`: Main training script
   - `main_evaluate.py`: Evaluation script
   - `final_trainer.py`: Script to train final models on combined data

4. **Prediction Scripts**:
   - `predict.py`: Makes predictions on new data
   - `predict_repro.py`: Specialized prediction for reproducibility study
   - `DBL_predict.py`: Prediction functions for DBL models

5. **Utilities**:
   - `performance_tools.py`: Evaluation metrics and visualization tools

## License
This project is released and distributed under the [MIT license](./LICENSE).

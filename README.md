# Real Estate Price Prediction (Tunisia)

Machine learning project for Tunisian real-estate pricing using regression and classification, with a notebook workflow and a Streamlit inference app.

## Overview

- Dataset size: 2,458 properties
- Goal 1 (regression): predict property price in TND
- Goal 2 (classification): predict price class (Low / High)
- Methodology: CRISP-DM style workflow in `prediction.ipynb`

## Dataset

Main dataset: `dataset_clean.csv`

The project uses property and location features such as:
- Property attributes (`Area`, `room`, `bathroom`, `pieces`)
- Geospatial data (`latt`, `long`, `distance_to_capital`)
- Amenities (`garage`, `pool`, `elevator`, `furnished`, etc.)
- Categorical fields (`location`, `city`, `governorate`, `age`)

## Models and Artifacts

- `regression_model.pkl`: trained price regression model
- `classification_model.pkl`: trained binary classification model
- `preprocessing_artifacts.pkl`: preprocessing metadata (feature columns, target-encoding maps)

## Model Performance

Metrics reported from `prediction.ipynb` test set evaluation:

### Regression

- Random Forest Regressor: `R² = 0.7328`, `MAE = 119,876 TND`
- XGBoost Regressor: `R² = 0.7621`, `MAE = 112,514 TND` (best regression model)

### Classification

- Random Forest Classifier: `Accuracy = 0.8879` (88.79%)
- Gradient Boosting Classifier: `Accuracy = 0.8789` (87.89%)

## Project Files

- `prediction.ipynb`: end-to-end analysis, preprocessing, training, and evaluation
- `streamlit_app.py`: interactive inference interface
- `dataset_clean.csv`: cleaned training/inference dataset
- `catboost_info/`: CatBoost training logs and diagnostics

## Installation

Use your virtual environment (recommended) and install the required packages:

```bash
pip install pandas numpy scikit-learn streamlit jupyter catboost
```

## Run the Project

### 1- Run the notebook

```bash
jupyter notebook prediction.ipynb
```

### 2- Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Requirements

- pandas
- numpy
- scikit-learn
- jupyter
- streamlit
- catboost

## Notes

- Keep `streamlit_app.py`, `dataset_clean.csv`, and all `.pkl` artifact files in the same project folder.
- The app automatically loads category values from `dataset_clean.csv` and preprocessing settings from `preprocessing_artifacts.pkl`.

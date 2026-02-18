# Real Estate Price Prediction - Tunisia

A machine learning project following the CRISP-DM framework to predict real estate prices in Tunisia using regression and classification models.

## Project Overview

This project analyzes 2,458 Tunisian properties and builds predictive models for property prices (TND - Tunisian Dinars).

### Dataset
- **Records**: 2,458 properties
- **Features**: 19 numeric features including:
  - Property characteristics (Area, rooms, bathrooms, pieces)
  - Location data (latitude, longitude, distance to capital)
  - Amenities (pool, garden, elevator, furnished, etc.)
- **Target**: `price_tnd` (Price in Tunisian Dinars)

## Models

### Regression Models
1. **Linear Regression** - R² = 0.2840
2. **Random Forest** - R² = 0.4808 (Best)
3. **Gradient Boosting** - R² = 0.4428

### Classification Models
- Predicts price categories: Low, Medium, High

## Top Features Influencing Price

1. **Area** (58.3%) - Most important
2. **Latitude** (12.4%)
3. **Longitude** (9.3%)
4. **Bathrooms** (4.4%)
5. **Pieces** (3.3%)

## Files

- `prediction.ipynb` - Complete CRISP-DM analysis and model training
- `dataset_clean.csv` - Input dataset
- `*.pkl` - Trained model files

## Usage

Run the Jupyter notebook to see the complete analysis:

```bash
jupyter notebook prediction.ipynb
```

## Requirements

- pandas
- numpy
- scikit-learn
- jupyter

## Installation

```bash
pip install -r requirements.txt
jupyter notebook
```

## Results

The Random Forest regression model achieved the best performance with R² = 0.4808 and MAE = 208,877 TND.

Top 3 features directly influencing price:
- Area (58.3%)
- Latitude (12.4%)
- Longitude (9.3%)

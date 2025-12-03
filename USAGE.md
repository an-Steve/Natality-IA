# Natality-IA: Complete Usage Guide

**Natality-IA** is a machine learning project for analyzing and predicting fertility rates using demographic data from around the world.

*Developed by Anton Nelcon Steve & Cortada Lucas*  
*Master 1 Informatique des Big Data – Université Paris 8*

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Setup](#data-setup)
4. [Usage Guide](#usage-guide)
5. [Project Structure](#project-structure)
6. [Module Documentation](#module-documentation)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

**Natality-IA** analyzes fertility rate trends across countries and time periods, using machine learning to:
- Process and clean demographic data
- Extract meaningful features from time series
- Train predictive models
- Generate insightful visualizations
- Make predictions on future fertility rates

### Key Features

- **Data Processing**: Converts wide-format fertility data into clean, long-format datasets
- **Feature Engineering**: Creates lag features, rolling statistics, and country-specific metrics
- **Model Training**: Trains Linear Regression models with standardization
- **Predictions**: Generates predictions on new/test data
- **Visualizations**: Creates comprehensive plots for data exploration and analysis

---

## 🔧 Installation

### Prerequisites

- Python 3.9.13 (as specified in `pyproject.toml`)
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/an-Steve/Natality-IA.git
cd Natality-IA
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib & seaborn**: Visualization
- **loguru**: Logging
- **typer**: CLI interface
- **tqdm**: Progress bars
- And other dependencies

### Step 4: Verify Installation

```bash
python -c "import ProjetNatalite; print('Installation successful!')"
```

---

## 📊 Data Setup

### Option 1: Use Your Own Data

Place your `fertility_rate.csv` file in the `data/raw/` directory.

**Expected format:**
```csv
Country,1960,1961,1962,...,2023
Afghanistan,7.45,7.45,7.45,...,4.32
Albania,6.58,6.54,6.48,...,1.54
...
```

- First column: Country names
- Subsequent columns: Years with fertility rate values

### Option 2: Download Sample Data

If you have a URL to a fertility rate dataset, create a download script:

```python
# scripts/download_data.py
import pandas as pd
from pathlib import Path

# Example: Download from a public source
url = "YOUR_DATA_URL_HERE"
output_path = Path("data/raw/fertility_rate.csv")

df = pd.read_csv(url)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Data downloaded to {output_path}")
```

---

## 🚀 Usage Guide

### Complete Pipeline (Recommended for First Run)

Run the entire ML pipeline from data processing to predictions:

```bash
python -m ProjetNatalite.main pipeline
```

This executes:
1. Data processing
2. Feature generation
3. Model training
4. Predictions

### Individual Commands

#### 1. Process Raw Data

Transform raw CSV data into clean long-format dataset:

```bash
python -m ProjetNatalite.main process-data
```

**Options:**
```bash
python -m ProjetNatalite.main process-data \
    --input-path data/raw/fertility_rate.csv \
    --output-path data/processed/dataset.csv
```

**Output:** `data/processed/dataset.csv` with columns: `Country`, `Year`, `FertilityRate`

#### 2. Generate Features

Create features for machine learning from processed dataset:

```bash
python -m ProjetNatalite.features
```

**Options:**
```bash
python -m ProjetNatalite.features \
    --input-path data/processed/dataset.csv \
    --features-output-path data/processed/features.csv \
    --labels-output-path data/processed/labels.csv
```

**Features created:**
- Lag features (previous 1, 2, 3 years)
- Rolling statistics (mean, std)
- Change metrics (absolute and percentage)
- Country-specific statistics
- Time normalization

**Output:** 
- `data/processed/features.csv` (14 features)
- `data/processed/labels.csv` (target values)

#### 3. Train Model

Train a Linear Regression model:

```bash
python -m ProjetNatalite.main train-model
```

**Options:**
```bash
python -m ProjetNatalite.main train-model \
    --features-path data/processed/features.csv \
    --labels-path data/processed/labels.csv \
    --model-path models/model.pkl
```

**Output:** 
- Trained model saved as `models/model.pkl`
- Training metrics (R², RMSE, MAE)
- Feature importance ranking

#### 4. Make Predictions

Generate predictions using trained model:

```bash
python -m ProjetNatalite.main predict-model
```

**Options:**
```bash
python -m ProjetNatalite.main predict-model \
    --features-path data/processed/test_features.csv \
    --model-path models/model.pkl \
    --predictions-path data/processed/test_predictions.csv
```

**Output:** `data/processed/test_predictions.csv` with predictions

#### 5. Generate Visualizations

Create plots and visualizations:

```bash
python -m ProjetNatalite.plots
```

**Options:**
```bash
python -m ProjetNatalite.plots \
    --input-path data/processed/dataset.csv \
    --features-path data/processed/features.csv \
    --output-dir reports/figures
```

**Plots generated:**
- `fertility_trends.png` - Top 10 countries' trends
- `global_average.png` - Global average over time
- `distribution.png` - Fertility rate distributions
- `heatmap.png` - Country × Year heatmap
- `correlation_matrix.png` - Feature correlations

---

## 📁 Project Structure

```
Natality-IA/
├── data/
│   ├── raw/                    # Original, immutable data
│   │   └── fertility_rate.csv
│   ├── processed/              # Cleaned and processed data
│   │   ├── dataset.csv
│   │   ├── features.csv
│   │   └── labels.csv
│   ├── interim/                # Intermediate transformations
│   └── external/               # Third-party data
│
├── models/                     # Trained models
│   └── model.pkl
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── ProjetIA_new.ipynb
│
├── ProjetNatalite/            # Source code
│   ├── __init__.py
│   ├── main.py                # CLI entry point
│   ├── config.py              # Configuration and paths
│   ├── dataset.py             # Data processing
│   ├── features.py            # Feature engineering
│   ├── classification.py      # Model classes
│   ├── plots.py               # Visualization functions
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── train.py          # Training pipeline
│   │   └── predict.py        # Inference pipeline
│   │
│   └── Classification Dossier/
│       ├── linear_classifier.py
│       ├── random_forest.py
│       └── K-Means.py
│
├── reports/
│   └── figures/               # Generated plots and figures
│
├── tests/                     # Unit tests
│
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── README.md                 # Project overview
└── USAGE.md                  # This file
```

---

## 📚 Module Documentation

### ProjetNatalite.dataset

**Purpose:** Load and process raw fertility rate data

**Key Functions:**
- `load_and_process_data(input_path)` - Transforms wide to long format
- `main()` - CLI command for data processing

**Transformations:**
- Converts wide format (countries × years) to long format
- Removes missing values
- Sorts by country and year
- Validates data integrity

### ProjetNatalite.features

**Purpose:** Generate features for machine learning

**Key Functions:**
- `create_features(df)` - Creates feature matrix and labels
- `main()` - CLI command for feature generation

**Features Created:**
- **Lag Features:** Previous 1-3 years of fertility rates
- **Rolling Statistics:** 3-year rolling mean and std
- **Change Metrics:** Absolute and percentage changes
- **Country Statistics:** Mean, std, min, max per country
- **Time Features:** Normalized year values
- **Encoding:** Country categorical encoding

### ProjetNatalite.classification

**Purpose:** Machine learning model classes

**Main Class: LinearClassifier**

Methods:
- `__init__(model_path)` - Initialize or load model
- `train(X, y, test_size, random_state, normalize)` - Train model
- `predict(X)` - Make predictions
- `evaluate(X, y, threshold)` - Evaluate performance
- `save_model(path)` - Save to pickle
- `load_model(path)` - Load from pickle
- `get_feature_importance()` - Get feature coefficients

### ProjetNatalite.modeling.train

**Purpose:** Train models and save them

**Key Functions:**
- `main(features_path, labels_path, model_path)` - Training pipeline

**Process:**
1. Load features and labels
2. Split into train/test sets
3. Standardize features
4. Train Linear Regression model
5. Evaluate on test set
6. Save trained model
7. Display metrics and feature importance

### ProjetNatalite.modeling.predict

**Purpose:** Generate predictions with trained models

**Key Functions:**
- `main(features_path, model_path, predictions_path)` - Inference pipeline

**Process:**
1. Load trained model
2. Load test features
3. Apply same preprocessing
4. Generate predictions
5. Save predictions to CSV

### ProjetNatalite.plots

**Purpose:** Create visualizations

**Key Functions:**
- `plot_fertility_trends(df, output_dir, top_n)` - Trend lines
- `plot_global_average(df, output_dir)` - Global statistics
- `plot_distribution(df, output_dir)` - Histograms and boxplots
- `plot_heatmap(df, output_dir, sample_countries)` - Country × Year heatmap
- `plot_correlation_matrix(features_df, output_dir)` - Feature correlations

---

## 💡 Examples

### Example 1: Quick Start

```bash
# 1. Place your data
# Copy fertility_rate.csv to data/raw/

# 2. Run complete pipeline
python -m ProjetNatalite.main pipeline

# 3. Check outputs
# - data/processed/dataset.csv
# - data/processed/features.csv
# - models/model.pkl
# - data/processed/test_predictions.csv
```

### Example 2: Custom Pipeline

```bash
# Process data with custom paths
python -m ProjetNatalite.main process-data \
    --input-path data/raw/my_fertility_data.csv \
    --output-path data/processed/my_dataset.csv

# Generate features
python -m ProjetNatalite.features \
    --input-path data/processed/my_dataset.csv

# Train with custom test size
python -m ProjetNatalite.modeling.train \
    --features-path data/processed/features.csv \
    --labels-path data/processed/labels.csv \
    --model-path models/my_model.pkl
```

### Example 3: Using in Python Scripts

```python
from ProjetNatalite.dataset import load_and_process_data
from ProjetNatalite.features import create_features
from ProjetNatalite.classification import LinearClassifier
from pathlib import Path

# Load and process data
df = load_and_process_data(Path("data/raw/fertility_rate.csv"))

# Create features
X, y = create_features(df)

# Train model
classifier = LinearClassifier()
metrics = classifier.train(X, y, test_size=0.2)
print(f"Test R² Score: {metrics['test_r2']:.4f}")

# Save model
classifier.save_model(Path("models/my_model.pkl"))

# Make predictions
predictions = classifier.predict(X)
```

### Example 4: Generate Only Visualizations

```bash
# Generate all plots
python -m ProjetNatalite.plots \
    --input-path data/processed/dataset.csv \
    --output-dir reports/figures

# Plots will be saved in reports/figures/
# - fertility_trends.png
# - global_average.png
# - distribution.png
# - heatmap.png
# - correlation_matrix.png
```

---

## 🔍 Troubleshooting

### Issue: ModuleNotFoundError

**Problem:** Cannot import ProjetNatalite

**Solution:**
```bash
# Make sure you installed in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)"  # Windows PowerShell
```

### Issue: FileNotFoundError for data

**Problem:** Cannot find `data/raw/fertility_rate.csv`

**Solution:**
1. Ensure file exists: `Test-Path data/raw/fertility_rate.csv`
2. Check file name spelling
3. Verify you're in project root directory
4. Use `--input-path` to specify custom location

### Issue: Model training fails with NaN values

**Problem:** NaN values in features

**Solution:**
- Features automatically remove NaN rows from lag features
- If issue persists, check raw data for missing values
- Ensure sufficient data (need at least 4 years per country for lag features)

### Issue: Import errors for scikit-learn, pandas, etc.

**Problem:** Missing dependencies

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check versions
pip list | grep -E "pandas|numpy|scikit-learn"
```

### Issue: Plots not generating

**Problem:** Matplotlib backend issues on Windows

**Solution:**
```python
# Add to plots.py if needed
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
```

---

## 🎓 Model Performance

Expected performance metrics on fertility rate prediction:

- **R² Score:** 0.95-0.99 (very high predictability for time series with lag features)
- **RMSE:** 0.1-0.3 (low error in fertility rate units)
- **MAE:** 0.05-0.2 (mean absolute error)

Most important features:
1. `FertilityRate_lag1` - Previous year's rate
2. `FertilityRate_rolling_mean_3` - 3-year average
3. `Country_mean_fertility` - Country baseline
4. `FertilityRate_lag2` - Two years ago

---

## 📧 Support & Contact

For questions or issues:

- **Authors:** Anton Nelcon Steve & Cortada Lucas
- **Institution:** Master 1 Informatique des Big Data – Université Paris 8
- **GitHub:** https://github.com/an-Steve/Natality-IA

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Fertility rate data sources (add your source here)
- Université Paris 8 - Master Informatique des Big Data
- Scikit-learn, Pandas, and open-source ML community

---

**Last Updated:** December 2025

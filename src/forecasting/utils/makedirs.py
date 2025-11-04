from forecasting.utils.libraries_others import os

# Create folder for dataset
os.makedirs('../data', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../data/raw', exist_ok=True)

# Create folder for Exploratory Data Analysis
os.makedirs('../eda', exist_ok=True)
os.makedirs('../eda/results', exist_ok=True)
os.makedirs('../eda/results/bivariate_analysis', exist_ok=True)
os.makedirs('../eda/results/univariate_analysis', exist_ok=True)
os.makedirs('../eda/results/univariate_analysis/seasonal_decompose', exist_ok=True)

# Create folder for inference
os.makedirs('../inference/nbeats', exist_ok=True)
os.makedirs('../inference/nbeats/results', exist_ok=True)
os.makedirs('../inference/tft', exist_ok=True)
os.makedirs('../inference/tft/results', exist_ok=True)

# Create folder for tuning
os.makedirs('../tuning/nbeats', exist_ok=True)
os.makedirs('../tuning/nbeats/results', exist_ok=True)
os.makedirs('../tuning/tft', exist_ok=True)
os.makedirs('../tuning/tft/results', exist_ok=True)
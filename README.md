# PREDIKSI KUALITAS UDARA BERBASIS DATA MULTIVARIAT SERTA METEOROLOGIS MENGGUNAKAN TEMPORAL FUSION TRANSFORMER DAN N-BEATS DI KABUPATEN GRESIK

## Directory structure
```
├─ data/                                                        <- All data stored in here.
│  ├─ predicted/                                                <- Results of predicted data from inference model.
│  │  ├─ a1_non-outliers_pre-normalization.csv                  <- Predicted data using input non outliers data and prenormalization.
│  │  ├─ p1_trial-error_non-outliers_pre-normalization.csv      <- Open-source license if one is chosen # TODO : remove this in future
│  │  └─ p1_trial-error_non-outliers.csv                        <- Open-source license if one is chosen # TODO : remove this in future
│  │
│  ├─ processed/                                                <- Results of preprocessed data.
│  │  ├─ correlation_scores.csv                                 <- Correlation results.
│  │  ├─ future_covariates_one_hot_encoding.csv                 <- Future dataset with one hot encoding technic.
│  │  ├─ future_covariates.csv                                  <- Future dataset.
│  │  ├─ past_covariates_nonoutliers_with_pre_normalization.csv <- Past dataset with non outliers and prenormalization.
│  │  ├─ past_covariates_nonoutliers.csv                        <- Past dataset with non outliers.
│  │  └─ past_covariates.csv                                    <- Past covariates.
│  │
│  └─ raw/                                                      <- Raw data.
│     ├─ future_covariates.xlsx                                 <- Raw data of future dataset.
│     ├─ past_covariates_notfix.xlsx                            <- Raw data of past dataset # TODO : remove this in future
│     └─ past_covariates.xlsx                                   <- Raw data of past dataset.
│
├─ models/                                                      <- All models stored in here.
│  ├─ checkpoint_tuning_nbeats/                                 <- N-BEATS tuned model stored in here.
│  │  └─ {model_name}/                                          <- Model name that has been train.
│  │     ├─ checkpoints/                                        <- Saved trained model with .ckpt file.
│  │     │
│  │     ├─ logs/                                               <- Trained model logs.
│  │     │
│  │     └─ _model.pth.tar                                      <- Main model file.
│  │
│  ├─ inference_nbeats/                                         <- Open-source license if one is chosen
│  │
│  └─ trial_and_error/                                          <- N-BEATS and TFT model trial & error.
│     └─ {model_name}/
│        ├─ checkpoints/
│        │
│        ├─ logs/
│        │
│        └─ _model.pth.tar
│
├─ notebooks/                                                   <- All jupyter notebooks stored in here.
│  ├─ exploratory_data_analysis.ipynb                           <- EDA script.
│  ├─ inference.ipynb                                           <- Inference script.
│  └─ trial_error.ipynb                                         <- Trial and error script.
│
├─ reports/                                                     <- All figures results and tuned record stored in here.
│  ├─ figures/                                                  <- All figures results stored in here.
│  │  ├─ bivariate_analysis/                                    <- All figures of bivariate analysis results stored in here.
│  │  │
│  │  └─ univariate_analysis/                                   <- All figures of univariate analysis results stored in here.
│  │     └─ seasonal_decompose/                                 <- All figures of seasonal decompose trand results stored in here.
│  │
│  └─ nbeats_params_results.xlsx                                <- N-BEATS tune recorded.
│
├─ src/                                                         <- All python file stored in here.
│  └─ forecasting/                                              <- For forecasting context.
│     ├─ constants/                                             <- Python file for constants, definition, and enumeration.
│     │  ├─ __init__.py
│     │  ├─ color.py                                            <- Script to initialize used color.
│     │  ├─ columns.py                                          <- Script to initialize columns.
│     │  └─ enums.py                                            <- Script to initialize enumeration.
│     │
│     ├─ models/                                                <- Python file to configure models.
│     │  ├─ __init__.py
│     │  ├─ nbeats_functions.py                                 <- All N-BEATS main functions.
│     │  ├─ nbeats_trial_error.py                               <- Script to try and error.
│     │  ├─ nbeats_tuning.py                                    <- Script for tuning N-BEATS.
│     │  └─ recreate_nbeats_tuning_folder.py                    <- Script for create tuned nbeats model to avoid redundancy of model with same params.
│     │
│     ├─ utils/                                                 <- Python file for utility libraries, variables, and functions.
│     │  ├─ __init__.py
│     │  ├─ data_split.py                                       <- Function to split data.
│     │  ├─ extract_best_epochs.py                              <- Function to extract best epochs.
│     │  ├─ libraries_data_handling.py                          <- All import for data handling libraries.
│     │  ├─ libraries_modelling.py                              <- All improt for data modelling libraries.
│     │  ├─ libraries_others.py                                 <- All import for others libraries.
│     │  ├─ libraries_plotting.py                               <- All import for plotting data libraries.
│     │  ├─ memory.py                                           <- Functions to delete unnecessary variables to save RAM.
│     │  ├─ ordinal_encode.py                                   <- Functions to encode or decode columns.
│     │  ├─ pre_normalizatioin.py                               <- Functions to pre-normalization.
│     │  └─ visualization.py                                    <- Functions to visualize data.
│     │
│     └─ __init__.py
│
├─ README.md                                                    <- Project documentation.
└─ requirements_3012010008.txt                                  <- The requirements library for this project.
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dendraap/undergraduate-thesis.git
cd undergraduate-thesis
```

### 2. Create a Python Virtual Environment

Create a virtual environment to isolate the project dependencies.

#### Windows (Command Prompt)

```cmd
python -m venv contoh_env
```

#### Linux / macOS

```bash
python3 -m venv contoh_env
```

### 3. Activate the Virtual Environment

#### Windows (Command Prompt)

```cmd
contoh_env\Scripts\activate
```

#### Linux / macOS

```bash
source contoh_env/bin/activate
```

After activation, your terminal prompt should look similar to:

```text
(contoh_env) C:\Users\username\undergraduate-thesis>
```

or

```text
(contoh_env) user@hostname:~/undergraduate-thesis$
```

### 4. Upgrade pip (Recommended)

```bash
python -m pip install --upgrade pip
```

### 5. Install the Required Packages

Install all project dependencies using the provided requirements file.

```bash
pip install -r requirements_3012010008.txt
```

### 6. Register the Environment as a Jupyter Kernel (Optional)

If you plan to use Jupyter Notebook or JupyterLab, register the virtual environment as a kernel.

```bash
python -m ipykernel install --user --name=contoh_env --display-name="Python (contoh_env)"
```

### 7. Launch JupyterLab

```bash
jupyter lab
```

### 8. Deactivate the Virtual Environment

When you have finished working on the project:

```bash
deactivate
```
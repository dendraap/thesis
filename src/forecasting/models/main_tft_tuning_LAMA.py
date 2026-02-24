from src.forecasting.utils.libraries_data_handling import pd, np
from src.forecasting.utils.libraries_others import re, os, time, Enum, Optional, gc, pickle, json
from src.forecasting.utils.libraries_plotting import plt
from src.forecasting.utils.libraries_modelling import torch, concatenate, TimeSeries, Scaler, TFTModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, plot_contour, plot_optimization_history, plot_param_importances, QuantileRegression, MeanAbsolutePercentageError, mean_absolute_percentage_error
from src.forecasting.constants.columns import col_decode, col_encode
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import ColumnGroup, PeriodList
from src.forecasting.utils.data_split import dataframe_train_test_split, dataframe_train_test_split, timeseries_train_test_split
from src.forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint
from src.forecasting.models.tft_tuning_w_optuna import tft_tuning_w_optuna
from src.forecasting.models.empty_worst_model import empty_worst_model


def print_callback(study, trial):
    print(f'\nTrial {trial.number} done ✅')
    print(f'Value: {trial.value}')
    print(f'Params: {trial.params}')
    print(f'✅ Best so far: {study.best_value} with: \n{study.best_trial.params}\n')


if __name__ == "__main__":
    # ========================= SET UP ========================= #

    # Initialize internal precision of matrix multiplication
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    # Make dir to store results
    os.makedirs('models/checkpoint_tuning_tft/', exist_ok=True)

    # Setting number after coma to max 5 digits
    np.set_printoptions(suppress=True, precision=5)


    # ========================= LOAD DATASET ========================= #
    # Load xlsx dataset
    df_past   = pd.read_csv('data/processed/past_covariates_nonoutliers.csv')
    df_future = pd.read_csv('data/processed/future_covariates_one_hot_encoding.csv')
    # df_future = pd.read_csv('data/processed/future_covariates.csv')


    # ========================= DATA PREPROCESSING ========================= #
    # Convert timestamp to datatime
    df_past['t'] = pd.to_datetime(df_past['t'], format='%Y-%m-%d %H:%M:%S')

    # Set index
    df_past = df_past.set_index('t').asfreq('h')

    # Convert timestamp to datatime
    df_future['t'] = pd.to_datetime(df_future['t'], format='%Y-%m-%d %H:%M:%S')

    # Set index
    df_future = df_future.set_index('t').asfreq('h')

    # Cut categorical data end time to match with df_past
    df_future = df_future.iloc[:len(df_past)]


    ## ========================= LOAD CORRELATION RESULTS ========================= ##
    # Load correlation results
    results_r = pd.read_csv('data/processed/correlation_scores.csv')

    # Preparing feature selection input
    X_num = df_past[df_past.columns[ColumnGroup.TARGET:]]

    # Take very low correlation level (0.00 - 0.199) to drop
    X_num_drop = results_r[results_r['Correlation'] <= 0.2]['Feature'].to_list()

    # Encode drop colomns name
    X_num_drop = [col_encode[feature] for feature in X_num_drop]

    # Drop columns
    X_num = X_num.drop(columns=X_num_drop)

    ## ========================= DATA SPLIT ========================= ##
    # Split dataset into Y and X
    Y = df_past[df_past.columns[:ColumnGroup.TARGET]].astype('float32')
    X_past = X_num.astype('float32')
    X_future = df_future.astype('float32')

    # Split to data train 90% and test 10%
    Y_train, Y_test           = dataframe_train_test_split(Y, test_size=0.1)
    X_past_train, X_past_test = dataframe_train_test_split(X_past, test_size=0.1)
    X_future_train, X_future_test = dataframe_train_test_split(X_future, test_size=0.1)

    # Change to TimeSeries Dataset
    Y_train        = TimeSeries.from_dataframe(Y_train, value_cols=Y_train.columns.tolist(), freq='h').astype('float32')
    X_past_train   = TimeSeries.from_dataframe(X_past_train, value_cols=X_past_train.columns.tolist(), freq='h').astype('float32')
    X_future_train = TimeSeries.from_dataframe(X_future_train, value_cols=X_future_train.columns.tolist(), freq='h').astype('float32')
    Y_test         = TimeSeries.from_dataframe(Y_test, value_cols=Y_test.columns.tolist(), freq='h').astype('float32')
    X_past_test    = TimeSeries.from_dataframe(X_past_test, value_cols=X_past_test.columns.tolist(), freq='h').astype('float32')
    X_future_test  = TimeSeries.from_dataframe(X_future_test, value_cols=X_future_test.columns.tolist(), freq='h').astype('float32')


    ## ========================= NORMALIZATION ========================= ##
    # Preparing the Scalers
    Y_scaler        = Scaler()
    X_past_scaler   = Scaler()
    X_future_scaler = Scaler()

    # Normalize data
    Y_train_transformed        = Y_scaler.fit_transform(Y_train).astype('float32')
    X_past_train_transformed   = X_past_scaler.fit_transform(X_past_train).astype('float32')
    X_future_train_transformed = X_future_scaler.fit_transform(X_future_train).astype('float32')

    # initialize save_location
    save_path = 'reports/tft_tuned_params.xlsx'
    

    # ========================= DATA MODELLING ========================= #
    # Tuning using Optuna
    study_tft = optuna.create_study(direction='minimize')
    study_tft.optimize(
        lambda trial: tft_tuning_w_optuna(
            Y                = Y_train_transformed,
            X_past           = X_past_train_transformed,
            X_future         = X_future_train_transformed,
            Y_actual         = Y,
            Y_scaler         = Y_scaler,
            pre_normalization= False,
            max_epochs       = 150,
            col_list         = X_num.columns.to_list(),
            col_is_one_hot   = True if len(df_future.columns) > 6 else False,
            custom_checkpoint= True,
            save_path        = save_path,
            trial            = trial
        ), 
        n_trials=4000, 
        callbacks=[print_callback]
    )
from src.forecasting.utils.libraries_data_handling import pd, np
from src.forecasting.utils.libraries_others import re, os, time, Enum, Optional, gc, pickle, json
from src.forecasting.utils.libraries_plotting import plt
from src.forecasting.utils.libraries_modelling import torch, concatenate, TimeSeries, Scaler, TFTModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, plot_contour, plot_optimization_history, plot_param_importances, QuantileRegression, MeanAbsolutePercentageError, mean_absolute_percentage_error
from src.forecasting.constants.columns import col_decode, col_encode
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import ColumnGroup, PeriodList
from src.forecasting.utils.data_split import dataframe_train_valid_test_split
from src.forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint
from src.forecasting.models.tft_tuning_w_optuna import tft_tuning_w_optuna
from src.forecasting.models.empty_worst_model import empty_worst_model
from src.forecasting.utils.scale_timeseries_per_component import scale_Y_timeseries_per_component, scale_X_timeseries_per_component


def get_targets(df, target_cols=None):
    if target_cols is None:
        target_cols = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6']
    available = [col for col in target_cols if col in df.columns]
    return df[available].astype('float32')

def get_features(df, target_cols=None):
    if target_cols is None:
        target_cols = ['y1','y2','y3','y4','y5','y6']
    available_targets = [col for col in target_cols if col in df.columns]
    return df.drop(columns=available_targets).astype('float32')

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
    work_dir = 'models/checkpoint_tuning_tft2/'
    os.makedirs('models/checkpoint_tuning_tft2/', exist_ok=True)

    # Setting number after coma to max 5 digits
    np.set_printoptions(suppress=True, precision=5)


    # ========================= LOAD DATASET ========================= #
    # Load xlsx dataset
    ## ======= CHANGE NUMBER BELOW FOR CHOOSE THE DATASET ======= ##
    dataset_used = 2
    ## ======= CHANGE NUMBER ABOVE FOR CHOOSE THE DATASET ======= ##
    
    df_past      = None
    dataset_type = None
    prenorm_type = None

    df_actual    = pd.read_csv('data/processed/past_covariates_noOutliers_5.csv').drop(
        columns=['x1','x2','x3','x4_zero', 'x4_nonzero','x5','x6','x7_sin', 'x7_cos','x8_zero', 'x8_nonzero'])
    
    if dataset_used == 1:
        dataset_type = 'sqrt'
        prenorm_type = 'sqrt'
        df_past = pd.read_csv('data/processed/past_covariates_sqrt_transform.csv')

    elif dataset_used == 2:
        dataset_type = 'sqrt_NoOzon_5'
        prenorm_type = 'sqrt'
        df_past = pd.read_csv('data/processed/past_covariates_sqrt_transform_NoOzon_5.csv')

        # Drop y6
        df_actual = df_actual.drop(columns=['y6'])

    elif dataset_used == 3:
        dataset_type = 'log1p'
        prenorm_type = 'log1p'
        df_past = pd.read_csv('data/processed/past_covariates_log_transform.csv')

    elif dataset_used == 4:
        dataset_type = 'log1p_NoOzon'
        prenorm_type = 'log1p'
        df_past = pd.read_csv('data/processed/past_covariates_log_transform_NoOzon.csv')

        # Drop y6
        df_actual = df_actual.drop(columns=['y6'])

    elif dataset_used == 5:
        dataset_type = 'log1p_NoOzon_5'
        prenorm_type = 'log1p'
        df_past = pd.read_csv('data/processed/past_covariates_log_transform_NoOzon_5.csv')

        # Drop y6
        df_actual = df_actual.drop(columns=['y6'])

    elif dataset_used == 6:
        dataset_type = 'no_outliers'
        prenorm_type = None
        df_past = pd.read_csv('data/processed/past_covariates_noOutliers.csv')

    elif dataset_used == 7:
        dataset_type = 'no_outliers_noOzon_5'
        prenorm_type = None
        df_past = pd.read_csv('data/processed/past_covariates_noOutliers_NoOzon_5.csv')

        # Drop y6
        df_actual = df_actual.drop(columns=['y6'])
        
    else:
        dataset_type = 'default'
        prenorm_type = None
        df_past = pd.read_csv('data/processed/past_covariates.csv')

    
    df_future = pd.read_csv('data/processed/future_covariates_optimized.csv')


    # ========================= DATA PREPROCESSING ========================= #
    # Convert timestamp to datatime
    df_past['t']   = pd.to_datetime(df_past['t'], format='%Y-%m-%d %H:%M:%S')
    df_future['t'] = pd.to_datetime(df_future['t'], format='%Y-%m-%d %H:%M:%S')
    df_actual['t'] = pd.to_datetime(df_actual['t'], format='%Y-%m-%d %H:%M:%S')

    # Set index
    df_past   = df_past.set_index('t').asfreq('h')
    df_future = df_future.set_index('t').asfreq('h')
    df_actual = df_actual.set_index('t').asfreq('h')

    # Cut categorical data end time to match with df_past
    df_future = df_future.iloc[:len(df_past)]


    ## ========================= LOAD CORRELATION RESULTS ========================= ##
    # Load correlation results
    results_r = pd.read_csv('data/processed/correlation_scores.csv')

    # Take very low correlation level (0.00 - 0.199) to drop
    dropped_covariates = results_r[results_r['Correlation'] <= 0.2]['Feature'].to_list()

    # Encode drop colomns name
    dropped_covariates = [col_encode[feature] for feature in dropped_covariates]

    # Drop covariates columns
    ## ======= CHANGE NUMBER BELOW WHETHER TO DROP OR NOT ======= ##
    drop_cols = True
    ## ======= CHANGE NUMBER ABOVE WHETHER TO DROP OR NOT ======= ##
    
    if drop_cols == True and dataset_used <= 6:
        df_past = df_past.drop(columns=['x4_zero', 'x4_nonzero', 'x5', 'x7_sin', 'x7_cos'])
    elif drop_cols == True and dataset_used > 6:
        df_past = df_past.drop(columns=['x4', 'x5', 'x7'])

    
    ## ========================= DATA SPLIT ========================= ##
    # Split dataset into Y and X
    Y        = get_targets(df_past)
    X_past   = get_features(df_past)
    X_future = df_future.astype('float32')

    # Split to data train and test
    ## ======= CHANGE NUMBER BELOW FOR CHOOSE DATA SPLIT SIZE ======= ##
    valid_size = 0.2
    test_size  = 0.1
    ## ======= CHANGE NUMBER ABOVE FOR CHOOSE DATA SPLIT SIZE ======= ##

    Y_train, Y_valid, Y_test = dataframe_train_valid_test_split(
        Y, valid_size=valid_size, test_size=test_size
    )

    X_past_train, X_past_valid, X_past_test = dataframe_train_valid_test_split(
        X_past, valid_size=valid_size, test_size=test_size
    )

    X_future_train, X_future_valid, X_future_test = dataframe_train_valid_test_split(
        X_future, valid_size=valid_size, test_size=test_size
    )

    # Change to TimeSeries Dataset
    Y_train        = TimeSeries.from_dataframe(Y_train, value_cols=Y_train.columns.tolist(), freq='h').astype('float32')
    X_past_train   = TimeSeries.from_dataframe(X_past_train, value_cols=X_past_train.columns.tolist(), freq='h').astype('float32')
    X_future_train = TimeSeries.from_dataframe(X_future_train, value_cols=X_future_train.columns.tolist(), freq='h').astype('float32')
    Y_valid        = TimeSeries.from_dataframe(Y_valid, value_cols=Y_valid.columns.tolist(), freq='h').astype('float32')
    X_past_valid   = TimeSeries.from_dataframe(X_past_valid, value_cols=X_past_valid.columns.tolist(), freq='h').astype('float32')
    X_future_valid = TimeSeries.from_dataframe(X_future_valid, value_cols=X_future_valid.columns.tolist(), freq='h').astype('float32')
    Y_test         = TimeSeries.from_dataframe(Y_test, value_cols=Y_test.columns.tolist(), freq='h').astype('float32')
    X_past_test    = TimeSeries.from_dataframe(X_past_test, value_cols=X_past_test.columns.tolist(), freq='h').astype('float32')
    X_future_test  = TimeSeries.from_dataframe(X_future_test, value_cols=X_future_test.columns.tolist(), freq='h').astype('float32')


    ## ========================= NORMALIZATION ========================= ##
    # Initialize X Columns to normalize
    x_normalize_cols = None
    if drop_cols == True:
        x_normalize_cols = ['x1', 'x3', 'x6']
    elif drop_cols == False:
        x_normalize_cols = ['x1', 'x3', 'x5', 'x6']
    
    # Initialize Y scalers
    Y_scaler        = Scaler()
    X_past_scaler   = Scaler()

    Y_train_transformed      = Y_scaler.fit_transform(Y_train)
    X_past_train_transformed = X_past_scaler.fit_transform(X_past_train[x_normalize_cols])

    Y_valid_transformed      = Y_scaler.transform(Y_valid)
    X_past_valid_transformed = X_past_scaler.transform(X_past_valid[x_normalize_cols])

    # initialize save_location
    save_path = 'reports/tft_tuned_params_optimized.xlsx'
    

    # ========================= DATA MODELLING ========================= #
    # Tuning using Optuna
    study_nbeats = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, max_resource=150, reduction_factor=3
        )
    )
    study_nbeats.optimize(
        lambda trial: tft_tuning_w_optuna(
            dataset_type     = dataset_type,
            prenorm_type     = prenorm_type,
            Y_train          = Y_train_transformed,
            X_past_train     = X_past_train_transformed,
            X_future_train   = X_future_train,
            Y_valid          = Y_valid_transformed,
            X_past_valid     = X_past_valid_transformed,
            X_future_valid   = X_future_valid,
            Y_scaler         = Y_scaler,
            X_scaler         = X_past_scaler,
            Y_actual         = df_actual.loc[:Y_valid.end_time()],
            validation_split = valid_size,
            max_epochs       = 150,
            Y_col_list       = Y.columns.to_list(),
            X_col_list       = X_past.columns.to_list(),
            custom_checkpoint= True,
            save_path        = save_path,
            work_dir         = work_dir,
            trial            = trial
        ), 
        n_trials=2000, 
        callbacks=[print_callback]
    )
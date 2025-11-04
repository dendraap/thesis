from forecasting.utils.libraries_data_handling import pd, np
from forecasting.utils.libraries_others import re, os, time, Enum, Optional, gc, pickle, json
from forecasting.utils.libraries_plotting import plt
from forecasting.utils.libraries_modelling import torch, TimeSeries, Scaler, NBEATSModel, EarlyStopping, ParameterSampler, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from forecasting.constants.columns import col_decode, col_encode
from forecasting.utils.memory import cleanup
from forecasting.constants.enums import ColumnGroup, PeriodList
from forecasting.utils.data_split import dataframe_train_test_split, dataframe_train_test_split, timeseries_train_test_split
from forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint


# ========================= FUNCTION DEFINITIONS ========================= #
# Initialize Function
def build_fit_nbeats_model(
    Y                : TimeSeries,
    X                : TimeSeries,
    max_epochs       : int,
    batch_size       : int,
    num_stacks       : int,
    num_blocks       : int,
    num_layers       : int,
    layer_widths     : int,
    include_encoders : bool,
    dropout          : float,
    validation_split : float,
    model_name       : str,
    work_dir         : str
) -> tuple[NBEATSModel, Scaler, Scaler]:
    """
        Function to build Fit of N-BEATS Model.

        Args:
            Y (TimeSeries)          : Targeted variables to predict. 
            X (TimeSeries)          : Exogenous variables to predict Y.
            batch_size (int)        : Number of data points before making update. Larger -> more robust but need more Memory
            num_stacks (int)        : Number of stacks in N-BEATS.
            num_blocks (int)        : Number of blocks of each stacks in N-BEATS.
            num_Layers (int)        : Number of fully connected layers of each blocks in N-BEATS
            layer_widths (int)      : Number of neuron patterns. Larger -> need more resource.
            include_encoders (bool) : Optionally, adding some cyclic covariates ex. (hour, dayofweek, week, etc)
            dropout (float)         : Dropout probability to be used in fully connected layers.
            validation_split (float): To split data input into train and validation to monitor val_loss.
            model_name (str)        : The model name to prevent error for same name.
            work_dir (str)          : Path location to save checkpoints best epochs model.

        Returns:
            tuple[NBEATSModel, Scaler, Scaler] : This function return the model fit, Y scaler, and X scaler fit.
    """

    # Initialize Fixed params
    IN_CHUNK     = int(PeriodList.W1)
    OUT_CHUNK    = int(PeriodList.D1)
    RANDOM_STATE = 1502

    # Validation split
    Y_fit, Y_val = timeseries_train_test_split(Y, test_size=validation_split)
    X_fit, X_val = timeseries_train_test_split(X, test_size=validation_split)

    # Make Scaler()
    Y_fit_scaler = Scaler()
    X_fit_scaler = Scaler()

    # Normalize data
    Y_fit_transformed = Y_fit_scaler.fit_transform(Y_fit)
    Y_val_transformed = Y_fit_scaler.fit_transform(Y_val)
    X_fit_transformed = X_fit_scaler.fit_transform(X_fit)
    X_val_transformed = X_fit_scaler.fit_transform(X_val)

    # Initialize EarlyStopping that monitor the validation loss
    early_stopper = EarlyStopping(
        monitor   = 'val_loss',
        patience  = 8,
        min_delta = 0.01,
        mode      = 'min'
    )

    # Detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            'accelerator': 'gpu',
            'devices'    : [0],
            'callbacks'  : [early_stopper],
        }
    else:
        pl_trainer_kwargs = {'callbacks': [early_stopper]}
    
    # Optionally also add cyclically encoded as a past covariate (for additional features)
    encoders = {'cyclic': {'past': ['hour', 'week', 'month']}} if include_encoders else None

    # Build N-BEATS model
    model = NBEATSModel(
        input_chunk_length  = IN_CHUNK,
        output_chunk_length = OUT_CHUNK,
        random_state        = RANDOM_STATE,
        n_epochs            = max_epochs,
        batch_size          = batch_size,
        num_stacks          = num_stacks,
        num_blocks          = num_blocks,
        num_layers          = num_layers,
        layer_widths        = layer_widths,
        pl_trainer_kwargs   = pl_trainer_kwargs,
        add_encoders        = encoders,
        dropout             = dropout,
        save_checkpoints    = True,
        model_name          = model_name,
        work_dir            = work_dir
    )

    # Train the model
    model.fit(
        series              = Y_fit_transformed,
        val_series          = Y_val_transformed,
        past_covariates     = X_fit_transformed,
        val_past_covariates = X_val_transformed
    )

    # reload best model over course of training
    model = NBEATSModel.load_from_checkpoint(
        model_name = model_name,
        work_dir   = work_dir
    )

    # Clean up memory
    cleanup(
        Y_fit, Y_val, X_fit, X_val, 
        Y_fit_transformed, X_fit_transformed, 
        Y_val_transformed, X_val_transformed
    )

    return model, Y_fit_scaler, X_fit_scaler

def evaluate_cv(
    forecasts  : list[TimeSeries],
    scaler     : Scaler,
    df_actual  : pd.DataFrame,
) -> float:
    """
    Evaluating rolling forecast results using MAPE.
    Args:
        forecasts (list[TimeSeries]) : List of forecasted TimeSeries from historical_forecast().
        scaler (Scaler)              : Scaler used to inverse transform forecasted values.
        df_actual (pd.DataFrame)     : Dataset actual for comparison.

    Returns:
        float: Overall average MAPE across all components and folds.

    """

    # Initialize MAPE results for each components
    mape_results = []
    components   = forecasts[0].components
    num_folds    = len(forecasts)

    # Iterate through each components
    for i, comp in enumerate(components):

        # Iterate through each rolling forecast folds
        for j in range(num_folds):

            # Inverse transform forecast results
            forecast_unscaled = scaler.inverse_transform(forecasts[j])

            # Set predicted component
            ts          = forecast_unscaled[comp]
            pred_series = pd.Series(ts.values().flatten(), index=ts.time_index)

            # Set start & end forecast time
            start = pred_series.index.min()
            end   = pred_series.index.max()

            # Get actual series based on start - end predict time
            actual_series = df_actual.loc[start:end, comp]

            # Calculate MAPE
            mape = mean_absolute_percentage_error(actual_series, pred_series)
            mape_results[comp].append(mape)

            # Clean up memory
            cleanup(forecast_unscaled, ts, pred_series, actual_series)
        
    # Clean up memory
    cleanup(mape_results, components, num_folds)
    
    return np.mean(mape_results)


def tuning(
    Y                : TimeSeries,
    X                : TimeSeries,
    Y_actual         : pd.DataFrame,
    max_epochs       : int,
    batch_size       : int,
    params_grid      : dict[str, np.ndarray],
    n_iter           : int,
    include_encoders : bool,
    save_path        : str | None = None
) -> pd.DataFrame:
    """
    Function hyperparameter tuning for N-BEATS using random search (parameter sampler) and rolling forecast evaluation.

    Args:
        Y (TimeSeries)                      : Target series.
        X (TimeSeries)                      : Past Covariates.
        Y_actual (pd.DataFrame)             : Actual targeted data to compare.
        max_epochs (int)                    : Max training epochs.
        batch_size (int)                    : Number of batch size for each epochs iteration.
        params_grid (dict[str, np.ndarray]) : List of hyperparameter sample form.
        n_iter (int)                        : Number of random hyperparameter sample form to evaluate.
        include_encoders (bool)             : Whether to inluce add_encoders or not.
        save_path (str)                     : Path location to save tuning results as xlsx or not.

    Returns:
        pd.DataFrame : This function returns tuning results contain parameters and MAPE.
    """

    # Initialize results
    results = []

    # Initialize parameter to evaluate using random search
    params_list = list(ParameterSampler(params_grid, n_iter=n_iter, random_state=1502))

    tuning_start = time.time()

    # Iterate through each parameter take
    for params in params_list:
        print(f'\nTuning with params: {params}')

        # Generate model name and work dir
        model_name = (
            f'nbeats_st{params['num_stacks']}__ly{params['num_layers']}_bl{params['num_blocks']}'
            f'_wd{params['layer_widths']}_dp{params['dropout']}'
        )
        work_dir = '../models/checkpoint_tuning_nbeats/'

        # Fit model
        start_time = time.time()

        model, Y_fit_scaler, X_fit_scaler = build_fit_nbeats_model(
            Y                = Y,
            X                = X,
            max_epochs       = max_epochs,
            batch_size       = batch_size,
            num_stacks       = params['num_stacks'],
            num_blocks       = params['num_blocks'],
            num_layers       = params['num_layers'],
            layer_widths     = params['layer_widths'],
            include_encoders = include_encoders,
            dropout          = params['dropout'],
            validation_split = 0.2,
            model_name       = model_name,
            work_dir         = work_dir
        )

        cost_time = time.time() - start_time

        print(f'N-BEATS Fit cost: {cost_time:.2f} seconds')

        # Cross Validation with Rolling Forecast
        forecast_horizon = 24
        cv_test = model.historical_forecasts(
            series           = Y,
            past_covariates  = X,
            start            = Y[-(7 * forecast_horizon)].start_time(),
            forecast_horizon = forecast_horizon,
            stride           = forecast_horizon,
            retrain          = False,
            last_points_only = False,
        )

        # Evaluate
        mape_cv = evaluate_cv(
            forecasts  = cv_test,
            scaler     = Y_fit_scaler,
            df_actual  = Y_actual,
        )

        # initialize fixed params (used in build_fit_nbeats_model())to store in results
        fixed_params = {
            'input_chunk_length'  : int(PeriodList.W1),
            'output_chunk_length' : int(PeriodList.D1),
            'random_state'        : 1502,
            'n_epochs'            : extract_best_epoch_from_checkpoint(work_dir, model_name),
            'batch_size'          : batch_size,
            'fit_cost'            : cost_time
        }

        # EarlyStopping config to store in results
        early_stopping_config = {
            'monitor'  : 'val_loss',
            'patience' : 8,
            'min_delta': 0.01,
            'mode'     : 'min'
        }

        # Trainer config to store in results
        pl_trainer_kwargs = {
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices'    : [0] if torch.cuda.is_available() else None,
            'callbacks'  : early_stopping_config
        }

        # Encoder config to store in results
        encoders = {'cyclic': {'past': ['hour', 'week', 'month']}} if include_encoders else None
        
        # Append result
        results.append({
            'MAPE'           : mape_cv,
            **params,
            **fixed_params,
            'early_stopping' : json.dumps(early_stopping_config),
            'trainer_config' : json.dumps(pl_trainer_kwargs),
            'encoder_config' : json.dumps(encoders) if encoders else None
        })

        # Clean up memory
        cleanup(model, Y_fit_scaler, X_fit_scaler, cv_test)

    tuning_cost = time.time() - tuning_start
    print(f'N-BEATS Tuning cost: {tuning_cost:.2f} seconds')
    
    # Change results to dataframe
    df_results = pd.DataFrame(results).sort_values(by='MAPE').reset_index(drop=True)

    if save_path:
        df_results.to_excel(save_path, index=False)

    return df_results


# ========================= SET UP ========================= #
# Make dir to store results
os.makedirs('../models/checkpoint_tuning_nbeats/')

# Setting number after coma to max 5 digits
np.set_printoptions(suppress=True, precision=5)


# ========================= LOAD DATASET ========================= #
# Load xlsx dataset
df_past     = pd.read_csv('../data/processed/past_covariates.csv')
df_category = pd.read_csv('../data/processed/future_covariates.csv')


# ========================= DATA PREPROCESSING ========================= #
# Convert timestamp to datatime
df_past['t'] = pd.to_datetime(df_past['t'], format='%Y-%m-%d %H:%M:%S')

# Set index
df_past = df_past.set_index('t').asfreq('h')

# Convert timestamp to datatime
df_category['t'] = pd.to_datetime(df_category['t'], format='%Y-%m-%d %H:%M:%S')

# Set index
df_category = df_category.set_index('t').asfreq('h')

# Cut categorical data end time to match with df_past
df_category = df_category.iloc[:len(df_past)]


## ========================= LOAD CORRELATION RESULTS ========================= ##
# Load correlation results
results_r = pd.read_csv('../data/processed/correlation_scores.csv')

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
Y = df_past[df_past.columns[:ColumnGroup.TARGET]]
X = pd.concat([X_num, df_category], axis=1)

# Split to data train 80% and test 20%
Y_train, Y_test = dataframe_train_test_split(Y, test_size=0.1)
X_train, X_test = dataframe_train_test_split(X, test_size=0.1)

# Change to TimeSeries Dataset
Y_train = TimeSeries.from_dataframe(Y_train, value_cols=Y_train.columns.tolist(), freq='h')
X_train = TimeSeries.from_dataframe(X_train, value_cols=X_train.columns.tolist(), freq='h')
Y_test  = TimeSeries.from_dataframe(Y_test, value_cols=Y_test.columns.tolist(), freq='h')
X_test  = TimeSeries.from_dataframe(X_test, value_cols=X_test.columns.tolist(), freq='h')

# Change unsplitted feature for inference
Y_series = TimeSeries.from_dataframe(Y, value_cols=Y.columns.tolist(), freq='h')
X_series = TimeSeries.from_dataframe(X, value_cols=X.columns.tolist(), freq='h')


## ========================= NORMALIZATION ========================= ##
# Preparing the Scalers
Y_scaler = Scaler()
X_scaler = Scaler()

# Normalize data
Y_train_transformed  = Y_scaler.fit_transform(Y_train)
Y_test_transformed   = Y_scaler.fit_transform(Y_test)

X_train_transformed  = X_scaler.fit_transform(X_train)
X_test_transformed   = X_scaler.fit_transform(X_test)

# Normalize data for inference
Y_series_transformed = Y_scaler.fit_transform(Y_series)
X_series_transformed = X_scaler.fit_transform(X_series)


# ========================= DATA MODELLING ========================= #
# Initialize parameter grid possibilites
params_grid = {
    'num_stacks'   : [10, 20, 30],
    'num_blocks'   : [1, 2, 4],
    'num_layers'   : [2, 4, 6],
    'layer_widths' : [256, 512],
    'dropout'      : [0.1, 0.2, 0.3]
}

# Make a tuning
tuning_results = tuning(
    Y                = Y_train_transformed,
    X                = X_train_transformed,
    Y_actual         = Y,
    max_epochs       = 50,
    batch_size       = 32,
    params_grid      = params_grid,
    n_iter           = 5,
    include_encoders = False,
    save_path = '../reports/nbeats_tuning_results.xlsx'
)
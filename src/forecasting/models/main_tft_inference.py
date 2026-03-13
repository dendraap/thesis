from src.forecasting.utils.libraries_data_handling import pd, np
from src.forecasting.utils.libraries_others import re, os, time, Enum, Optional, gc, pickle, json, ast
from src.forecasting.utils.libraries_plotting import plt
from src.forecasting.utils.libraries_modelling import torch, concatenate, TimeSeries, Scaler, TFTModel, plot_contour, plot_optimization_history, plot_param_importances, QuantileRegression
from src.forecasting.constants.columns import col_decode, col_encode
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import ColumnGroup, PeriodList
from src.forecasting.utils.data_split import dataframe_train_valid_test_split
from src.forecasting.models.tft_build import tft_build
from src.forecasting.models.evaluate_cv_timeseries import evaluate_cv_timeseries
from src.forecasting.models.evaluate_test_timeseries import evaluate_test_timeseries
from src.forecasting.models.tft_store_to_excel import tft_inference_store_to_excel
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

def get_dataset_types(save_path):
    df = pd.read_excel(save_path)
    df = df[df['status'] == 'SUCCESS']
    return df['dataset_type'].unique().tolist()

def load_top_models_per_dataset(save_path, dataset_type, top_n):
    df = pd.read_excel(save_path)
    df = df[(df['dataset_type'] == dataset_type) & (df['status'] == 'SUCCESS')].sort_values('MAPE_sum')
    return df.head(top_n)

def load_dataset_for_inference(
    dataset_type : str,
    valid_size   : float,
    drop_cols    : bool,
    test_size    : float = 0.1,
):
    # ========================= LOAD DATA ========================= #
    # Load actual data
    df_actual = pd.read_csv('data/processed/past_covariates_noOutliers_5.csv').drop(
        columns=['x1','x2','x3','x4_zero', 'x4_nonzero','x5','x6','x7_sin', 'x7_cos','x8_zero', 'x8_nonzero'])

    # Load past dataset
    if dataset_type == 'sqrt_NoOzon_5':
        prenorm_type = 'sqrt'
        df_past = pd.read_csv(
            'data/processed/past_covariates_sqrt_transform_NoOzon_5.csv'
        )
        df_actual = df_actual.drop(columns=['y6'])

    elif dataset_type == 'log1p_NoOzon_5':
        prenorm_type = 'log1p'
        df_past = pd.read_csv(
            'data/processed/past_covariates_log_transform_NoOzon_5.csv'
        )
        df_actual = df_actual.drop(columns=['y6'])

    elif dataset_type == 'no_outliers_noOzon_5':
        prenorm_type = None
        df_past = pd.read_csv(
            'data/processed/past_covariates_noOutliers_NoOzon_5.csv'
        )
        df_actual = df_actual.drop(columns=['y6'])

    else:
        raise ValueError(f'Unknown dataset_type: {dataset_type}')

    # Load future dataset
    df_future = pd.read_csv('data/processed/future_covariates_optimized.csv')

    # ========================= DATA PREPROCESSING ========================= #
    # Set index
    for name, df in zip(
        ['past','future','actual'],
        [df_past, df_future, df_actual]
    ):
        df['t'] = pd.to_datetime(df['t'])
        df.set_index('t', inplace=True)
        df.sort_index(inplace=True)
        df.asfreq('h')
    
    # Clip future dataset
    df_future = df_future.iloc[:len(df_past)]

    ## ========================= LOAD CORRELATION RESULTS ========================= ##
    # Load correlation results
    results_r = pd.read_csv('data/processed/correlation_scores.csv')

    # Take very low correlation level (0.00 - 0.199) to drop
    dropped_covariates = results_r[results_r['Correlation'] <= 0.2]['Feature'].to_list()

    # Encode drop colomns name
    dropped_covariates = [col_encode[feature] for feature in dropped_covariates]

    # Drop covariates columns
    if drop_cols == True:
        df_past = df_past.drop(columns=['x4_zero', 'x4_nonzero', 'x5', 'x7_sin', 'x7_cos'])

    ## ========================= DATA SPLIT ========================= ##
    # Split dataset into Y and X
    Y        = get_targets(df_past)
    X_past   = get_features(df_past)
    X_future = df_future.astype('float32')

    # Split to data train and test
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
    def to_ts(df):
        return TimeSeries.from_dataframe(
            df, value_cols=df.columns.tolist(), freq='h'
        ).astype('float32')

    Y_train_ts = to_ts(Y_train)
    Y_valid_ts = to_ts(Y_valid)
    Y_test_ts  = to_ts(Y_test)

    X_past_train_ts = to_ts(X_past_train)
    X_past_valid_ts = to_ts(X_past_valid)
    X_past_test_ts  = to_ts(X_past_test)

    X_future_train_ts = to_ts(X_future_train)
    X_future_valid_ts = to_ts(X_future_valid)
    X_future_test_ts  = to_ts(X_future_test)

    # For X_future > data test
    future_start     = Y_test_ts.end_time() + pd.Timedelta(hours=1)
    X_future_more_ts = X_future.loc[future_start:]
    X_future_more_ts = to_ts(X_future_more_ts)

    ## ========================= NORMALIZATION ========================= ##
    # Initialize X Columns to normalize
    x_normalize_cols = None
    if drop_cols == True:
        x_normalize_cols = ['x1', 'x3', 'x6']
    elif drop_cols == False:
        x_normalize_cols = ['x1', 'x3', 'x5', 'x6']

    # Initialize Y and X scalers
    Y_scaler      = Scaler()
    X_past_scaler = Scaler()

    Y_train_scaled = Y_scaler.fit_transform(Y_train_ts)
    Y_valid_scaled = Y_scaler.transform(Y_valid_ts)
    Y_test_scaled  = Y_scaler.transform(Y_test_ts)

    X_past_train_scaled = X_past_scaler.fit_transform(X_past_train_ts[x_normalize_cols])
    X_past_valid_scaled = X_past_scaler.transform(X_past_valid_ts[x_normalize_cols])
    X_past_test_scaled  = X_past_scaler.transform(X_past_test_ts[x_normalize_cols])

    ## ========================= MERGE TRAIN + VALID ========================= ##
    Y_train_full        = Y_train_scaled.append(Y_valid_scaled)
    X_past_train_full   = X_past_train_scaled.append(X_past_valid_scaled)
    X_future_train_full = X_future_train_ts.append(X_future_valid_ts)

    return (
        df_actual,
        Y_train_full,
        X_past_train_full,
        X_future_train_full,
        Y_train_scaled,
        Y_valid_scaled,
        Y_test_scaled,
        X_past_train_scaled,
        X_past_valid_scaled,
        X_past_test_scaled,
        X_future_train_ts,
        X_future_valid_ts,
        X_future_test_ts,
        X_future_more_ts,
        Y_scaler,
        X_past_scaler,
        prenorm_type
    )

### ========================================================== ###
###                            MAIN                            ###
### ========================================================== ###
if __name__ == "__main__":
    
    # ========================= SET UP ========================= #
    GPU = False
    # Initialize internal precision of matrix multiplication
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(42)
        GPU = True
    
    # Make dir to store results
    work_dir  = 'models/checkpoint_inference_tft/'
    os.makedirs(work_dir, exist_ok=True)

    # Setting number after coma to max 5 digits
    np.set_printoptions(suppress=True, precision=5)

    # ========================= LOAD TUNED RESULTS ========================= #
    save_path     = 'reports/tft_inference_results.xlsx'
    tuned_path    = 'reports/tft_tuned_params_optimized.xlsx'
    dataset_types = get_dataset_types(tuned_path)

    # Check if save_path excel file is exist
    if save_path and os.path.exists(save_path):
        print(f'ℹ️ Excel already exists at {save_path}, skipping creation.\n')
        
    else:
        # Create new excel file
        columns = [
            'timestamp', 'valid_MAPE_sum', 'MAE_sum', 'MAPE_sum', 'MSE_sum', 'RMSE_sum',
            'dataset_type', 'n_predict', 'model_name', 'GPU', 'fit_cost_seconds', 'predict_cost_seconds',
            'valid_MAPE_y1','valid_MAPE_y2', 'valid_MAPE_y3', 'valid_MAPE_y4', 'valid_MAPE_y5','valid_MAPE_y6', 
            'MAE_y1','MAE_y2', 'MAE_y3', 'MAE_y4', 'MAE_y5','MAE_y6', 
            'MAPE_y1','MAPE_y2', 'MAPE_y3', 'MAPE_y4', 'MAPE_y5','MAPE_y6', 
            'MSE_y1','MSE_y2', 'MSE_y3', 'MSE_y4', 'MSE_y5','MSE_y6', 
            'RMSE_y1','RMSE_y2', 'RMSE_y3', 'RMSE_y4', 'RMSE_y5','RMSE_y6', 
            'input_chunk_length', 'output_chunk_length', 'n_epochs', 'batch_size', 'hidden_size', 
            'lstm_layers', 'num_attention_heads', 'dropout', 'lr', 'random_state', 'validation_split',
            'stride', 'Y_col_list', 'X_col_list', 'add_encoders', 'work_dir' 
        ]
        df_empty = pd.DataFrame(columns=columns)
        df_empty.to_excel(save_path, index=False)
        print(f'✅ Empty Excel file created with headers at {save_path}')


        # Clean up memory
        cleanup(df_empty)

    # Initialize excel save path
    existing_df = pd.read_excel(save_path)

    # Iterate though each dataset type
    for dataset_type in dataset_types:
        print('#######################################################')
        print(f'🚀 Processing dataset: {dataset_type}')
        print('#######################################################')

        # Get top_n best models
        top_models = load_top_models_per_dataset(
            tuned_path,
            dataset_type,
            top_n=20
        )

        # Iterate though each excel row
        for _, row in top_models.iterrows():

            # Get set up
            model_name          = str(row['model_name'])
            input_chunk_length  = int(row['input_chunk_length'])
            output_chunk_length = int(row['output_chunk_length'])
            n_epochs            = int(row['n_epochs'])
            batch_size          = int(row['batch_size'])
            hidden_size         = int(row['hidden_size'])
            lstm_layers         = int(row['lstm_layers']) 
            num_attention_heads = int(row['num_attention_heads'])
            dropout             = float(row['dropout'])
            lr                  = float(row['lr'])
            random_state        = int(row['random_state'])
            # validation_split    = float(row['validation_split'])
            validation_split    = 0.3
            stride              = int(row['stride'])
            Y_col_list          = ast.literal_eval(row['Y_col_list'])
            X_col_list          = ast.literal_eval(row['X_col_list'])
            add_encoders        = json.loads(row['add_encoders']) if pd.notna(row['add_encoders']) else None

            print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
            print('↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓\n')
            print(
                f'🔃 Retrain TFT Model with:\n'
                f'\t Model name: {model_name}\n'
                f'\t Epochs: {n_epochs}\n'
                f'\t Validation size: {validation_split}\n'
            )
            
            
            # Load dataset
            (
                df_actual,
                Y_train_full,
                X_past_train_full,
                X_future_train_full,
                Y_train_scaled,
                Y_valid_scaled,
                Y_test_scaled,
                X_past_train_scaled,
                X_past_valid_scaled,
                X_past_test_scaled,
                X_future_train_ts,
                X_future_valid_ts,
                X_future_test_ts,
                X_future_more_ts,
                Y_scaler,
                X_past_scaler,
                prenorm_type
            ) = load_dataset_for_inference(
                    dataset_type = dataset_type,
                    valid_size   = validation_split,
                    # drop_cols    = True,
                    drop_cols    = False,
                )
            
            # ========================= RETRAIN MODEL ========================= #
            fit_start = time.time()
            model = tft_build(
                Y_train             = Y_train_scaled,
                X_past_train        = X_past_train_scaled,
                X_future_train      = X_future_train_ts,
                Y_valid             = Y_valid_scaled,
                X_past_valid        = X_past_valid_scaled,
                X_future_valid      = X_future_valid_ts,
                input_chunk_length  = input_chunk_length,
                output_chunk_length = output_chunk_length,
                n_epochs            = n_epochs,
                batch_size          = batch_size,
                hidden_size         = hidden_size,
                lstm_layers         = lstm_layers,
                num_attention_heads = num_attention_heads,
                dropout             = dropout,
                add_encoders        = add_encoders,
                model_name          = model_name,
                work_dir            = work_dir,
                lr                  = lr,
                random_state        = random_state
            )
            fit_cost_seconds = time.time() - fit_start
            
            # Cross Validation with Rolling Forecast
            cv_test = model.historical_forecasts(
                series            = Y_train_scaled.append(Y_valid_scaled),
                past_covariates   = X_past_train_scaled.append(X_past_valid_scaled),
                future_covariates = X_future_train_ts.append(X_future_valid_ts),
                start             = Y_train_scaled.get_timestamp_at_point(input_chunk_length),
                forecast_horizon  = output_chunk_length,
                stride            = output_chunk_length,
                retrain           = False,
                last_points_only  = False,
            )
        
            # Validation Evaluate
            mape_cv = evaluate_cv_timeseries(
                forecasts    = cv_test,
                scaler       = Y_scaler,
                df_actual    = df_actual,
                prenorm_type = prenorm_type
            )
                
            # Save MAPE validation results
            valid_MAPE_sum     = sum(mape_cv.values())
        
            print(f'\n💹 MAPE_sum : {valid_MAPE_sum}')
            print(f'🧠 MAPE CV: {mape_cv}\n')

            # ========================= INFERENCE ========================= #
            # Forecast the future
            # n_predict       = len(Y_test_scaled) # Predict entire data test
            n_predict       = 12 # Predict only 12 step ahead on data test
            series_for_pred = Y_train_full
            past_for_pred   = X_past_train_full.append(X_past_test_scaled)
            future_for_pred = X_future_train_full.append(X_future_test_ts)
            
            predict_start = time.time()
            forecasts = model.predict(
                n                 = n_predict,
                series            = series_for_pred,
                past_covariates   = past_for_pred,
                future_covariates = future_for_pred,
                verbose           = True,
                n_jobs            = 1,
                random_state      = random_state,
            )
            predict_cost_seconds = time.time() - predict_start
            
            # Evaluate
            mse_test, rmse_test, mae_test, mape_test = evaluate_test_timeseries(
                forecasts    = forecasts,
                scaler       = Y_scaler,
                df_actual    = df_actual,
                prenorm_type = prenorm_type
            )
            
            # Save evaluation results
            MAE_sum  = sum(mae_test.values())
            MAPE_sum = sum(mape_test.values())
            MSE_sum  = sum(mse_test.values())
            RMSE_sum = sum(rmse_test.values())
            
            valid_mape_results = {**{f'valid_MAPE_{k}': v for k, v in mape_cv.items()}}
            mae_results        = {**{f'MAE_{k}': v for k, v in mae_test.items()}}
            mape_results       = {**{f'MAPE_{k}': v for k, v in mape_test.items()}}
            mse_results        = {**{f'MSE_{k}': v for k, v in mse_test.items()}}
            rmse_results       = {**{f'RMSE_{k}': v for k, v in rmse_test.items()}}

            print(f'\n💹 MAE_sum : {MAE_sum}')
            print(f'🧠 MAPE test: {mae_test}\n')
            print(f'\n💹 MAPE_sum : {MAPE_sum}')
            print(f'🧠 MAPE test: {mape_test}\n')
            print(f'\n💹 MSE_sum : {MSE_sum}')
            print(f'🧠 MSE test: {mse_test}\n')
            print(f'\n💹 RMSE_sum : {RMSE_sum}')
            print(f'🧠 RMSE test: {rmse_test}\n')

            # ========================= SAVE TO EXCEL ========================= #
            tft_inference_store_to_excel(
                valid_MAPE_sum      = valid_MAPE_sum,
                MAE_sum             = MAE_sum,
                MAPE_sum            = MAPE_sum,
                MSE_sum             = MSE_sum,
                RMSE_sum            = RMSE_sum,
                dataset_type        = dataset_type,
                n_predict           = n_predict,
                model_name          = model_name,
                GPU                 = GPU,
                fit_cost_seconds    = fit_cost_seconds,
                predict_cost_seconds= predict_cost_seconds,
                valid_mape_results  = valid_mape_results,
                mae_results         = mae_results,
                mape_results        = mape_results,
                mse_results         = mse_results,
                rmse_results        = rmse_results,
                input_chunk_length  = input_chunk_length,
                output_chunk_length = output_chunk_length,
                n_epochs            = n_epochs,
                batch_size          = batch_size,
                hidden_size         = hidden_size,
                lstm_layers         = lstm_layers,
                num_attention_heads = num_attention_heads,
                dropout             = dropout,
                lr                  = lr,
                random_state        = random_state,
                validation_split    = validation_split,
                stride              = stride,
                Y_col_list          = Y_col_list,
                X_col_list          = X_col_list,
                add_encoders        = add_encoders,
                work_dir            = work_dir,
                existing_df         = existing_df,
                save_path           = save_path,   
            )

            existing_df = pd.read_excel(save_path)

            # Clean up disk
            empty_worst_model(
                work_dir   = work_dir,
                excel_path = save_path,
                print_all  = False,
                patience   = 0.0
            )

            print('\n↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
            print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
        # Clean up memory
        cleanup(model, cv_test, mape_cv, forecasts, mae_test, mape_test, mse_test, rmse_test, existing_df)
from src.forecasting.utils.libraries_data_handling import np, pd, math
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import concatenate, TimeSeries, Scaler, mape, mae, rmse, mse
from src.forecasting.utils.memory import cleanup

def evaluate_test_timeseries(
    forecasts    : TimeSeries,
    scaler       : Scaler,
    df_actual    : pd.DataFrame,
    prenorm_type : str | None = None
) -> dict[str, float]:
    """
    Evaluating combined forecast results using MAPE per component.
    Args:
        forecasts (list[TimeSeries]) : List of forecasted TimeSeries from historical_forecast().
        scaler (Scaler)              : Y Scaler to inverse.
        df_actual (pd.DataFrame)     : Dataset actual for comparison.
        pre_norm (str | None = None) : Type prenormalization used to inverse transform.

    Returns:
        dict[str, float]: Dictionary of MAPE scores per component.
    """

    # Inverse scaling & prenorm transform per target
    pred = scaler.inverse_transform(forecasts)
    
    if prenorm_type is None:
        pass
    elif prenorm_type == 'sqrt':
        pred = pred ** 2
    elif prenorm_type == 'log1p':
        pred = pred.map(np.expm1)
 
    # Extact actual and prediction
    start  = pred.start_time()
    end    = pred.end_time()
    actual = df_actual.loc[start:end]

    actual = TimeSeries.from_dataframe(
        actual, value_cols=actual.columns.tolist(), freq='h'
    ).astype('float32')

    # Evaluate per variables
    mape_results = {}
    mse_results = {}
    mae_results = {}
    rmse_results = {}
    for col in pred.components:

        # Avoid NaN results
        try:
            mse_val  = mse(actual[col], pred[col])
            rmse_val = rmse(actual[col], pred[col])
            mae_val  = mae(actual[col], pred[col])
            mape_val = mape(actual[col], pred[col])
            
            if isinstance(mape_val, float) and math.isnan(mape_val):
                print('!! MAPE is NAN. Change to 9999')
                mape_results[col] = 9999
            else:
                mse_results[col]  = mse_val
                rmse_results[col] = rmse_val
                mae_results[col]  = mae_val
                mape_results[col] = mape_val
                
        except Exception as e:
            print(f'!! {e} MAPE is NAN. Change to 9999')
            mse_results[col] = 9999
            rmse_results[col] = 9999
            mae_results[col] = 9999
            mape_results[col] = 9999
            
    cleanup(pred, actual)
    return  mse_results, rmse_results, mae_results, mape_results
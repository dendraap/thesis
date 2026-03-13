from src.forecasting.utils.libraries_data_handling import pd
from src.forecasting.utils.libraries_others import os, json, datetime
from src.forecasting.utils.memory import cleanup


def nbeats_store_to_excel(
    model_name          : str,
    work_dir            : str,
    GPU                 : bool,
    dataset_type        : str,
    input_chunk_length  : int,
    output_chunk_length : int,
    batch_size          : int,
    num_stacks          : int,
    num_blocks          : int,
    num_layers          : int,
    layer_widths        : int,
    dropout             : float,
    lr                  : float,
    random_state        : int,
    validation_split    : float,
    Y_col_list          : list,
    X_col_list          : list,
    add_encoders        : bool,
    custom_checkpoint   : bool,
    status              : str,
    existing_df         : pd.DataFrame,
    save_path           : str,
    ram_usage_MB        : float = None,
    fit_cost_seconds    : float = None,
    best_epoch          : int   = None,
    best_val_mape       : float = None,
    best_val_loss       : float = None,
    MAPE_sum            : float = None,
    mape_results        : dict  = None
) -> None:
    """
    Function to store N-BEATS tuning results to excel.

    Args:
        model_name (str)                : Model name.
        work_dir (str)                  : Main dictionary of model name.
        GPU (bool)                      : Whether GPU is used or not.
        dataset_type (str)              : Type of dataset used (i.e. sqrt, sqrt_NoOzon, log1p, log1p_NoOzon, or without optimized dataset, default).
        input_chunk_length (int)        : Number of input chunk length used.
        output_chunk_length (int)       : Number of output chunk length used.
        batch_size (int)                : Number of batch size used.
        num_stacks (int)                : Number of stacks used.
        num_blocks (int)                : Number of blocks used.
        num_layers (int)                : Number of layers used.
        layer_widths (int)              : Number of layer widths used.
        dropout (float)                 : Number of dropout used.
        lr (float)                      : Number of learning rate used.
        random_state (int)              : Number of random state used.
        validation_split (float)        : Proportion of validation split used.
        Y_col_list (list)               : List of targeted columns used.
        X_col_list (list)               : List of covariates columns used.
        add_encoders (bool)             : Whether add encoders is used or not.
        custom_checkpoint (bool)        : Whether use custom checkpoint or not.
        status (str)                    : Model status when tuned (SUCCESS, PRUNED, or OOM SKIPPED).
        existing_df (pd.DataFrame)      : Existing dataframe of tuned history.
        save_path (str)                 : Existing dataframe location path.
        ram_usage_MB (float = None)     : When SUCCESS, store the ram usage on MB.
        fit_cost_seconds (float = None) : When SUCCESS, store the fit cost on seconds.
        best_epoch (int = None)         : When SUCCESS, store best epoch.
        best_val_mape (float = None)    : When SUCCESS, store best val_MeanAbsolutePercentageError.
        best_val_loss (float = None)    : When SUCCESS, store best val_loss.
        MAPE_SUM (float = None)         : When SUCCESS, store MAPE sumation of target variables.
        mape_results (dict = None)      : When SUCCESS, store MAPE results of each target variables.

    Return:
        None : This function only do store results to excel.
    
    """
    # Store pruned params to xlsx
    params_record = {
        'model_name'         : model_name,
        'GPU'                : True if GPU else False,
        'ram_usage_MB'       : ram_usage_MB,
        'fit_cost_seconds'   : fit_cost_seconds,
        'dataset_type'       : dataset_type,
        'input_chunk_length' : input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'n_epochs'           : best_epoch,
        'batch_size'         : batch_size,
        'num_stacks'         : num_stacks,
        'num_blocks'         : num_blocks,
        'num_layers'         : num_layers,
        'layer_widths'       : layer_widths,
        'dropout'            : dropout,
        'lr'                 : lr,
        'random_state'       : random_state,
        'validation_split'   : validation_split,
        'stride'             : output_chunk_length,
        'Y_col_list'         : Y_col_list,
        'X_col_list'         : X_col_list,
        'add_encoders'       : json.dumps(add_encoders) if add_encoders else None
    }

    # EarlyStopping config to store in results
    early_stopping_config = {
        'monitor'  : 'val_MeanAbsolutePercentageError',
        'patience' : 5,
        'min_delta': 0.01,
        'mode'     : 'min'
    }

    # ModelCheckpoint config to store in results
    checkpoints = 'checkpoints'
    if custom_checkpoint:
        checkpoint_config = {
            'dirpath'   : os.path.join(work_dir, model_name, checkpoints),
            'filename'  : "MAPE-best-epoch={epoch}-val_MAPE={val_MeanAbsolutePercentageError:.4f}-val_loss={val_loss:.4f}",
            'monitor'   : 'val_MeanAbsolutePercentageError',
            'save_top_k': 1,
            'mode'      : 'min',
            'auto_insert_metric_name': False
        }

    # Trainer config to store in results
    pl_trainer_kwargs = {
        'accelerator': 'gpu' if GPU else 'cpu',
        'devices'    : [0] if GPU else None,
        'callbacks'  : {
            'early_stopping'  : '',
            'model_checkpoint': '' if custom_checkpoint else None
        }
    }

    # Initialize result
    if status == 'SUCCESS':  
        df_results = pd.DataFrame([{
            'timestamp'         : datetime.now(),
            'MAPE_sum'          : MAPE_sum,
            **mape_results,
            'val_MAPE'          : best_val_mape,
            'val_loss'          : best_val_loss,
            'status'            : status,
            **params_record,
            'early_stopping'    : json.dumps(early_stopping_config),
            'checkpoint_config' : json.dumps(checkpoint_config) if custom_checkpoint else 'Default',
            'trainer_config'    : json.dumps(pl_trainer_kwargs),
        }])
    else:
        df_results = pd.DataFrame([{
            'timestamp'         : datetime.now(),
            'status'            : status,
            **params_record,
            'early_stopping'    : json.dumps(early_stopping_config),
            'checkpoint_config' : json.dumps(checkpoint_config) if custom_checkpoint else 'Default',
            'trainer_config'    : json.dumps(pl_trainer_kwargs),
        }])
        

    # Store results to existing record
    df_results = pd.concat([existing_df, df_results], ignore_index=True)

    # Save path optionally
    if save_path:
        df_results.to_excel(save_path, index=False)
        print(f'✅ Results saved to {save_path}\n')
        
    # Clean up memory
    cleanup(df_results)
    return None


def nbeats_inference_store_to_excel(
    valid_MAPE_sum      : float,
    MAE_sum             : float,
    MAPE_sum            : float,
    MSE_sum             : float,
    RMSE_sum            : float,
    dataset_type        : str,
    n_predict           : int,
    model_name          : str,
    GPU                 : bool,
    fit_cost_seconds    : float,
    predict_cost_seconds: float,
    valid_mape_results  : dict,
    mae_results         : dict,
    mape_results        : dict,
    mse_results         : dict,
    rmse_results        : dict,
    input_chunk_length  : int,
    output_chunk_length : int,
    n_epochs            : int,
    batch_size          : int,
    num_stacks          : int,
    num_blocks          : int,
    num_layers          : int,
    layer_widths        : int,
    dropout             : float,
    lr                  : float,
    random_state        : int,
    validation_split    : float,
    stride              : int,
    Y_col_list          : list,
    X_col_list          : list,
    add_encoders        : dict | bool,
    work_dir            : str,
    existing_df         : pd.DataFrame,
    save_path           : str,
) -> None:
    
    # Store pruned params to xlsx
    params_record = {
        'input_chunk_length'  : input_chunk_length,
        'output_chunk_length' : output_chunk_length,
        'n_epochs'            : n_epochs,
        'batch_size'          : batch_size,
        'num_stacks'          : num_stacks,
        'num_blocks'          : num_blocks,
        'num_layers'          : num_layers,
        'layer_widths'        : layer_widths,
        'dropout'             : dropout,
        'lr'                  : lr,
        'random_state'        : random_state,
        'validation_split'    : validation_split,
        'stride'              : stride,
        'Y_col_list'          : Y_col_list,
        'X_col_list'          : X_col_list,
        'add_encoders'        : json.dumps(add_encoders) if add_encoders else None
    }

    # Initialize result
    df_results = pd.DataFrame([{
        'timestamp'           : datetime.now(),
        'valid_MAPE_sum'      : valid_MAPE_sum,
        'MAE_sum'             : MAE_sum,
        'MAPE_sum'            : MAPE_sum,
        'MSE_sum'             : MSE_sum,
        'RMSE_sum'            : RMSE_sum,
        'dataset_type'        : dataset_type,
        'n_predict'           : n_predict,
        'model_name'          : model_name,
        'GPU'                 : GPU,
        'fit_cost_seconds'    : fit_cost_seconds,
        'predict_cost_seconds': predict_cost_seconds,
        **valid_mape_results,
        **mae_results,
        **mape_results,
        **mse_results,
        **rmse_results,
        **params_record,
    }])
        
    # Store results to existing record
    df_results = pd.concat([existing_df, df_results], ignore_index=True)

    # Save path optionally
    if save_path:
        df_results.to_excel(save_path, index=False)
        print(f'✅ Results saved to {save_path}\n')
        
    # Clean up memory
    cleanup(df_results)
    return None
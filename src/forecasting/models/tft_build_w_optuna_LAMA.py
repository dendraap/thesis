from src.forecasting.utils.libraries_others import os, re
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import torch, TimeSeries, TFTModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, QuantileRegression, MeanAbsolutePercentageError
from src.forecasting.utils.memory import cleanup

# to avoid import of both lightning and pytorch_lightning
class PatchedPruningCallback(optuna.integration.PyTorchLightningPruningCallback, Callback):
    pass

def tft_build_w_optuna(
    Y                   : TimeSeries,
    X_past              : TimeSeries,
    X_future            : TimeSeries,
    input_chunk_length  : int,
    output_chunk_length : int,
    n_epochs            : int,
    batch_size          : int,
    hidden_size         : int,
    lstm_layers         : int, 
    num_attention_heads : int,
    dropout             : float,
    add_encoders        : dict | None,
    validation_split    : float,
    model_name          : str,
    work_dir            : str,
    include_stopper     : bool,
    custom_checkpoint   : bool,
    lr                  : float,
    use_pruner          : bool,
    trial               : Trial
) -> TFTModel: 
    """
    Function to build Fit ofTFT Model with Optuna Tuning Optimization.

        Args:
            Y (TimeSeries)             : Targeted variables to predict. 
            X_past (TimeSeries)        : Past covariates to predict Y.
            X_future (TimeSeries)      : Future covariates to predict Y.
            input_chunk_length (int)   : How many model look to predict.
            output_chunk_length (int)  : How many model can produce prediction.
            batch_size (int)           : Number of data points before making update.
            hidden_size (int)          : Number of hidden_size in TFT.
            lstm_layers (int)          : Number of lstm_layers in TFT.
            num_attention_heads (int)  : Number of num_attention_heads in TFT
            add_encoders (dict | None) : Optionally, adding some cyclic covariates ex. (hour, dayofweek, week, etc)
            dropout (float)            : Dropout probability to be used in fully connected layers.
            validation_split (float)   : To split data input into train and validation to monitor val_loss.
            model_name (str)           : The model name to prevent error for same name.
            work_dir (str)             : Path location to save checkpoints best epochs model.
            include_stopper (bool)     : Whether to utilize EarlyStopping or not.
            custom_checkpoint (bool)   : Whether to load default checkpoint or custom checkpoint.
            lr (float)                 : Learning rate.
            use_pruner (bool)          : Whether to use pruner collbacks for optuna or not.

        Returns:
            TFTModel : This function return the model configuration.
    """

    # Split
    Y_fit, Y_val               = timeseries_train_test_split(Y, test_size =validation_split)
    X_past_fit, X_past_val     = timeseries_train_test_split(X_past, test_size=validation_split)
    X_future_fit, X_future_val = timeseries_train_test_split(X_future, test_size=validation_split)

    # Initialize TorchMetrics, used as the monitor
    torch_metrics = MeanAbsolutePercentageError()

    # pl_trainer_kwargs setup
    pl_trainer_kwargs = {}
    if torch.cuda.is_available():
        pl_trainer_kwargs['accelerator'] = 'gpu'
        pl_trainer_kwargs['devices']     = [0]
        num_workers                      = 4
    else :
        pl_trainer_kwargs['accelerator'] = 'cpu'
        pl_trainer_kwargs['devices']     = 1
        num_workers                      = 0

    callbacks     = []
    early_stopper = None
    if include_stopper:
        early_stopper = EarlyStopping(
            monitor   = 'val_MeanAbsolutePercentageError', #val_loss
            patience  = 8,
            min_delta = 0.01,
            mode      = 'min',
            verbose   = True
        )
        callbacks.append(early_stopper)
        
    # Custom model checkpoint setup
    checkpoints         = 'checkpoints'
    checkpoint_callback = None
    if custom_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath    = os.path.join(work_dir, model_name, checkpoints),
            filename   = 'MAPE-best-epoch={epoch}-val_MAPE={val_MeanAbsolutePercentageError:.4f}-val_loss={val_loss:.4f}',
            monitor    = 'val_MeanAbsolutePercentageError',
            save_top_k = 1,
            mode       = 'min',
            auto_insert_metric_name = False
        )
        callbacks.append(checkpoint_callback)

    # Optuna pruning callback
    if use_pruner:
        pruner = PatchedPruningCallback(trial, monitor='val_MeanAbsolutePercentageError')
        callbacks.append(pruner)

    if custom_checkpoint:
        pl_trainer_kwargs['enable_checkpointing'] = True
        
    pl_trainer_kwargs['callbacks'] = callbacks

    # reproducibility
    torch.manual_seed(42)

    # Initialize model
    model = TFTModel(
        input_chunk_length  = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs            = n_epochs,
        batch_size          = batch_size,
        hidden_size         = hidden_size,
        lstm_layers         = lstm_layers,
        num_attention_heads = num_attention_heads,
        dropout             = dropout,
        optimizer_kwargs    = {'lr': lr},
        likelihood          = QuantileRegression(),
        random_state        = 1502,
        torch_metrics       = torch_metrics,
        pl_trainer_kwargs   = pl_trainer_kwargs,
        model_name          = model_name,
        work_dir            = work_dir,
        log_tensorboard     = True,
        save_checkpoints    = True,   # Enable Darts default checkpoint (_model.pth.tar)
        add_encoders        = add_encoders
    )

    # Fit model
    model.fit(
        series                = Y_fit,
        past_covariates       = X_past_fit,
        future_covariates     = X_future_fit,
        val_series            = Y_val,
        val_past_covariates   = X_past_val,
        val_future_covariates = X_future_val,
        load_best             = True,
        stride                = 1,
        dataloader_kwargs     = {'num_workers': num_workers},
    )

    # Get best model .ckpt file. Used for custom ModelCheckoint
    ## Scan first then take to best epoch
    ckpt_dir  = os.path.join(work_dir, model_name, checkpoints)
    best_ckpt = None

    # Check if directory is exist
    if os.path.exists(ckpt_dir):
        ckpt_list = os.listdir(ckpt_dir)
        print(f'\n📂 Files in checkpoint dir: {ckpt_list}')

        pattern_custom  = r"MAPE-best-epoch=(\d+)-val_MAPE=(-?\d+(?:\.\d+)?)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_default = r"best-epoch=(\d+)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_last    = r"last-epoch=(\d+)(?:\.ckpt)?"

        # Load best model file_name based on usecase
        if custom_checkpoint :
            # Load from custom checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_custom, f):
                    best_ckpt = f
                    print('!! Model loaded from custom checkpoint')
                    break
        else :
            # Load from default checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_default, f):
                    best_ckpt = f
                    print('✅ Model loaded from default checkpoint')
                    break
        
        # To avoid function errors when best_ckpt not found
        if best_ckpt:
            model = model.load_from_checkpoint(
                model_name = model_name,
                work_dir   = work_dir,
                file_name  = best_ckpt
            )
        
        else:
            for f in ckpt_list:
                if re.search(pattern_last, f):
                    best_ckpt = f
                    print('⚠️ No best model found. Using last checkpoint')
                    break
    else:
        print('⚠️ No directory found, canceling load best model.')
        
    # Cleanup memory
    cleanup(ckpt_dir, ckpt_list, best_ckpt, callbacks, checkpoint_callback, early_stopper, add_encoders)
    return model

def tft_build(
    Y                   : TimeSeries,
    X_past              : TimeSeries,
    X_future            : TimeSeries,
    input_chunk_length  : int,
    output_chunk_length : int,
    n_epochs            : int,
    batch_size          : int,
    hidden_size         : int,
    lstm_layers         : int, 
    num_attention_heads : int,
    dropout             : float,
    add_encoders        : dict | None,
    validation_split    : float,
    model_name          : str,
    work_dir            : str,
    include_stopper     : bool,
    custom_checkpoint   : bool,
    lr                  : float,
    use_pruner          : bool,
) -> TFTModel: 
    """
    Function to build Fit ofTFT Model with Optuna Tuning Optimization.

        Args:
            Y (TimeSeries)             : Targeted variables to predict. 
            X_past (TimeSeries)        : Past covariates to predict Y.
            X_future (TimeSeries)      : Future covariates to predict Y.
            input_chunk_length (int)   : How many model look to predict.
            output_chunk_length (int)  : How many model can produce prediction.
            batch_size (int)           : Number of data points before making update.
            hidden_size (int)          : Number of hidden_size in TFT.
            lstm_layers (int)          : Number of lstm_layers in TFT.
            num_attention_heads (int)  : Number of num_attention_heads in TFT
            add_encoders (dict | None) : Optionally, adding some cyclic covariates ex. (hour, dayofweek, week, etc)
            dropout (float)            : Dropout probability to be used in fully connected layers.
            validation_split (float)   : To split data input into train and validation to monitor val_loss.
            model_name (str)           : The model name to prevent error for same name.
            work_dir (str)             : Path location to save checkpoints best epochs model.
            include_stopper (bool)     : Whether to utilize EarlyStopping or not.
            custom_checkpoint (bool)   : Whether to load default checkpoint or custom checkpoint.
            lr (float)                 : Learning rate.
            use_pruner (bool)          : Whether to use pruner collbacks for optuna or not.

        Returns:
            TFTModel : This function return the model configuration.
    """


    # Split
     # Split
    Y_fit, Y_val               = timeseries_train_test_split(Y, test_size =validation_split)
    X_past_fit, X_past_val     = timeseries_train_test_split(X_past, test_size=validation_split)
    X_future_fit, X_future_val = timeseries_train_test_split(X_future, test_size=validation_split)

    # Initialize TorchMetrics, used as the monitor
    torch_metrics = MeanAbsolutePercentageError()

    # pl_trainer_kwargs setup
    pl_trainer_kwargs = {}
    if torch.cuda.is_available():
        pl_trainer_kwargs['accelerator'] = 'gpu'
        pl_trainer_kwargs['devices']     = [0]
        num_workers                      = 4
    else :
        pl_trainer_kwargs['accelerator'] = 'cpu'
        pl_trainer_kwargs['devices']     = 1
        num_workers                      = 0

    callbacks     = []
    early_stopper = None
    if include_stopper:
        early_stopper = EarlyStopping(
            monitor   = 'val_MeanAbsolutePercentageError', #val_loss
            patience  = 8,
            min_delta = 0.01,
            mode      = 'min',
            verbose   = True
        )
        callbacks.append(early_stopper)
        
    # Custom model checkpoint setup
    checkpoints         = 'checkpoints'
    checkpoint_callback = None
    if custom_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath    = os.path.join(work_dir, model_name, checkpoints),
            filename   = 'MAPE-best-epoch={epoch}-val_MAPE={val_MeanAbsolutePercentageError:.4f}-val_loss={val_loss:.4f}',
            monitor    = 'val_MeanAbsolutePercentageError',
            save_top_k = 1,
            mode       = 'min',
            auto_insert_metric_name = False
        )
        callbacks.append(checkpoint_callback)

    if custom_checkpoint:
        pl_trainer_kwargs['enable_checkpointing'] = True
        
    pl_trainer_kwargs['callbacks'] = callbacks

    # reproducibility
    torch.manual_seed(42)

    # Initialize model
    model = TFTModel(
        input_chunk_length  = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs            = n_epochs,
        batch_size          = batch_size,
        hidden_size         = hidden_size,
        lstm_layers         = lstm_layers,
        num_attention_heads = num_attention_heads,
        dropout             = dropout,
        optimizer_kwargs    = {'lr': lr},
        likelihood          = QuantileRegression(),
        random_state        = 1502,
        torch_metrics       = torch_metrics,
        pl_trainer_kwargs   = pl_trainer_kwargs,
        model_name          = model_name,
        work_dir            = work_dir,
        log_tensorboard     = True,
        save_checkpoints    = True,   # Enable Darts default checkpoint (_model.pth.tar)
        add_encoders        = add_encoders
    )

    # Fit model
    model.fit(
        series                = Y_fit,
        past_covariates       = X_past_fit,
        future_covariates     = X_future_fit,
        val_series            = Y_val,
        val_past_covariates   = X_past_val,
        val_future_covariates = X_future_val,
        load_best             = True,
        stride                = 1,
        dataloader_kwargs     = {'num_workers': num_workers},
    )

    # Get best model .ckpt file. Used for custom ModelCheckoint
    ## Scan first then take to best epoch
    ckpt_dir  = os.path.join(work_dir, model_name, checkpoints)
    best_ckpt = None

    # Check if directory is exist
    if os.path.exists(ckpt_dir):
        ckpt_list = os.listdir(ckpt_dir)
        print(f'\n📂 Files in checkpoint dir: {ckpt_list}')

        pattern_custom  = r"MAPE-best-epoch=(\d+)-val_MAPE=(-?\d+(?:\.\d+)?)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_default = r"best-epoch=(\d+)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_last    = r"last-epoch=(\d+)(?:\.ckpt)?"

        # Load best model file_name based on usecase
        if custom_checkpoint :
            # Load from custom checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_custom, f):
                    best_ckpt = f
                    print('!! Model loaded from custom checkpoint')
                    break
        else :
            # Load from default checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_default, f):
                    best_ckpt = f
                    print('✅ Model loaded from default checkpoint')
                    break
        
        # To avoid function errors when best_ckpt not found
        if best_ckpt:
            model = model.load_from_checkpoint(
                model_name = model_name,
                work_dir   = work_dir,
                file_name  = best_ckpt
            )
        
        else:
            for f in ckpt_list:
                if re.search(pattern_last, f):
                    best_ckpt = f
                    print('⚠️ No best model found. Using last checkpoint')
                    break
    else:
        print('⚠️ No directory found, canceling load best model.')
        
    # Cleanup memory
    cleanup(ckpt_dir, ckpt_list, best_ckpt, callbacks, checkpoint_callback, early_stopper, add_encoders)
    return model
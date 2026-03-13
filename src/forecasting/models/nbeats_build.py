from src.forecasting.utils.libraries_others import os, re
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import torch, TimeSeries, NBEATSModel, GaussianLikelihood, MeanAbsolutePercentageError
from src.forecasting.utils.memory import cleanup

def nbeats_build(
    Y_train             : TimeSeries,
    X_train             : TimeSeries,
    Y_valid             : TimeSeries,
    X_valid             : TimeSeries,
    input_chunk_length  : int,
    output_chunk_length : int,
    n_epochs            : int,
    batch_size          : int,
    num_stacks          : int,
    num_blocks          : int,
    num_layers          : int,
    layer_widths        : int,
    dropout             : float,
    add_encoders        : dict | None,
    model_name          : str,
    work_dir            : str,
    lr                  : float,
    random_state        : int
) -> NBEATSModel: 
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
            lr (float)                 : Learning rate.

        Returns:
            TFTModel : This function return the model configuration.
    """

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

    # reproducibility
    torch.manual_seed(42)

    # Initialize model
    model = NBEATSModel(
        input_chunk_length  = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs            = n_epochs,
        batch_size          = batch_size,
        num_stacks          = num_stacks,
        num_blocks          = num_blocks,
        num_layers          = num_layers,
        layer_widths        = layer_widths,
        dropout             = dropout,
        optimizer_kwargs    = {'lr': lr},
        likelihood          = GaussianLikelihood(),
        random_state        = random_state,
        pl_trainer_kwargs   = pl_trainer_kwargs,
        model_name          = model_name,
        work_dir            = work_dir,
        log_tensorboard     = False,
        save_checkpoints    = False,   # Enable Darts default checkpoint (_model.pth.tar)
        add_encoders        = add_encoders
    )

    # Fit model
    model.fit(
        series                = Y_train,
        past_covariates       = X_train,
        val_series            = Y_valid,
        val_past_covariates   = X_valid,
        stride                = 1,
        dataloader_kwargs     = {'num_workers': num_workers},
    )
        
    # Cleanup memory
    cleanup(add_encoders)
    return model
# Modeling
import torch
from darts import concatenate
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from darts.models import TFTModel, NBEATSModel
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Tuning
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import Trial
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
)
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from optuna.exceptions import TrialPruned
from sklearn.model_selection import ParameterSampler

# Evaluation
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from darts.metrics import smape, mape, mae, rmse, mse
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)

# Interpretability
from darts.explainability import TFTExplainer
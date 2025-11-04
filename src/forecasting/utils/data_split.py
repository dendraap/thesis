from forecasting.utils.libraries_data_handling import pd
from forecasting.utils.libraries_modelling import TimeSeries

def dataframe_train_test_split(
    df        : pd.DataFrame,
    test_size : float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split dataframe dataset into train and test.

    Args:
        df (pd.DataFrame) : DataFrame input.
        test_size (float) : Percentage of data test size in decimal.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] : This function return train and test dataframes.
    """
    
    n = len(df)
    split = int(n * (1 - test_size))

    return df.iloc[:split], df.iloc[split:]


def timeseries_train_test_split(
    ts       : TimeSeries, 
    test_size: float
) -> tuple[TimeSeries, TimeSeries]:
    """
    Function to split timeseries dataset (used to split validation for training)

    Args:
        ts (TimeSeries)   : TimeSeries input.
        test_size (float) : Percentage of data test size in decimal.

    Returns:
        tuple[TimeSeries, TimeSeries] : This function return train and validation timeseries.
    """

    n     = len(ts)
    split = int(n * (1 - test_size))

    return ts[:split], ts[split:]

def timeseries_train_test_infer_split(
    df        : pd.DataFrame,
    end_train : pd.Timestamp,
    end_test  : pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function for TFT future dataset to split timeseries dataset into train, test, and inference dataset.

    Args:
        df (pd.DataFrame)        : DataFrame input.
        end_train (pd.Timestamp) : Last timestamp for training data.
        end_test (pd.Timestamp)  : Last timestamp for testing data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] : This function return train, test, and inference dataframes.
    """
    
    # Train: until end_train_time
    train = df[df.index <= end_train]

    # Test: after end_train_time to end_test_time
    test = df[(df.index > end_train) & (df.index <= end_test)]

    # Infer: after end_test_time
    infer = df[df.index > end_test]

    return train, test, infer
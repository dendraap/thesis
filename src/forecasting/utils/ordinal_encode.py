from forecasting.utils.libraries_data_handling import pd

def encode_ordinal(
    df      : pd.DataFrame,
    colname : str,
    encoder : dict[str, str]
) -> pd.Series:
    """
    Function to encode categorical data to numeric ordinal value

    Args:
        df (pd.DataFrame)       : DataFrame input.
        colname (str)           : Which column to change value as ordinal ranking.
        encoder (dict[str, str] : Encoder for oridinal format.

    Returns:
        pd.Series : This function return the series of dataframe.
    """

    return df[colname].replace(encoder).astype(float)
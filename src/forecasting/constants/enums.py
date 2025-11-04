from forecasting.utils.libraries_others import Enum

# Define column group
class ColumnGroup(int, Enum):
    """
    Enum class to define columns by type

    Inherits:
        int  : Allows enum members to behave like integers.
        Enum : Enables symbolic names for column group counts.

    Members:
        FEATURE_NUMERICAL   : Number of features numerical columns.
        FEATURE_CATEGORICAL : Number of features categorical columns.
        FEATURE             : Number of features columns.
        TARGET              : Number of target columns.
        ALL                 : Total number of columns (target + feature), timestamp is not included.
    """    
    FEATURE_NUMERICAL   = 8
    FEATURE_CATEGORICAL = 4
    FEATURE             = FEATURE_NUMERICAL + FEATURE_CATEGORICAL
    TARGET              = 6
    NUMERICAL           = FEATURE_NUMERICAL + TARGET
    ALL                 = FEATURE + TARGET
    

# Define PeriodList (Value per Hour)
class PeriodList(int, Enum):
    """
    Enum class to define time period lengths in hours for time series analysis.

    Inherits:
        int  : Enables direct use of period values as integers.
        Enum : Provides symbolic names for each period.

    Members:
        Y1 : 8760 hours (1 year)
        M6 : 4380 hours (6 months)
        M3 : 2190 hours (3 months)
        M1 : 730 hours  (1 month)
        W1 : 168 hours  (1 week)
        D1 : 24 hours   (1 day)

    Methods:
        label()    : Returns a human-readable label for the period.
        filename() : Returns a filename-friendly string for saving outputs.
    """
    
    Y1 = 8760
    M6 = 4380
    M3 = 2190
    M1 = 730
    W1 = 168
    D1 = 24

    def label(self):
        """
        Returns a descriptive label for the selected period.
        """
        return {
            PeriodList.Y1: 'Yearly Trend',
            PeriodList.M6: '6 Months Trend',
            PeriodList.M3: '3 Months Trend',
            PeriodList.M1: 'Montly Trend',
            PeriodList.W1: 'Weekly Trend',
            PeriodList.D1: 'Daily Trend'
        }.get(self, 'Unknown Period')
    
    def filename(self):
        """
        Returns a filename-safe string for the selected period.
        """
        return {
            PeriodList.Y1: 'yearly_trend',
            PeriodList.M6: '6_months_trend',
            PeriodList.M3: '3_months_trend',
            PeriodList.M1: 'montly_trend',
            PeriodList.W1: 'weekly_trend',
            PeriodList.D1: 'daily_trend'
        }.get(self, 'unknown-period')
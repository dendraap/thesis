from forecasting.utils.libraries_data_handling import pd, np
from forecasting.utils.libraries_plotting import plt, sd, LinearRegression, sns
from forecasting.utils.memory import cleanup
from forecasting.constants.enums import ColumnGroup, PeriodList


def descriptive_statistics_plot(
    df          : pd.DataFrame,
    color_var   : np.ndarray,
    show_values : bool,
) -> None:
    """
    Function to plot descriptive statistics.

    Args:
        df (pd.DataFrame)    : Dataframe input.
        color_var (np.array) : Color defined to plot each column.
        show_values (bool)   : Whether to show label values on plot.

    Returns:
        None : This function just return the plot.
    """      
    
    # Determine number of columns to plot
    n = len(df.columns)

    # Create subplots: one row per column, shared x-axis
    fig, ax = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)

    # Plot each column individually
    for i, col in enumerate(df.columns):
        ax[i].plot(df.index, df[col], color=color_var[i])
        ax[i].set_ylabel(col, fontsize=14)
        ax[i].tick_params(axis='x', rotation=90)

        # Optionally display value labels on each point
        if show_values:
            for x, y in zip(df.index, df[col]):
                ax[i].text(
                    x, y, f'{y:.1f}',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold'
                )

    # Set overall title and adjust layout to avoid overlap
    fig.suptitle('Past Covariates Descriptive Statistics Overview', fontsize=14, fontweight='bold')

    # Show plot data
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    plt.show()
    
    # Save the figure
    fig.savefig('../reports/figures/univariate_analysis/descriptive_statistics_overview.png', dpi=300)

    # Clear figure and variables from memory
    plt.close(fig)
    cleanup(fig, ax, n)
    
    return None


def extract_statistical_values(
    series : pd.Series, 
    statdas: pd.DataFrame
) -> dict[str, float]:
    """
    Function to extract statistical values from a series based on provided statistics dataframe.

    Args:
        series (pd.Series)     : Data column to analyze.
        statdas (pd.DataFrame) : DDescriptive statistics from df.describe().

    Returns:
        dict[str, float] : Dictionary containing Q1, Q3, median, mean, IQR, lower_adjacent, upper_adjacent.
    """
    
    # Extract relevant statistics
    q1     = statdas[series.name]['25%']
    median = statdas[series.name]['50%']
    q3     = statdas[series.name]['75%']
    mean   = statdas[series.name]['mean']
    iqr    = q3 - q1

    # Calculate lower and upper adjacent values
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Find the actual lower and upper adjacent values in the series
    lower_adjacent = series[series >= lower_bound].min()
    upper_adjacent = series[series <= upper_bound].max()

    return {
        'q1'            : q1,
        'q3'            : q3,
        'median'        : median,
        'mean'          : mean,
        'iqr'           : iqr,
        'lower_adjacent': lower_adjacent,
        'upper_adjacent': upper_adjacent
    }


def univariate_analysis(
    df           : pd.DataFrame,
    plot_type    : str,
    col_decode   : dict[str, str],
    nrows        : int,
    ncols        : int,
    figsize      : tuple[int, int],
    group        : ColumnGroup,
    color_variant: np.ndarray = None,
    statdas      : pd.DataFrame = None,
    **plot_kwargs
) -> None:
    """
    Function to plot data distribution of univariate analysis (line, scatter, histogram, & boxplot)

    Args:
        df (pd.DataFrame)           : Dataframe input.
        plot_type (str)             : Type of plot (histplot, boxplot, lineplot, scatterplot).
        col_decode (dict[str, str]) : Decode columns name.
        nrows (int)                 : Number of rows.
        ncols (int)                 : Number of columns.
        figsize (tuple[int, int])   : Figure size.
        group (ColumnGroup)         : Column group (numerical, categorical, all).
        color_variant (np.array)    : Color defined to plot each column.
        statdas (pd.DataFrame)      : Statistical description dataframe, only used for boxplot.
        **plot_kwargs               : Additional plot arguments.

    Returns:
        None: This function just to display plot data.
    """    
    
    # Initialize figure and axes
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.flatten()
    
    # Select columns based on group
    selected_columns = (
        # If group is numerical, select the first N numerical columns
        df.columns[:ColumnGroup.NUMERICAL] if group == ColumnGroup.NUMERICAL else
        
        # If group is categorical, select the last N categorical columns
        df.columns[-ColumnGroup.FEATURE_CATEGORICAL:] if group == ColumnGroup.FEATURE_CATEGORICAL else
        
        # Otherwise, select all columns
        df.columns
    )
    
    # Iterate through selected columns
    for i, col in enumerate(selected_columns):

        # Plot based on plot_type
        if plot_type == 'histplot':
            # Histogram

            ax[i].hist(df[col], bins=plot_kwargs.get('bins', 30), color=color_variant[i])
            ax[i].set_ylabel('Frequency')
            ax[i].set_xlabel('Value')
            ax[i].set_title(col_decode[col], fontweight='bold')

        elif plot_type == 'boxplot':
            # Boxplot
            
            ax[i].boxplot(df[col], vert=False, patch_artist=True, boxprops=dict(facecolor='white'), **plot_kwargs)
            ax[i].set_ylabel('')
            ax[i].set_xlabel('Value')
            ax[i].set_title(col_decode[col], fontweight='bold')
            
            stats          = extract_statistical_values(df[col], statdas)
            q1             = stats['q1']
            q3             = stats['q3']
            median         = stats['median']
            mean           = stats['mean']
            lower_adjacent = stats['lower_adjacent']
            upper_adjacent = stats['upper_adjacent']

            # Add Q1 and Q3 annotation
            ax[i].text(
                q1, 0.65, f'{q1:.2f}', color='b', ha='center', va='bottom',
                transform=ax[i].get_xaxis_transform(), rotation=90, backgroundcolor='w'
            )
            ax[i].text(
                q3, 0.65, f'{q3:.2f}', color='b', ha='center', va='bottom',
                transform=ax[i].get_xaxis_transform(), rotation=90, backgroundcolor='w'
            )

            # Add Lower and upper adjacent annotation
            ax[i].text(
                lower_adjacent, 0.2, f'{lower_adjacent:.2f}', color='purple',
                ha='center', va='bottom', rotation=90,
                transform=ax[i].get_xaxis_transform(), backgroundcolor='w'
            )
            ax[i].text(
                upper_adjacent, 0.2, f'{upper_adjacent:.2f}', color='purple',
                ha='center', va='bottom', rotation=90,
                transform=ax[i].get_xaxis_transform(), backgroundcolor='w'
            )

            # Change mean and median text position based on box plot position
            if i in [7, 8]: 
                # In this dataset, that plots position blocking the legend
                # Mean and median annotation position to left
                ax[i].text(
                    0.02, 0.95, f'Mean: {mean:.2f}', transform=ax[i].transAxes, 
                    fontsize=10, va='top', ha='left', color='g',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0)
                )
                ax[i].text(
                    0.02, 0.86, f'Median: {median:.2f}', transform=ax[i].transAxes, 
                    fontsize=10, va='top', ha='left', color='r',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0)
                )
                
            else:
                # If maen and median position not blocking the legend
                # Mean and median annotation position to right
                ax[i].text(
                    0.98, 0.95, f'Mean: {mean:.2f}', transform=ax[i].transAxes, 
                    fontsize=10, va='top', ha='right', color='g',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0)
                )
                ax[i].text(
                    0.98, 0.86, f'Median: {median:.2f}', transform=ax[i].transAxes, 
                    fontsize=10, va='top', ha='right', color='r',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0))

        else:
            # Lineplot and Scatterplot
            if plot_type == 'lineplot':
                ax[i].plot(df.index, df[col], color=color_variant[i], **plot_kwargs)

            elif plot_type == 'scatterplot':
                ax[i].scatter(df.index, df[col], color=color_variant[i], **plot_kwargs)

            # getattr(sns, plot_type)(x=df.index, y=df[col], ax=ax[i], color=color_variant[i], **plot_kwargs)
            ax[i].set_ylabel('Value')
            ax[i].set_xlabel('')
            ax[i].set_title(col_decode[col], fontweight='bold')

    # Show plot data    
    fig.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig(f'../reports/figures/univariate_analysis/{plot_type}_all_columns.png', dpi=300)

    # Clear figure and variables from memory
    plt.close(fig)
    cleanup(fig, ax, selected_columns)
        
    return None


# Function to handle layout and plot seasonal decompose
def seasonal_decomp_plot(
    res,                     # seasonal decompose result
    period_list: PeriodList, # period list enum
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Function to customize seasonal decomposition plot layout used by parent function.

    Args:
        res                      : An object with seasonal, trend, and resid attributes.
        period_list (PeriodList) : Enum indicating the period of the seasonal decomposition.

    Returns:
        tuple[plt.Figure, list[plt.Axes]]: To keep the figure and axes information for further customization.
    """    
    
    # Initialize figure
    fig = plt.figure(figsize=(14, 8))

    # Split into 2 separate gridspecs
    gs_top = plt.GridSpec(4, 1, hspace=0, top=0.93)
    gs_bottom = plt.GridSpec(4, 1, hspace=0.3)

    # Top axes with shared axes (Oberved, Seasonal, & Residual)
    ax0 = fig.add_subplot(gs_top[0])
    ax1 = fig.add_subplot(gs_top[1], sharex=ax0)
    ax2 = fig.add_subplot(gs_top[2], sharex=ax0)

    # Bottom axes with different axes (detailed seasonal plot)
    ax3 = fig.add_subplot(gs_bottom[3])

    # Plot top axes
    ## Observed
    ax0.plot(res.observed, color='cornflowerblue', label='Observed')
    ax0.legend(loc='upper left', prop={'weight': 'bold'})
    plt.setp(ax0.get_xticklabels(), visible=False) # Remove xlabel

    ## Seasonal
    ax1.plot(res.seasonal, color='limegreen', label='Seasonal Trend')
    ax1.legend(loc='upper left', prop={'weight': 'bold'})
    plt.setp(ax1.get_xticklabels(), visible=False) # Remove xlabel

    ## Residual
    ax2.plot(res.resid, color='coral', label='Residual')
    ax2.legend(loc='upper left', prop={'weight': 'bold'})

    # Plot bottom axes
    ## Get 1 seasonal trend
    n_trend         = 1
    cut             = n_trend*period_list.value
    seasonal_subset = res.seasonal[-cut:]

    ## Detailed seasonal trend
    ax3.plot(seasonal_subset, color='r', label='Detailed Seasonal Trend')
    ax3.legend(loc='upper left', prop={'weight': 'bold'})

    # Set xlabel based on period
    if period_list.value > PeriodList.M3.value:
        # Yearly & 6 months
        
        # Set x interval to every 3 months
        xtick_interval = PeriodList.M3.value
        xticks         = seasonal_subset.index[::xtick_interval]

        # Set x label
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([x.strftime('%Y-%m') for x in xticks])

    elif period_list.value == PeriodList.M3.value:
        # 3 months

        # Set x interval to every 1 month
        xtick_interval = PeriodList.M1.value
        xticks         = seasonal_subset.index[::xtick_interval]

        # Set x label
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([x.strftime('%Y-%m') for x in xticks])
    
    elif period_list.value > PeriodList.W1.value:
        # Montly
        
        # Set x interval to every 1 week
        xtick_interval = PeriodList.W1.value
        xticks         = seasonal_subset.index[::xtick_interval]
        
        # Set x label
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([x.strftime('%m-%d') for x in xticks])
        
    elif period_list.value == PeriodList.W1.value:
        # Weekly
        
        # Set x interval to every 1 day
        xtick_interval = PeriodList.D1.value
        xticks         = seasonal_subset.index[::xtick_interval]
        
        # Set x label
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([x.strftime('%A') for x in xticks])
        
    else:
        # Daily
        
        # Set x interval to every 3 hours
        xtick_interval = 3
        xticks         = seasonal_subset.index[::xtick_interval]
        
        # Set x label
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([x.strftime('%H:%M') for x in xticks])
    
    return fig, [ax0, ax1, ax2, ax3]

# Seasonal Decomposition Function to Visualize Trend, Seasonality, and Residuals
def seasonal_decomp(
        df         : pd.DataFrame,
        col_decode : dict[str, str],
        period_list: PeriodList,
) -> None:
    """
    Main function to plot seasonal decomposition for each column based on defined period.

    Args:
        df (pd.DataFrame)           : Dataframe input.
        col_decode (dict[str, str]) : A dictionary to decode column names.
        period_list (PeriodList)    : A period enum to define the seasonal decomposition period.

    Returns:
        None: This function just to display plot data.
    """    

    # Seasonal decomposition initialize for each column
    for col in df:
        res       = sd(df[col], period=period_list, extrapolate_trend='freq')
        fig, ax = seasonal_decomp_plot(res, period_list)

        # Apply new title
        fig.suptitle(f'{period_list.label()} of {col_decode[col]}', fontweight='bold', fontsize=14)

        # Save figure per subplots
        fig.savefig(f'../reports/figures/univariate_analysis/seasonal_decompose/{period_list.filename()}-{col}.png', dpi=300)
        plt.show()

        # Clear figure from memory
        plt.close(fig)
        cleanup(res, fig, ax)

    return None


# Bivariate Analysis Function to Visualize 1 Target Variable to All Exogenous Features
def bivariate_analysis(
    df           : pd.DataFrame,
    Y            : pd.DataFrame,
    X            : pd.DataFrame,
    col_decode   : dict[str, str],
    nrows        : int,
    ncols        : int,
    color_variant: np.ndarray,
    figsize      : tuple[int, int],
) -> None:
    """
    Function to plot data distribution of bivariate analysis (regplot)

    Args:
        df (pd.DataFrame)           : Dataframe input.
        Y                           : Targetted dataframe input.
        X                           : Independent dataframe input.
        col_decode (dict[str, str]) : Decode columns name.
        nrows (int)                 : Number of rows.
        ncols (int)                 : Number of columns.
        figsize (tuple[int, int])   : Figure size.
        color_variant (np.array)    : Apply different colors for each scatterplots.
        group (ColumnGroup)         : Column group (numerical, categorical, all).
        **plot_kwargs               : Additional plot arguments.

    Returns:
        None: This function just to display plot data.
    """    
    
    # Initialize figure and ax
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax      = ax.flatten()
    
    # Initialize Y value
    y = Y.squeeze().values
    
    # Iterate through meteorological columns
    for i, col in enumerate(X.columns):
        x = df[col].values
        axes = ax[i]
        
        # Scatter plot
        axes.scatter(x, y, color=color_variant[i], alpha=0.6)
        
        # Linear regression line
        model = LinearRegression()
        x_reshape = x.reshape(-1, 1)
        model.fit(x_reshape, y)
        x_sorted = np.sort(x)
        y_pred = model.predict(x_sorted.reshape(-1, 1))
        
        # Initialize color line (avoid same color as scatter color)
        line_color = 'b' if color_variant[i] in ['orangered', 'magenta', 'deeppink', 'maroon'] else 'r'
        axes.plot(x_sorted, y_pred, color=line_color, linewidth=2)
        
        # Set label
        axes.set_xlabel(col_decode[col], fontsize=12, fontweight='bold')
        axes.set_ylabel('')

    # Set overall title
    fig.suptitle(f'{col_decode[Y.name]} vs Independent Variables', fontsize=14, fontweight='bold')
    
    # Show data plot
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    # Save fig
    fig.savefig(f'../reports/figures/bivariate_analysis/bivariate_{Y.name}.png', dpi=300)
    plt.show()

    # Clear figure from memory
    plt.close(fig)
    cleanup(fig, ax, y, model)
    
    return None


def correlation_heatmap(
    df       : pd.DataFrame,
    method   : str,
    title    : str,
    save_path: str,
    figsize  : tuple[int, int] = (14, 6),
    annot    : bool = True,
    fmt      : str = '.2f',
    cmap     : str ='coolwarm',
    center   : float = 0.0,
) -> None:
    """
    Plot correlation heatmap (Pearson or Spearman). 
    Filtered with signification level.

    Args:
        df_corr (pd.DataFrame)    : Correlation results with dataframe format.
        method (str)              : Correlation method (Pearson or Spearman).
        title (str)               : Title of the plot.
        save_path (str)           : Path to save the figure.
        figsize (tuple[int, int]) : Size of the figure.
        annot (bool)              : Wheter to annotate values.
        fmt (str)                 : Format for annotation text.
        cmap (str)                : Colormap.
        center (float)            : Netral values for colormap.

    Returns:
        None : This function just to display plot data.
    """

    # Transform dataframe structure
    pivot = df.pivot(index='Target', columns='Feature', values='Correlation')

    fig = plt.figure(figsize=figsize)

    # Plot correlatioin heatmap
    sns.heatmap(pivot, annot=annot, fmt=fmt, cmap=cmap, center=center)
    fig.suptitle(title, fontweight='bold', fontsize=14)
    fig.tight_layout()

    # Save the figure
    fig.savefig(save_path, dpi=300)

    # Show plot data
    plt.show()

    # Clear figure from memory
    plt.close(fig)
    cleanup(fig, pivot)

    return None


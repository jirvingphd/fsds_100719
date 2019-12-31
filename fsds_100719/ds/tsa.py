## Lab Function
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

from IPython.display import display

## Lab Function
def stationarity_check(TS,plot=True,col=None,rollwindow=8):
    """
    Performs the Augmented Dickey-Fuller unit root test on a time series.

    - The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative that there is no unit root. 
        - A unit root (also called a unit root process or a difference stationary process) 
        is a stochastic trend in a time series, sometimes called a “random walk with drift”; 
        - If a time series has a unit root, it shows a systematic pattern that is unpredictable, and non-stationary.
        
    From: https://learn.co/tracks/data-science-career-v2/module-4-a-complete-data-science-project-using-multiple-regression/working-with-time-series-data/time-series-decomposition
    """
    # Import adfuller
    from statsmodels.tsa.stattools import adfuller

    if col is not None:
        # Perform the Dickey Fuller Test
        dftest = adfuller(TS[col]) # change the passengers column as required 
    else:
        dftest=adfuller(TS)
 
    if plot:
        # Calculate rolling statistics
        rolmean = TS.rolling(window = rollwindow, center = False).mean()
        rolstd = TS.rolling(window = rollwindow, center = False).std()

        #Plot rolling statistics:
        fig = plt.figure(figsize=(12,6))
        orig = plt.plot(TS, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print ('[i] Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','# of Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        

    sig = dfoutput['p-value']<.05
    print (dfoutput)
    print()
    if sig:
        print(f"[i] p-val {dfoutput['p-value'].round(4)} is <.05, so we reject the null hypothesis.")
        print("\tThe time series is NOT stationary.")
    else:
        print(f"[i] p-val {dfoutput['p-value'].round(4)} is >.05, therefore we support the null hypothesis.")
        print('\tThe time series IS stationary.')
    
    return dfoutput


def calc_bollinger_bands(ts,window=20,col=None):
    """Calculates Bollinger Bands for time series. If ts is a dataframe, col specifies data.
    Normally used for financial/stock market data and uses 20 days for rolling calculations."""

    bands_df = pd.DataFrame()
    if col is not None:
        ts=ts[col]

    ## Calc rolling Moving Average
    mean = ts.rolling(window).mean()
    std = ts.rolling(window).std()

    ## Calc MA +2*std(window)
    upper = mean+ 2*(std)

    ## Lower
    lower = mean -2*(std)

    ## COMBINE DATA INTO 1 DF
    bands_df['Raw Data'] = ts
    bands_df['Rolling Mean'] = mean
    bands_df['Lower Band'] = lower
    bands_df['Upper Band'] = upper

    return bands_df

def calc_bollinger_bands_plot(ts,window=20,col=None,
                              figsize=(10,6),
                              set_kws=dict(
                              ylabel='House Price ($)',
                              title="Bollinger Bands")
                              ):                           
    """Calculates Bollinger Bands for time series. If ts is a dataframe, col specifies data.
    Normally used for financial/stock market data and uses 20 days for rolling calculations.
    """
    
    plot_df = calc_bollinger_bands(ts=ts,window=window,col=col)#,figsize=figsize)
    
    ## SPECIFY STYLE PER COLUMN
    plot_styles = {}
    plot_styles['Raw Data'] = dict(lw=1,ls='-',c='black')
    plot_styles['Rolling Mean'] = dict(lw=3,alpha=0.6, c='green')
    plot_styles['Lower Band'] = dict(lw=2,ls=':',c='blue')
    plot_styles['Upper Band'] = dict(lw=2,ls=':',c='red')    

    ## Make figure and loop through columns
    fig,ax = plt.subplots(figsize=figsize)
    for col in list(plot_df.columns):
        plot_df[col].plot(**plot_styles[col])
 
    ax.legend()
    ax.set(**set_kws)
    return fig,ax

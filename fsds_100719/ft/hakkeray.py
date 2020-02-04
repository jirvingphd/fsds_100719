"""
Name: Ru Keïn
Email: rukeine@gmail.com
GitHub Profile: https://github.com/hakkeray
"""
# /***************//***************//***************/
# /* HAKKERAY.py *//* Ru Kein *//* www.hakkeray.com */ 
# /***************//***************//***************/
#  ________________________
# | hakkeray |  Updated:  |
# | v2.0.0   |  2.03.2020 |
# ------------------------
#       \                    / \  //\
#        \    |\___/|      /   \//  \\
#             /0  0  \__  /    //  | \ \    
#            /     /  \/_/    //   |  \  \  
#            @_^_@'/   \/_   //    |   \   \ 
#            //_^_/     \/_ //     |    \    \
#         ( //) |        \///      |     \     \
#       ( / /) _|_ /   )  //       |      \     _\
#     ( // /) '/,_ _ _/  ( ; -.    |    _ _\.-~        .-~~~^-.
#   (( / / )) ,-{        _      `-.|.-~-.           .~         `.
#  (( // / ))  '/\      /                 ~-. _ .-~      .-~^-.  \
#  (( /// ))      `.   {            }                   /      \  \
#   (( / ))     .----~-.\        \-'                 .~         \  `. \^-.
#              ///.----..>        \             _ -~             `.  ^-`  ^-_
#                ///-._ _ _ _ _ _ _}^ - - - - ~                     ~-- ,.-~





#import required libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
import numpy as np
from numpy import log 
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import IPython.display as dp


plt.style.use('seaborn-bright')
mpl.style.use('seaborn-bright')
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 24}
mpl.rc('font', **font)


"""

>>>>>>>>>>>>>>>>>> TIME SERIES <<<<<<<<<<<<<<<<<<<<<<

* makeTime()
* checkTime()
* mapTime()


"""

def makeTime(data, idx):
    df = data.copy()
    df[idx] = pd.to_datetime(df[idx], errors='coerce')
    df['DateTime'] = df[idx].copy()
    df.set_index(idx, inplace=True, drop=True)
    return df
    

def mapTime(d, xcol, ycol='MeanValue', X=None, vlines=None, MEAN=True):
    """
    'Maps' a timeseries plot of zipcodes 
    
    # fig,ax = mapTime(d=HUDSON, xcol='RegionName', ycol='MeanValue', MEAN=True, vlines=None)
    
    **ARGS
    d: takes a dictionary of dataframes OR a single dataframe
    xcol: column in dataframe containing x-axis values (ex: zipcode)
    ycol: column in dataframe containing y-axis values (ex: price)
    X: list of x values to plot on x-axis (defaults to all x in d if empty)
    
    **kw_args
    mean: plots the mean of X (default=True)
    vlines : default is None: shows MIN_, MAX_, crash 
    
    *Ex1: `d` = dataframe
    mapTime(d=NY, xcol='RegionName', ycol='MeanValue', X=list_of_zips)
    
    *Ex2: `d` = dictionary of dataframes
    mapTime(d=NYC, xcol='RegionName', y='MeanValue')
    """
    #import required libraries
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    from pandas import Series
    # from pandas.plotting import register_matplotlib_converters
    # register_matplotlib_converters()
    import numpy as np
    from numpy import log 
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    import IPython.display as dp
    
    # create figure for timeseries plot
    fig, ax = plt.subplots(figsize=(21,13))
    plt.title(label=f'Time Series Plot: {str(ycol)}')
    ax.set(title='Mean Home Values', xlabel='Year', ylabel='Price($)', font_dict=font_title)  
    
    zipcodes = []
    #check if `d` is dataframe or dictionary
    if type(d) == pd.core.frame.DataFrame:
        # if X is empty, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d[xcol].unique())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        for zc in zipcodes:
            if zc < breakpoint:
                ls='-'
            else:
                ls='--'
            ts = d[zc][ycol].rename(zc)#.loc[zc]
            ts = d[ycol].loc[zc]
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls)
        ## Calculate and plot the MEAN
        
        if MEAN:
            mean = d[ycol].mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
    
    elif type(d) == dict:
        # if X passed in as empty list, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d.keys())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        # create empty dictionary for plotting 
        txd = {}
        # create different linestyles for zipcodes (easier to distinguish if list is long)
        for i,zc in enumerate(zipcodes):
            if i < breakpoint:
                ls='-'
            else:
                ls='--'
            # store each zipcode as ts  
            ts = d[zc][ycol].rename(zc)
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls, lw=2);
            txd[zc] = ts
            
        if MEAN:
            mean = pd.DataFrame(txd).mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
            
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
            
    if vlines:
        ## plot crash, min and max vlines
        crash = '01-2009'
        ax.axvline(crash, label='Housing Index Drops',color='red',ls=':',lw=2)
        MIN_ = ts.loc[crash:].idxmin()
        MAX_ = ts.loc['2004':'2010'].idxmax()
        ax.axvline(MIN_, label=f'Min Price Post Crash {MIN_}', color='black',lw=2)    
        ax.axvline(MAX_,label='Max Price', color='black', ls=':',lw=2) 

    return fig, ax





#### clockTime() --- time-series snapshot statistical summary ###
#
#  /\    /\    /\    /\    
# / CLOCKTIME STATS /
#     \/    \/    \/
#  

"""
clockTime()

Dependencies include the following METHODS:
- check_time(data, time) >>> convert to datetimeindex
- test_time(TS, y) >>> dickey-fuller (stationarity) test
- roll_time() >>> rolling mean
- freeze_time() >>> seasonality check
- diff_time() >>> differencing 
- autoplot() >>> autocorrelation and partial autocorrelation plots

"""
# class clockTime():
#     def __init__(data, time, x1, x2, y, freq=None):

#         self.data = data
#         self.time = time
#         self.x1 = x1
#         self.x2 = x2
#         self.y = y
#         self.freq = freq

def clockTime(ts, lags, d, TS, y):
    """    
     /\    /\    /\    /\  ______________/\/\/\__-_-_
    / CLOCKTIME STATS /  \/
        \/    \/    \/    

    # clockTime(ts, lags=43, d=5, TS=NY, y='MeanValue',figsize=(13,11))
    #
    # ts = df.loc[df['RegionName']== zc]["MeanValue"].rename(zc).resample('MS').asfreq()
    """
    # import required libraries
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import log 
    import pandas as pd
    from pandas import Series
    from pandas.plotting import autocorrelation_plot
    from pandas.plotting import lag_plot
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf  
    
    print(' /\\   '*3+' /')
    print('/ CLOCKTIME STATS')
    print('    \/'*3)

    #**************#   
    # Plot Time Series
    #original
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    ts.plot(label='Original', ax=axes[0,0],c='red')
    # autocorrelation 
    autocorrelation_plot(ts, ax=axes[0,1], c='magenta') 
    # 1-lag
    autocorrelation_plot(ts.diff().dropna(), ax=axes[1,0], c='green')
    lag_plot(ts, lag=1, ax=axes[1,1])
    
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    # DICKEY-FULLER Stationarity Test
    # TS = NY | y = 'MeanValue'
    dtest = adfuller(TS[y].dropna())
    if dtest[1] < 0.05:
        ## difference data before checking autoplot
        stationary = False
        r = 'rejected'
    else:
        ### skip differencing and check autoplot
        stationary = True 
        r = 'accepted'

    #**************#
    # ts orders of difference
    ts1 = ts.diff().dropna()
    ts2 = ts.diff().diff().dropna()
    ts3 = ts.diff().diff().diff().dropna()
    ts4 = ts.diff().diff().diff().diff().dropna()
    tdiff = [ts1,ts2,ts3,ts4]
    # Calculate Standard Deviation of Differenced Data
    sd = []
    for td in tdiff:
        sd.append(np.std(td))
    
    #sd = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD = pd.DataFrame(data=sd,index=['ts1',' ts2', 'ts3', 'ts4'], columns={'sd'})
    #SD['sd'] = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD['D'] = ['d=1','d=2','d=3','d=4']
    MIN = SD.loc[SD['sd'] == np.min(sd)]['sd']

    # Extract and display full test results 
    output = dict(zip(['ADF Stat','p-val','# Lags','# Obs'], dtest[:4]))
    for key, value in dtest[4].items():
        output['Crit. Val (%s)'%key] = value
    output['min std dev'] = MIN
    output['NULL HYPOTHESIS'] = r
    output['STATIONARY'] = stationary
     
    # Finding optimal value for order of differencing
    from pmdarima.arima.utils import ndiffs
    adf = ndiffs(x=ts, test='adf')
    kpss = ndiffs(x=ts, test='kpss')
    pp = ndiffs(x=ts, test='pp')
        
    output['adf,kpss,pp'] = [adf,kpss,pp]

    #**************#
    # show differencing up to `d` on single plot (default = 5)
    fig2 = plt.figure(figsize=(13,5))
    ax = fig2.gca()
    for i in range(d):
        ax = ts.diff(i).plot(label=i)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # DIFFERENCED SERIES
    fig3 = plt.figure(figsize=(13,5))
    ts1.plot(label='d=1',figsize=(13,5), c='blue',lw=1,alpha=.7)
    ts2.plot(label='d=2',figsize=(13,5), c='red',lw=1.2,alpha=.8)
    ts3.plot(label='d=3',figsize=(13,5), c='magenta',lw=1,alpha=.7)
    ts4.plot(label='d=4',figsize=(13,5), c='green',lw=1,alpha=.7)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True, 
               fancybox=True, facecolor='lightgray')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    #**************#
    
    # Plot ACF, PACF
    fig4,axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    plot_acf(ts1,ax=axes[0,0],lags=lags)
    plot_pacf(ts1, ax=axes[0,1],lags=lags)
    plot_acf(ts2,ax=axes[1,0],lags=lags)
    plot_pacf(ts2, ax=axes[1,1],lags=lags)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # plot rolling mean and std
    #Determine rolling statistics
    rolmean = ts.rolling(window=12, center=False).mean()
    rolstd = ts.rolling(window=12, center=False).std()
        
    #Plot rolling statistics
    fig = plt.figure(figsize=(13,5))
    orig = plt.plot(ts, color='red', label='original')
    mean = plt.plot(rolmean, color='cyan', label='rolling mean')
    std = plt.plot(rolstd, color='orange', label='rolling std')
    
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left") 
    plt.title('Rolling mean and standard deviation')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # # Check Seasonality 
    """
    Calculates and plots Seasonal Decomposition for a time series
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomp = seasonal_decompose(ts, model='additive') # model='multiplicative'

    decomp.plot()
    ts_seas = decomp.seasonal

    ax = ts_seas.plot(c='green')
    fig = ax.get_figure()
    fig.set_size_inches(13,11)

    ## Get min and max idx
    min_ = ts_seas.idxmin()
    max_ = ts_seas.idxmax()
    min_2 = ts_seas.loc[max_:].idxmin()

    ax.axvline(min_,label=min_,c='red')
    ax.axvline(max_,c='red',ls=':', lw=2)
    ax.axvline(min_2,c='red', lw=2)

    period = min_2 - min_
    ax.set_title(f'Season Length = {period}')

    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
   
    #*******#
    clock = pd.DataFrame.from_dict(output, orient='index')
    print(' /\\   '*3+' /')
    print('/ CLOCK-TIME STATS')
    print('    \/'*3)
    
    #display results
    print('---'*9)
    return clock








"""

>>>>>>>>>>>>>>>>>> Machine Learning MODELS <<<<<<<<<<<<<<<<<<<<<<

* ttXsplit()


"""


#### ----> ttXsplit()



def ttXsplit(tx, tSIZE, tMIN):
    """
    # train, test = ttXsplit(ts, 0.2, 2)
    """
    # idXsplit
    import math
    idx_split = math.floor(len(tx.index)*(1-tSIZE))
    
    n = len(tx.iloc[idx_split:]) 
    if n < tMIN:
        idx_split = (len(tx) - tMIN)

    train = tx.iloc[:idx_split]
    test = tx.iloc[idx_split:]
    print(f'train: {len(train)} | test: {len(test)}')
    
    return train, test


def mind_your_PDQs(P=range(0,3), D=range(1,3), Q=range(0,3), s=None):
    """

    pdqs = mind_your_PDQs()
    pdqs['pdq']

    """
    import itertools
    pdqs = {}
    
    if s is None:
        pdqs['pdq'] = list(itertools.product(P,D,Q))
    else:
        pdqs['PDQs'] = list(itertools.product(P,D,Q,s))
    return pdqs


def stopwatch(time='time'): 
    """
    # stopwatch('stop')

    """
    import datetime as dt
    import tzlocal as tz
    if time == 'now':
        now = dt.datetime.now(tz=tz.get_localzone())
        print(now)
    if time=='start':
        now = dt.datetime.now(tz=tz.get_localzone())
        start = now.strftime('%m/%d/%Y - %I:%M:%S %p')
        print('start:', start)
        
    elif time == 'stop':
        now = dt.datetime.now(tz=tz.get_localzone())
        stop = now.strftime('%m/%d/%Y - %I:%M:%S %p')
        print('stop:', stop)
    
    elif time == 'time':
        now = dt.datetime.now(tz=tz.get_localzone())
        time = now.strftime('%m/%d/%Y - %I:%M:%S %p')
        print(time,'|', now)
        
    return time


# Run a grid with pdq and seasonal pdq parameters calculated above and get the best AIC value

def gridMAX(ts, pdq, PDQM=None, verbose=False):
    """
    Ex:
    gridX, best_params = gridMAX(ts,pdq=pdq)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import statsmodels.api as sm   
    stopwatch('start')

    print(f'[*] STARTING GRID SEARCH')
    
    # store to df_res
    grid = [['pdq','PDQM','AIC']]
    
    for comb in pdq:
        if PDQM is None:
            PDQM=[(0, 0, 0, 0)]
        for combs in PDQM:
            mod = sm.tsa.statespace.SARIMAX(ts,
                                            order=comb,
                                            seasonal_order=combs,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            output = mod.fit()

            grid.append([comb, combs, output.aic])
            if verbose:
                print('ARIMA {} x {}12 : AIC Calculated ={}'.format(comb, 
                                                                    combs, 
                                                                   output.aic))           
    
    stopwatch('stop')
    print(f"[**] GRID SEARCH COMPLETE")
    gridX = pd.DataFrame(grid[1:], columns=grid[0])
    gridX = gridX.sort_values('AIC').reset_index()
    best_params = dict(order=gridX.iloc[0].loc['pdq'])
    best_pdq = gridX.iloc[0][1]
    best_pdqm = gridX.iloc[0][2]
    display(gridX, best_params)
    return gridX, best_params      






def calcROI(investment, final_value):
    """This function takes in a series of forecasts to predict the return
    on investment spanning the entire forecast."""
    r = np.round(((final_value - investment) / investment)*100,3)
    return r


#ts = NYC[zc]['MeanValue'].rename(zc)
def forecastX(model_output, train, test, start=None, end=None, get_metrics=False):
    """
    Uses get_prediction=() and conf_int() methods from statsmodels 
        get_prediction (exog,transform,weightsrow_labels,pred_kwds)
    """
    if start is None:
        start = test.index[0]     
    if end is None:
        end = test.index[-1]    
        
    # Get predictions starting from 2013 and calculate confidence intervals.
    prediction = model_output.get_prediction(start=start,end=end, dynamic=True)
    
    forecast = prediction.conf_int()
    forecast['predicted_mean'] = prediction.predicted_mean
    fc_plot = pd.concat([forecast, train], axis=1)

    ## Get ROI Forecast:
    r = calcROI(investment=forecast['predicted_mean'].iloc[0], 
                final_value=forecast['predicted_mean'].iloc[-1])

    zc = train.name

    fig, ax = plt.subplots(figsize=(21,13))
    train.plot(ax=ax,label='Training Data',lw=4) # '1996-04-01, # 2013-11-01
    test.plot(ax=ax,label='Test Data',lw=4) # '2013-12-01 , '2018-04-01
    
    forecast['predicted_mean'].plot(ax=ax, label='Forecast', color='magenta',lw=4)

    ax.fill_between(forecast.index, 
                    forecast.iloc[:,0], 
                    forecast.iloc[:,1],
                    color="white", 
                    alpha=.5, 
                    label = 'conf_int')
    
    ax.fill_betweenx(ax.get_ylim(), test.index[0], test.index[-1], color='darkslategray',alpha=0.5, zorder=-1)
    ax.fill_betweenx(ax.get_ylim(), start, end, color='darkslategray',zorder=-1)
    
    ax.legend(loc="upper left",bbox_to_anchor=(1.04,1), ncol=2,fontsize='small',frameon=True, fancybox=True, framealpha=.15, facecolor='k')
    ax.set(title=f"Predictions for {zc}: ROI = {r}%")
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Home Value $USD')
    fig = ax.get_figure()
    fc_plot['zipcode']= train.name
    plt.show()
    
    if get_metrics == True:
        metrics = model_evaluation(ts_true=test, ts_pred=forecast['predicted_mean'])

    return r, forecast, fig, ax

# r,forecast, fig, ax = forecastX(model_output, train, test, get_metrics=True)
# forecast
# r



def gridMAXmeta(KEYS, s=False):
    """
    Opt1: gridMAXmeta(KEYS=NYC, s=False)
    
    KEYS: dict of dfs or timeseries
    NOTE: if passing in dict of full dataframes, s=True
    (gridMAXmeta will create dict of ts for you)
    
    Opt2: gridMAXmeta(KEYS=txd, s=True)
    KEYS: dictionary of ts - skip the meta ts creation
    """
    if s is False:
        # loop thru each zipcode to create timeseries from its df in NYC dfdict
        txd = {}
        for i,zc in enumerate(NYC):
            # store each zipcode as ts  
            ts = NYC[zc]['MeanValue'].rename(zc)
            txd[zc] = ts
    else:
        txd = KEYS
        
    pdqs = mind_your_PDQs()
    metagrid = {} 
    ROI = {}

    for zc,ts in txd.items():
        print('\n')
        print('---'*30)
        print('---'*30)
        print(f'ZIPCODE: {zc}')
        ## Train test split
        train, test = ttXsplit(ts, 0.1, 2)


        ## gridMAX gridsearch
        ###### TEST DATA ####
        gridX, best_params = gridMAX(train, pdq=pdqs['pdq'])
        metagrid[zc]={}
        metagrid[zc]['gridX']= gridX.iloc[0]
        metagrid[zc]['pdq'] = best
        metagrid[zc]['aic'] = gridX.iloc[0][3]
        
        ## Using best params
        best_params
        
        ##### SARIMAX: USING ENTIRE TIME SERIES ###
        model_output = SARIMAX(ts,
                               **best_params,
                               enforce_invertibility=False,
                               enforce_stationarity=False).fit()

        metagrid[zc]['model'] = model_output

        r, forecast,fig,ax = forecastX(model_output,
                                       train, test, 
                                       start=ts.index[-1],
                                       end=ts.index.shift(24)[-1],
                                       get_metrics=False)
        metagrid[zc]['forecast'] = forecast
        ROI[zc] = r
        metagrid[zc]['ROI'] = r
        ROI[zc] = r
        
    return metagrid, ROI

# metagrid, ROI = gridMAXmeta(KEYS=NYC, s=False)

# import fsds_100719 as fs 
# from fsds_100719.ds import ihelp,ihelp_menu, reload
# from fsds_100719.jmi import print_docstring_template
# print(f"[i] You're using V {fs.__version__} of fsds.")
# HOT_STATS() function: display statistical summaries of a feature column

def hot_stats(data, column, verbose=False, t=None):
    """
    Scans the values of a column within a dataframe and displays its datatype, 
    nulls (incl. pct of total), unique values, non-null value counts, and 
    statistical info (if the datatype is numeric).
    
    ---------------------------------------------
    
    Parameters:
    
    **args:
    
        data: accepts dataframe
    
        column: accepts name of column within dataframe (should be inside quotes '')
    
    **kwargs:
    
        verbose: (optional) accepts a boolean (default=False); verbose=True will display all 
        unique values found.   
    
        t: (optional) accepts column name as target to calculate correlation coefficient against 
        using pandas data.corr() function. 
    
    -------------
    
    Examples: 
    
    hot_stats(df, 'str_column') --> where df = data, 'string_column' = column you want to scan
    
    hot_stats(df, 'numeric_column', t='target') --> where 'target' = column to check correlation value
    
    -----------------
    Developer notes: additional features to add in the future:
    -get mode(s)
    -functionality for string objects
    -pass multiple columns at once and display all
    -----------------
    
    """
    # assigns variables to call later as shortcuts 
    feature = data[column]
    rdash = "-------->"
    ldash = "<--------"
    
    # figure out which hot_stats to display based on dtype 
    if feature.dtype == 'float':
        hot_stats = feature.describe().round(2)
    elif feature.dtype == 'int':
        hot_stats = feature.describe()
    elif feature.dtype == 'object' or 'category' or 'datetime64[ns]':
        hot_stats = feature.agg(['min','median','max'])
        t = None # ignores corr check for non-numeric dtypes by resetting t
    else:
        hot_stats = None

    # display statistics (returns different info depending on datatype)
    print(rdash)
    print("HOT!STATS")
    print(ldash)
    
    # display column name formatted with underline
    print(f"\n{feature.name.upper()}")
    
    # display the data type
    print(f"Data Type: {feature.dtype}\n")
    
    # display the mode
    print(hot_stats,"\n")
    print(f"à-la-Mode: \n{feature.mode()}\n")
    
    # find nulls and display total count and percentage
    if feature.isna().sum() > 0:  
        print(f"Found\n{feature.isna().sum()} Nulls out of {len(feature)}({round(feature.isna().sum()/len(feature)*100,2)}%)\n")
    else:
        print("\nNo Nulls Found!\n")
    
    # display value counts (non-nulls)
    print(f"Non-Null Value Counts:\n{feature.value_counts()}\n")
    
    # display count of unique values
    print(f"# Unique Values: {len(feature.unique())}\n")
    # displays all unique values found if verbose set to true
    if verbose == True:
        print(f"Unique Values:\n {feature.unique()}\n")
        
    # display correlation coefficient with target for numeric columns:
    if t != None:
        corr = feature.corr(data[t]).round(4)
        print(f"Correlation with {t.upper()}: {corr}") 
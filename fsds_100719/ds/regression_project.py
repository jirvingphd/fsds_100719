"""A Collection of functions from ft study group for section 25."""
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sms
    import statsmodels.formula.api as smf

def make_ols_f(df,target='price',cat_cols = [],
               col_list=None, show_summary=True,exclude_cols=[]):
    """
    Uses the formula api of Statsmodels for ordinary least squares regression.
    
    Args:
        df (Frame): data
        target (str, optional): Column to predict. Defaults to 'price'.
        cat_cols (list, optional): Columns to treat as categorical (and one-hot).
        col_list ([type], optional): List of columns to use. Defaults to all columns besides exclude_cols.
        show_summary (bool, optional): Display the model.summary() before returning model. Defaults to True.
        exclude_cols (list, optional): List of column names to exclude. Defaults to []. 
            - Note: if a column name doesn't appear in the dataframe, there will be no error nor warning message.
    
    Returns:
        model: The fit statsmodels OLS model
    """
    import statsmodels.api as sms
    import statsmodels.formula.api as smf
    from IPython.display import display
    
    if col_list is None:
        col_list = list(df.drop(target,axis=1).columns)
        
    ## remove exclude cols
    [col_list.remove(ecol) for ecol in exclude_cols if ecol in col_list]

    features = '+'.join(col_list)


    for col in cat_cols:
        features = features.replace(col,f"C({col})")



    formula = target+'~'+features #target~predictors

    model = smf.ols(formula=formula, data=df).fit()
    
    if show_summary:
        display(model.summary())

    return model

## diagnostic function

def diagnose_model(model):
    """
    Displays the QQplot and residuals of the model.    
    Args:
        model (statsmodels ols): A fit statsmodels ols model.
    
    Returns:
        fig (Figure): Figure object for output figure
        ax (list): List of axes for subplots. 
    """
    
    import matplotlib.pyplot as plt
    import statsmodels.api as sms
    import statsmodels.formula.api as smf
    import scipy.stats as stats
    
    resids = model.resid
    
    fig,ax = plt.subplots(ncols=2,figsize=(10,5))
    sms.qqplot(resids, stats.distributions.norm,
              fit=True, line='45',ax=ax[0])
    xs = np.linspace(0,1,len(resids))
    ax[1].scatter(x=xs,y=resids)
    
    return fig,ax 

# def find_outliers_Z(df,col):
#     """Use scipy to calcualte absoliute Z-scores 
#     and return boolean series where True indicates it is an outlier
    
#     Args:
#         df (Frame): DataFrame containing column to analyze
#         col (str): Name of column to test.
        
#     Returns:
#         idx_outliers (Series): series of  True/False for each row in col
        
#     Ex:
#     >> idx_outs = find_outliers(df['bedrooms'])
#     >> df_clean = df.loc[idx_outs==False]"""
#     from scipy import stats
#     import numpy as np


#     col = df[col]
#     z = np.abs(stats.zscore(col))
#     idx_outliers = np.where(z>3,True,False)
#     return idx_outliers


# def find_outliers_IQR(df,col):
#     """
#     Use Tukey's Method of outlier removal AKA InterQuartile-Range Rule
#     and return boolean series where True indicates it is an outlier.
#     - Calculates the range between the 75% and 25% quartiles
#     - Outliers fall outside upper and lower limits, using a treshold of  1.5*IQR the 75% and 25% quartiles.
    
#     IQR Range Calculation:    
#         res = df.describe()
#         IQR = res['75%'] -  res['25%']
#         lower_limit = res['25%'] - 1.5*IQR
#         upper_limit = res['75%'] + 1.5*IQR
    
#     Args:
#         df ([type]): [description]
#         col ([type]): [description]
    
#     Returns:
#         [type]: [description]
#     """
#     res = df[col].describe()
#     IQR = res['75%'] -  res['25%']
#     lower_limit = res['25%'] - 1.5*IQR
#     upper_limit = res['75%'] + 1.5*IQR
    
#     idx_goodvals = (df[col]<upper_limit) & (df[col]>lower_limit) 
    
#     return ~idx_goodvals


def vif_ols(df,exclude_col = None, cat_cols = []):
    """
    Performs variance inflation factor analysis on all columns in dataframe
    to identify Multicollinear data. The target column (indicated by exclude_col parameter)
        
    Args:
        df (Frame): data
        exclude_col (str): Column to exclude from OLS model. (for VIF calculations).
        cat_cols (list, optional): List of columns to treat as categories for make_ols_f
    
    Returns:
        res (Framee): DataFrame with results of VIF modeling (VIF and R2 score for each feature)
    """
    
    # let's check each column, build a model and get the r2
    import fsds_100719 as fs
    vif_scores = [['Column','VIF','R2']]

    if exclude_col is not None:
        df = df.drop(exclude_col,axis=1)
        
    for column in df.columns:
        columns_to_use = df.drop(columns=[column]).columns
        target = column
        linreg = make_ols_f(df, target=target, cat_cols=cat_cols,
                            col_list=columns_to_use,show_summary=False)
        R2 = linreg.rsquared
        VIF = 1 / (1 - R2)
    #     print(f"VIF for {column} = {VIF}")
        vif_scores.append([column, VIF, R2])

    res = fs.ds.list2df(vif_scores,index_col='Column')
    res.sort_values('VIF',ascending=False,inplace=True)
    res['use']=res['VIF'] <5
    return res


def scrub_df(data,drop_cols =['id','date','view'],
                       repl_dict = {'sqft_basement':('?','0.0')},
                       recast_dict = {'sqft_basement':'float'},
                       fillna_dict = {'waterfront':0,'yr_renovated':0},
                      verbose=True):
    """Performs entire scrubbing process. Default args are for mod 1 proj housing data.
    
    Performs scrubbing process on the df in the following order:
    1. Drop cols in the drop_cols list
    2. Replace values using repl_dict
    3. Recast dtypes using recast_dict
    4. Fillna using fillna_dict
    
    Args:
        data (Frame): df to scrub
        drop_cols (list): list of col names to drop
        repl_dict (dict): Key=column name, 
                        value= tuple/list with (current value, new value)
        recast_dict(dict): key=column name, value= dtype
        fillna_dict (dict): key=column name, value=value to fill na
        verbose (bool, default=True): 
    """
    import pandas as pd
    from IPython.display import display
    df = data.copy()
    ## Drop cols
    if len(drop_cols)>0:
        for col in drop_cols:
            try:
                df.drop(col, axis=1,inplace=True)
            except Exception as e:
                print(f"[!] Erorr while dropping cols:")
                print(f"\t- Error msg: {e}")
        

    ## Replacing Values
    for col,replace in repl_dict.items():
        df[col] = df[col].replace(replace[0], replace[1])


    ## Recasting datatypes
    for col,dtype in recast_dict.items():
        df[col] = df[col].astype(dtype)
    df.dtypes

    ## Fill Null values / zeros
    for col,val in fillna_dict.items():
        import types
        if isinstance(val, types.FunctionType):
            fill_val = val(df[col])
        else:
            fill_val = val
        
        df[col].fillna(fill_val,inplace=True)
        
    if verbose:
        display(df.head())
        display(df.info())

    return df
    
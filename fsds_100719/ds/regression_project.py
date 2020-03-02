
"""A Collection of functions from ft study group for section 25."""
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sms
    import statsmodels.formula.api as smf
    from IPython.display import display


import scipy.stats as stats
import statsmodels.api as sms
import statsmodels.formula.api as smf


def make_ols_f(df,target,col_list=None,exclude_cols=[],
               cat_cols = [],  show_summary=True,
               diagnose=True,
               return_formula=False):
    """
    Makes statsmodels formula-based regression with options to make categorical columns.    
    Args:
        df (Frame): df with data
        target (str): target column name
        col_list (list, optional): List of predictor columns. Defaults to all except target.
        exclude_cols (list, optional): Columns to remove from col_list. Defaults to [].
        cat_cols (list, optional): Columns to process as categorical using f'C({col})". Defaults to [].
        show_summary (bool, optional): Display model.summary(). Defaults to True.
        diagnose (bool, optional): Plot Q-Q plot & residuals. Defaults to True.
        return_formula (bool, optional): Return formula with model. Defaults to False.
    
    Returns:
        model : statsmodels ols model
        formula : str formula from model, only if return_formula == True
        

    NOTE EXAMPLE WITH MOD 1 PROJECT HOUSING
        model = make_ols_f(df, target='price',
                            cat_cols = ['zipcode','grade'], 
                            exclude_cols= ['id']))
        
        # If return_formula == True
        model, formula = make_ols_f(df, target='price',
                               cat_cols = ['zipcode','grade'],
                               exclude_cols= ['id']))
    """
    import statsmodels.formula.api as smf
    import matplotlib.pyplot as plt
    
    if col_list is None:
        col_list = list(df.drop(target,axis=1).columns)
        
    ## remove exclude cols
    [col_list.remove(ecol) for ecol in exclude_cols if ecol in col_list]

    ## Make rightn side of formula eqn
    features = '+'.join(col_list)

    # ADD C() around categorical cols
    for col in cat_cols:
        features = features.replace(col,f"C({col})")

    ## MAKE FULL FORMULA
    formula = target+'~'+features #target~predictors
    #print(formula)
    
    ## Fit model
    model = smf.ols(formula=formula, data=df).fit()
    
    ## Display summary
    if show_summary:
        display(model.summary())
        
    ## Plot Q-Qplot & model residuals
    if diagnose:
        diagnose_model(model)
        plt.show()

    # Returns formula or just mmodel
    if return_formula:
        return model,formula
    else:
        return model


def diagnose_model(model):
    """
    Plot Q-Q plot and model residuals from statsmodels ols model.
    
    Args:
        model (smf.ols model): statsmodels formula ols 
    
    Returns:
        fig, ax: matplotlib objects
    """
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



def scrub_df(data,drop_cols =[],#['id','date','view'],
                       repl_dict = {},#{'sqft_basement':('?','0.0')},
                       recast_dict = {},#{'sqft_basement':'float'},
                       fillna_dict = {},#{'waterfront':0,'yr_renovated':0},
                      verbose=1):
    """
    Performs scrubbing process on the df in the following order:
    1. Drop cols in the drop_cols list
    2. Replace values using repl_dict
    3. Recast dtypes using recast_dict
    4. Fillna using fillna_dict
    
    Args:
        data (Frame):
        drop_cols (list):
        repl_dict (dict): Key=column name, 
                            value= dict of {to_replace:replace_with}
        recast_dict(dict): 
        fillna_dict(dict): key = column name,
                            val = value to fill with or function to apply 
                            
                            
    NOTE FOR USING WITH MOD 1 PROJECT HOUSING DATA
    scrub_df(data,drop_cols =['id','date','view'],
                       repl_dict = {'sqft_basement':{'?':'0.0'}},
                       recast_dict = {'sqft_basement':'float'},
                       fillna_dict = {'waterfront':0,'yr_renovated':0},
                       verbose=1):
    
    """
    import copy
    import pandas as pd
    import types
    
    df = copy.deepcopy(data)#.copy()
    
    ## Drop cols
    drop_cols = []
    [df.drop(col,axis=1,inplace=True) for col in drop_cols if col in df.columns]


    ## Replacing Values
    for col,replace_vals in repl_dict.items():
            df[col] = df[col].replace(replace_vals)


    
    ## Fill Null values / zeros
    for col,val in fillna_dict.items():
        if isinstance(val, types.FunctionType):
            fill_val = val(df[col])
        else:
            fill_val = val
        
        df[col].fillna(fill_val,inplace=True)
 
    ## Recasting datatypes
    for col,dtype in recast_dict.items():
        df[col] = df[col].astype(dtype)
    df.dtypes
    
    ## display preview
    if verbose>0:
        display(df.head())
    if verbose>1:
        display(df.info())

    return df
    
    


def get_heatmap_mask(corr):
    """"Gets triangle mask for df.corr() for plotting heatmap with sns."""
    import numpy as np
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)]=True
    return mask


def plot_multicollinearity(df,annot=True,fig_size=None):
    """EDA: Plots results from df.corr() in a correlation heat map for multicollinearity.
    Returns fig, ax objects"""
    import seaborn as sns
    sns.set(style="white")
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Compute the correlation matrix
    corr = df.corr()

    # # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # idx = np.triu_indices_from(mask)
    # mask[idx] = True
    mask  = get_heatmap_mask(corr)
    # Set up the matplotlib figure
    if fig_size==None:
        figsize=(16,16)
    else:
        figsize = fig_size

    f, ax = plt.subplots(figsize=(figsize))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=annot, cmap=cmap, center=0,

    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return f, ax
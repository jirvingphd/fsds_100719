
"""A collection of convenient csv urls and sklearn datasets as dataframes."""
def load_data(*args,**kwargs):
    raise Exception('load_data() has been replaced by individual load functions. i.e. fs.datasets.load_boston()')



def read_csv_from_url(url,verbose=False,read_csv_kwds={}):
    """Loading function to load all .csv datasets.
    Args:
        url (str): csv raw link
        verbose (bool): Controls display of loading message and .head()
        read_csv_kwds (dict): dict of commands to feed to pd.read_csv()
    Returns:
        df (DataFrame): the dataset("""
    import pandas as pd
    from IPython.display import display
    ## Load and return dataset
    # if verbose: 
        # print(f"[i] Loading {dataset} from url:\n{url}")
    if read_csv_kwds is not None:
        df = pd.read_csv(url,**read_csv_kwds)
    else:
        df = pd.read_csv(url)
    if verbose:
        display(df.head())
    return df


def load_heroes_info(verbose=False,read_csv_kwds={}):
    url = 'https://raw.githubusercontent.com/jirvingphd/dsc-data-cleaning-project-online-ds-ft-100719/master/heroes_information.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)
    

def load_heroes_powers(verbose=False,read_csv_kwds={}):
    url = "https://raw.githubusercontent.com/learn-co-students/dsc-data-cleaning-project-online-ds-ft-100719/master/super_hero_powers.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)

def load_titanic(verbose=False,read_csv_kwds={}):
    url ="https://raw.githubusercontent.com/jirvingphd/dsc-dealing-missing-data-lab-online-ds-ft-100719/master/titanic.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_mod1_proj(verbose=False,read_csv_kwds={}):
    url = "https://raw.githubusercontent.com/learn-co-students/dsc-v2-mod1-final-project-online-ds-ft-100719/master/kc_house_data.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)
        

def load_population(verbose=False,read_csv_kwds={}):
    url = "https://raw.githubusercontent.com/learn-co-students/dsc-subplots-and-enumeration-lab-online-ds-ft-100719/master/population.csv"
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)

def load_autompg(verbose=True,read_csv_kwds={}):
    
    if verbose:
        print('[i] Source url with details: https://www.kaggle.com/uciml/autompg-dataset')
    
    url = 'https://raw.githubusercontent.com/jirvingphd/dsc-dealing-with-categorical-variables-online-ds-ft-100719/master/auto-mpg.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)




def load_boston(verbose=False):
    
    ## Load Sklearn Datasets
    from sklearn import datasets
    import pandas as pd 
    
    if verbose:
        print("[i] Loading boston housing dataset from sklearn.datasets")
    ## load data dict
    data_dict =  datasets.load_boston()
    # load features
    df_features = pd.DataFrame(data_dict['data'],columns=data_dict['feature_names'])
    # load targets]
    df_features['price'] =data_dict['target']
    
    # set output df
    df = df_features
    if verbose:
        print(data_dict['DESCR'])
    
    return df 

def load_iris(verbose=False):
    from sklearn import datasets
    import pandas as pd
    if verbose:
        print('[i] Loading iris datset from sklearn.datasets')
    data_dict =  datasets.load_iris()
    
    # Get dataframe
    df_features = pd.DataFrame(data_dict['data'],columns=data_dict['feature_names'])
    df_features['target'] = data_dict['target']


    # Get mapper for target names
    iris_map = dict(zip( 
        list(set(data_dict['target'])),
        data_dict['target_names'])
                )
    df_features['target_name']=df_features['target'].map(iris_map)
    df = df_features
    if verbose:
        print(data_dict['DESCR'])   
    return df


def load_height_weight(verbose=False,read_csv_kwds={}):
    """Loads height vs weight dataset"""
    url='https://raw.githubusercontent.com/jirvingphd/dsc-probability-density-function-online-ds-ft-100719/master/weight-height.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_iowa_prisoners(verbose=False,vers='raw',read_csv_kwds={}):
    import pandas as pd
    if 'raw' in vers:
        url ='https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/datasets/FULL_3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.csv'
    else:
        url = 'https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/datasets/Iowa_Prisoners_Renamed_Columns_fsds_100719.csv'
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)
    #pd.set_option('display.precision',3)
    return df

def load_height_by_country(verbose=False,read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/height_by_country_age18.csv'
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)

    if verbose:
        source="http://ncdrisc.org/data-downloads-height.html"
        print(f'Source of dataset: {source}')
        
    return df

def load_yields(verbose=False,version='full',read_csv_kwds=dict(sep=r'\s+', index_col=0)):
    """Loads dataset from Polynomial Regression readme"""
    
    if version=='full':
        url = 'https://raw.githubusercontent.com/jirvingphd/dsc-bias-variance-trade-off-online-ds-pt-100719/master/yield2.csv'
    else:
        url='https://raw.githubusercontent.com/jirvingphd/dsc-polynomial-regression-online-ds-pt-100719/master/yield.csv'

    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)
    
    return df


### TIME SERIES

# baltimore_crime ="https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/BPD_Part_1_Victim_Based_Crime_Data.csv"
# std_rates = "https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/STD%20Cases.csv"
# no_sex_xlsx = "https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/Americans%20Sex%20Frequency.xlsx"

# learn_passengers="https://raw.githubusercontent.com/learn-co-students/dsc-removing-trends-lab-online-ds-ft-100719/master/passengers.csv"

def load_ts_baltimore_crime_full(read_csv_kwds={}):
    url ="https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/BPD_Part_1_Victim_Based_Crime_Data.csv"
    return  read_csv_from_url(url, verbose=False,read_csv_kwds=read_csv_kwds)

### TIME SERIES
def load_ts_baltimore_crime_counts(read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/baltimore_crime_counts_2014-2019.csv"
    return  read_csv_from_url(url, verbose=False,read_csv_kwds=read_csv_kwds)


def load_ts_mintemp(verbose=False,read_csv_kwds={}):
    """Loads min_temp.csv from """
    if verbose:
        print("From Introduction to Time Series")
    url='https://raw.githubusercontent.com/jirvingphd/dsc-introduction-to-time-series-online-ds-ft-100719/master/min_temp.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_ts_nyse_monthly(verbose=False,read_csv_kwds={}):
    """Loads NYSE_.csv from """
    if verbose:
        print("From Introduction to Time Series")
    url='https://raw.githubusercontent.com/jirvingphd/dsc-introduction-to-time-series-online-ds-ft-100719/master/NYSE_monthly.csv'
    return  read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)


def load_ts_exch_rates(verbose=False,read_csv_kwds={}):
    # if verbose:
    url="https://raw.githubusercontent.com/jirvingphd/dsc-basic-time-series-models-online-ds-ft-100719/master/exch_rates.csv"
    return read_csv_from_url(url, verbose=verbose, read_csv_kwds=read_csv_kwds)


def load_ts_google_trends(read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/dsc-corr-autocorr-in-time-series-online-ds-ft-100719/master/google_trends.csv'
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)


def load_ts_winning_400m(read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/dsc-arma-models-lab-online-ds-ft-100719/master/winning_400m.csv"
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)


def load_ts_std_cases(read_csv_kwds={}):
    url = 'https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/STD%20Cases.csv'
    return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)

def load_ts_american_sex_frequency(read_csv_kwds={}):
    url = 'https://raw.githubusercontent.com/jirvingphd/fsds_100719/master/fsds_100719/data/Americans%20Sex%20Frequency.xlsx'
    import pandas as pd
    
    return pd.read_excel(url,**read_csv_kwds)
    # return read_csv_from_url(url,verbose=False, read_csv_kwds=read_csv_kwds)

# def load_ts_co2(read_csv_kwds={}):
#     import statsmodels.api as sm
#     df = sm.datasets.co2.load()
#     return df


def load_AB_multiple_choice(verbose=False,read_csv_kwds={}):
    url='https://raw.githubusercontent.com/jirvingphd/dsc-in-depth-ab-testing-lab-online-ds-pt-100719/master/multipleChoiceResponses_cleaned.csv'
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)

    if verbose:
        from IPython.display import display
        display(df.head())
        
    return df

def load_AB_homepage_actions(verbose=False,read_csv_kwds={}):
    url="https://raw.githubusercontent.com/jirvingphd/dsc-website-ab-testing-lab-online-ds-pt-100719/master/homepage_actions.csv"
    df = read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)#pd.read_csv(url_iowa_raw,index_col=0)

    if verbose:
        from IPython.display import display
        display(df.head())
        
    return df



def load_stock_df(**kwargs):
    import pandas as pd
    url ='https://raw.githubusercontent.com/jirvingphd/capstone-project-using-trumps-tweets-to-predict-stock-market/master/data/SP500_1min_01_23_2020_full.xlsx'
    stock_df = pd.read_excel(url,**kwargs)
    return stock_df
    

def load_heart_disease(verbose=False,read_csv_kwds={}):
    import pandas as pd
    url = "https://raw.githubusercontent.com/jirvingphd/dsc-gaussian-naive-bayes-lab-online-ds-pt-100719/solution/heart.csv"
    return read_csv_from_url(url, verbose=verbose,read_csv_kwds=read_csv_kwds)
    

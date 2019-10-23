"""A collection of convenient csv urls for dataframe loading"""
"""A collection of convenient csv urls for dataframe loading"""
def load(dataset, verbose=False, read_csv_kwds=None):
    """
    Loads a DataFrame of the requested dataset from Learn.co lessons
    or from sklearn.datasets. (see args for options) 
    
    Args:
        dataset (str): Name of dataset to load.
            Options are: 
                # FROM LESSONS
                - 'heroes_info'
                - 'heroes_powers'
                - 'titanic'
                - 'mod1_kc_housing'
                - 'population'
                
                ## FROM SKLEARN.DATASETS
                - 'boston'
                - 'iris'
        verbose (bool): If true, print url or dataset description (sklearn)
        read_csv_kwds(dict): Keywords to pass into pandas when reading csvs.
        
    Returns:
        df
    """
    import pandas as pd
    from sklearn import datasets
    
    def read_csv_from_url(url,dataset=dataset,verbose=verbose,read_csv_kwds=read_csv_kwds):
        ## Load and return dataset
        if verbose: 
            print(f"[i] Loading {dataset} from url:\n{url}")
        if read_csv_kwds is not None:
            df = pd.read_csv(url,**read_csv_kwds)
        else:
            df = pd.read_csv(url)
        return df


    url=[]
    ## Load datasets from urls
    if 'heroes_info' in dataset:
        url = 'https://raw.githubusercontent.com/jirvingphd/dsc-data-cleaning-project-online-ds-ft-100719/master/heroes_information.csv'
        df = read_csv_from_url(url)
    elif 'heroes_powers' in dataset:
        url = "https://raw.githubusercontent.com/learn-co-students/dsc-data-cleaning-project-online-ds-ft-100719/master/super_hero_powers.csv"
        df =read_csv_from_url(url)
        
    elif 'titanic' in dataset:
        url ="https://raw.githubusercontent.com/jirvingphd/dsc-dealing-missing-data-lab-online-ds-ft-100719/master/titanic.csv"
        df = read_csv_from_url(url)
        
    elif 'mod1_kc_housing' in dataset:
        url = "https://raw.githubusercontent.com/learn-co-students/dsc-v2-mod1-final-project-online-ds-ft-100719/master/kc_house_data.csv"
        df = read_csv_from_url(url)
        
    elif 'population' in dataset:
        url = "https://raw.githubusercontent.com/learn-co-students/dsc-subplots-and-enumeration-lab-online-ds-ft-100719/master/population.csv"
        df  = read_csv_from_url(url)
    
    

        
    ## Load Sklearn Datasets
    elif 'boston' in dataset:
        
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
            
    elif 'iris' in dataset:
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
    else:
        msg = f"Dataset '{dataset}' not found."
        raise Exception(msg)
    
    return df
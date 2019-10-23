"""A collection of convenient csv urls for dataframe loading"""
"""A collection of convenient csv urls for dataframe loading"""
def load(dataset, verbose=False, read_csv_kwds=None):
    """
    Loads a DataFrame of the requested dataset. 
    
    Args:
        dataset (str): Name of dataset to load.
        Options are: 
            - 'heroes_info'
            - 'heroes_powers'
            - 'boston'
            - 'iris'
    """
    import pandas as pd
    from sklearn import datasets
    
    def read_csv_from_url(url,dataset=dataset,read_csv_kwds=read_csv_kwds):
        ## Load and return dataset
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
    

        
    ## Load Sklearn Datasets
    elif 'boston' in dataset:
        
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
        print('Loading iris datset from sklearn.datasets')
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
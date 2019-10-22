
from sklearn.model_selection._split import _BaseKFold
class BlockTimeSeriesSplit(_BaseKFold): #sklearn.model_selection.TimeSeriesSplit):
    """A variant of sklearn.model_selection.TimeSeriesSplit that keeps train_size and test_size
    constant across folds.
    Requires n_splits,train_size,test_size. train_size/test_size can be integer indices or float ratios """
    def __init__(self, n_splits=5,train_size=None, test_size=None, step_size=None, method='sliding'):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        if 'sliding' in method or 'normal' in method:
            self.method = method
        else:
            raise  Exception("Method may only be 'normal' or 'sliding'")

    def split(self,X,y=None, groups=None):
        import numpy as np
        import math
        method = self.method
        ## Get n_samples, trian_size, test_size, step_size
        n_samples = len(X)
        test_size = self.test_size
        train_size =self.train_size


        ## If train size and test sze are ratios, calculate number of indices
        if train_size<1.0:
            train_size = math.floor(n_samples*train_size)

        if test_size <1.0:
            test_size = math.floor(n_samples*test_size)

        ## Save the sizes (all in integer form)
        self._train_size = train_size
        self._test_size = test_size

        ## calcualte and save k_fold_size
        k_fold_size = self._test_size + self._train_size
        self._k_fold_size = k_fold_size



        indices = np.arange(n_samples)

        ## Verify there is enough data to have non-overlapping k_folds
        if method=='normal':
            import warnings
            if n_samples // self._k_fold_size <self.n_splits:
                warnings.warn('The train and test sizes are too big for n_splits using method="normal"\n\
                switching to method="sliding"')
                method='sliding'
                self.method='sliding'



        if method=='normal':

            margin = 0
            for i in range(self.n_splits):

                start = i * k_fold_size
                stop = start+k_fold_size

                ## change mid to match my own needs
                mid = int(start+self._train_size)
                yield indices[start: mid], indices[mid + margin: stop]


        elif method=='sliding':

            step_size = self.step_size
            if step_size is None: ## if no step_size, calculate one
                ## DETERMINE STEP_SIZE
                last_possible_start = n_samples-self._k_fold_size #index[-1]-k_fold_size)\
                step_range =  range(last_possible_start)
                step_size = len(step_range)//self.n_splits
            self._step_size = step_size


            for i in range(self.n_splits):
                if i==0:
                    start = 0
                else:
                    start = prior_start+self._step_size #(i * step_size)

                stop =  start+k_fold_size
                ## change mid to match my own needs
                mid = int(start+self._train_size)
                prior_start = start
                yield indices[start: mid], indices[mid: stop]



def undersample_df_to_match_classes(df,class_column='delta_price_class', class_values_to_keep=None,verbose=1):
    """Resamples (undersamples) input df so that the classes in class_column have equal number of occruances.
    If class_values_to_keep is None: uses all classes. """
    import pandas as pd
    import numpy as np

    ##  Get value counts and classes
    class_counts = df[class_column].value_counts()
    classes = list(class_counts.index)

    if verbose>0:
        print('Initial Class Value Counts:')
        print('%: ',class_counts/len(df))

    ## use all classes if None
    if class_values_to_keep is None:
        class_values_to_keep = classes


    ## save each group's indices in dict
    class_dict = {}
    for curr_class in classes:

        if curr_class in class_values_to_keep:
            class_dict[curr_class] = {}

            idx = df.loc[df[class_column]==curr_class].index

            class_dict[curr_class]['idx'] = idx
            class_dict[curr_class]['count'] = len(idx)
        else:
            continue


    ## determine which class count to match
    counts = [class_dict[k]['count'] for k in class_dict.keys()]
    # get number of samples to match
    count_to_match = np.min(counts)

    if len(np.unique(counts))==1:
        raise Exception('Classes are already balanced')

    # dict_resample = {}
    df_sampled = pd.DataFrame()
    for k,v in class_dict.items():
        temp_df = df.loc[class_dict[k]['idx']]
        temp_df =  temp_df.sample(n=count_to_match)
        # dict_resample[k] = temp_df
        df_sampled =pd.concat([df_sampled,temp_df],axis=0)

    ## sort index of final
    df_sampled.sort_index(ascending=False, inplace=True)

    # print(df_sampled[class_column].value_counts())

    if verbose>0:
        check_class_balance(df_sampled, col=class_column)
        # class_counts = [class_column].value_counts()

        # print('Final Class Value Counts:')
        # print('%: ',class_counts/len(df))

    return df_sampled




def find_null_idx(df,column=None):
    """returns the indices of null values found in the series/column.
    if df is a dataframe and column is none, it returns a dictionary
    with the column names as a value and  null_idx for each column as the values.
    Example Usage:
    1)
    >> null_idx = get_null_idx(series)
    >> series_null_removed = series[null_idx]
    2)
    >> null_dict = get_null_idx()
    """
    import pandas as pd
    import numpy as np
    idx_null = []
    # Raise an error if df is a series and a column name is given
    if isinstance(df, pd.Series) and column is not None:
        raise Exception('If passing a series, column must be None')
    # else if its a series, get its idx_null
    elif isinstance(df, pd.Series):
        series = df
        idx_null = series.loc[series.isna()==True].index

    # else if its a dataframe and column is a string:
    elif isinstance(df,pd.DataFrame) and isinstance(column,str):
            series=df[column]
            idx_null = series.loc[series.isna()==True].index

    # else if its a dataframe
    elif isinstance(df, pd.DataFrame):
        idx_null = {}

        # if no column name given, use all columns as col_list
        if column is None:
            col_list =  df.columns
        # else use input column as col_list
        else:
            col_list = column

        ## for each column, get its null idx and add to dictioanry
        for col in col_list:
            series = df[col]
            idx_null[col] = series.loc[series.isna()==True].index
    else:
        raise Exception('Input df must be a pandas DataFrame or Series.')
    ## return the index or dictionary idx_null
    return idx_null




def check_class_balance(df,col ='delta_price_class_int',note='',
                        as_percent=True, as_raw=True):
    import numpy as np
    dashes = '---'*20
    print(dashes)
    print(f'CLASS VALUE COUNTS FOR COL "{col}":')
    print(dashes)
    # print(f'Class Value Counts (col: {col}) {note}\n')

    ## Check for class value counts to see if resampling/balancing is needed
    class_counts = df[col].value_counts()

    if as_percent:
        print('- Classes (%):')
        print(np.round(class_counts/len(df)*100,2))
    # if as_percent and as_raw:
    #     # print('\n')
    if as_raw:
        print('- Class Counts:')
        print(class_counts)
    print('---\n')

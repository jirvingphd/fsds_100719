# from ..ds import *
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
# import scipy.stats as sts
# from IPython.display import display
# from sklearn.model_selection._split import _BaseKFold
# class BlockTimeSeriesSplit(_BaseKFold): #sklearn.model_selection.TimeSeriesSplit):
#     """A variant of sklearn.model_selection.TimeSeriesSplit that keeps train_size and test_size
#     constant across folds.
#     Requires n_splits,train_size,test_size. train_size/test_size can be integer indices or float ratios """
#     def __init__(self, n_splits=5,train_size=None, test_size=None, step_size=None, method='sliding'):
#         super().__init__(n_splits, shuffle=False, random_state=None)
#         self.train_size = train_size
#         self.test_size = test_size
#         self.step_size = step_size
#         if 'sliding' in method or 'normal' in method:
#             self.method = method
#         else:
#             raise  Exception("Method may only be 'normal' or 'sliding'")

#     def split(self,X,y=None, groups=None):
#         import numpy as np
#         import math
#         method = self.method
#         ## Get n_samples, trian_size, test_size, step_size
#         n_samples = len(X)
#         test_size = self.test_size
#         train_size =self.train_size


#         ## If train size and test sze are ratios, calculate number of indices
#         if train_size<1.0:
#             train_size = math.floor(n_samples*train_size)

#         if test_size <1.0:
#             test_size = math.floor(n_samples*test_size)

#         ## Save the sizes (all in integer form)
#         self._train_size = train_size
#         self._test_size = test_size

#         ## calcualte and save k_fold_size
#         k_fold_size = self._test_size + self._train_size
#         self._k_fold_size = k_fold_size



#         indices = np.arange(n_samples)

#         ## Verify there is enough data to have non-overlapping k_folds
#         if method=='normal':
#             import warnings
#             if n_samples // self._k_fold_size <self.n_splits:
#                 warnings.warn('The train and test sizes are too big for n_splits using method="normal"\n\
#                 switching to method="sliding"')
#                 method='sliding'
#                 self.method='sliding'



#         if method=='normal':

#             margin = 0
#             for i in range(self.n_splits):

#                 start = i * k_fold_size
#                 stop = start+k_fold_size

#                 ## change mid to match my own needs
#                 mid = int(start+self._train_size)
#                 yield indices[start: mid], indices[mid + margin: stop]


#         elif method=='sliding':

#             step_size = self.step_size
#             if step_size is None: ## if no step_size, calculate one
#                 ## DETERMINE STEP_SIZE
#                 last_possible_start = n_samples-self._k_fold_size #index[-1]-k_fold_size)\
#                 step_range =  range(last_possible_start)
#                 step_size = len(step_range)//self.n_splits
#             self._step_size = step_size


#             for i in range(self.n_splits):
#                 if i==0:
#                     start = 0
#                 else:
#                     start = self._prior_start+self._step_size #(i * step_size)

#                 stop =  start+k_fold_size
#                 ## change mid to match my own needs
#                 mid = int(start+self._train_size)
#                 self._prior_start = start
#                 yield indices[start: mid], indices[mid: stop]



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


#####
class LabelLibrary():
    """A Multi-column version of sklearn LabelEncoder, which fits a LabelEncoder
   to each column of a df and stores it in the index dictionary where
   .index[keyword=colname] returns the fit encoder object for that column.

   Example:
   lib =LabelLibrary()

   # Be default, lib will fit all columns.
   lib.fit(df)
   # Can also specify columns
   lib.fit(df,columns=['A','B'])

   # Can then transform
   df_coded = lib.transform(df,['A','B'])
   # Can also use fit_transform
   df_coded = lib.fit_transform(df,columns=['A','B'])

   # lib.index contains each col's encoder by col name:
   col_a_classes = lib.index('A').classes_

   """

    def __init__(self):#,df,features):
        """creates self.index and self.encoder"""
        self.index = {}
        from sklearn.preprocessing import LabelEncoder as encoder
        self.encoder=encoder
        # self. = df
        # self.features = features
        
        


    def fit(self,df,columns=None):
        """ Creates an encoder object and fits to each columns.
        Fit encoder is saved in the index dictionary by key=column_name"""
        if columns==None:
            columns = df.columns
#             if any(df.isna()) == True:
#                 num_null = sum(df.isna().sum())
#                 print(f'Replacing {num_null}# of null values with "NaN".')
#                 df.fillna('NaN',inplace=True)


        for col in columns:

            if any(df[col].isna()):
                num_null = df[col].isna().sum()
                Warning(f'For {col}: Replacing {num_null} null values with "NaN".')
                df[col].fillna('NaN',inplace=True)

            # make the encoder
            col_encoder = self.encoder()

            #fit with label encoder
            self.index[col] = col_encoder.fit(df[col])


    def transform(self,df, columns=None):
        import pandas as pd
        df_coded = pd.DataFrame()

        if columns==None:
            df_columns=df.columns
            columns = df_columns
        else:
            df_columns = df.columns


        for dfcol in df_columns:
            if dfcol in columns:
                fit_enc = self.index[dfcol]
                df_coded[dfcol] = fit_enc.transform(df[dfcol])
            else:
                df_coded[dfcol] = df[dfcol]
        return df_coded

    def fit_transform(self,df,columns=None):
        self.fit(df,columns)
        df_coded = self.transform(df,columns)
        return df_coded

    def inverse_transform(self,df,columns = None):
        import pandas as pd

        df_reverted = pd.DataFrame()

        if columns==None:
            columns=df.columns

        for col in columns:
            fit_enc = self.index[col]
            df_reverted[col] = fit_enc.inverse_transform(df[col])
        return df_reverted



#################### GENERAL HELPER FUNCTIONS #####################
def is_var(name):
    x=[]
    try: eval(name)
    except NameError: x = None

    if x is None:
        return False
    else:
        return True



def print_docstring_template(style='google',object_type='function',show_url=False, to_clipboard=False):
    """ Prints out docstring template for that is copy/paste ready.
    May choose 'google' or 'numpy' style docstrings and templates
    are available different types ('class','function','module_function').
    
    Args:
        style (str, optional): Which docstring style to return. Options are 'google' and 'numpy'. Defaults to 'google'.
        object_type (str, optional): Which type of template to return. Options are 'class','function','module_function'. Defaults to 'function'.
        show_url (bool, optional): Whether to display link to reference page for style-type. Defaults to False.
    
    Returns:
        [type]: [description]
    """
    template_dict ={}
    template_dict['numpy']={}
    template_dict['numpy']['url']='https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy'
    template_dict['numpy']['function'] = '''
    def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    '''
    template_dict['numpy']['module_function'] = '''
    def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Parameters`` section.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    If *args or **kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name : type
            description

            The description may span multiple lines. Following lines
            should be indented to match the first line of the description.
            The ": type" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : :obj:`str`, optional
        The second parameter.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    bool
        True if successful, False otherwise.

        The return type is not optional. The ``Returns`` section may span
        multiple lines and paragraphs. Following lines should be indented to
        match the first line of the description.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    """'''
    
    template_dict['numpy']['class'] = '''
    class ExampleClass(object):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    attr1 : str
        Description of `attr1`.
    attr2 : :obj:`int`, optional
        Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1 : str
            Description of `param1`.
        param2 : :obj:`list` of :obj:`str`
            Description of `param2`. Multiple
            lines are supported.
        param3 : :obj:`int`, optional
            Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list of str: Doc comment *before* attribute, with type specified
        self.attr4 = ["attr4"]

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

        @property
        def readonly_property(self):
            """str: Properties should be documented in their getter method."""
            return "readonly_property"

        @property
        def readwrite_property(self):
            """:obj:`list` of :obj:`str`: Properties with both a getter and setter
            should only be documented in their getter method.

            If the setter method contains notable behavior, it should be
            mentioned here.
            """
            return ["readwrite_property"]

        @readwrite_property.setter
        def readwrite_property(self, value):
            value

        def example_method(self, param1, param2):
            """Class methods are similar to regular functions.

            Note
            ----
            Do not include the `self` parameter in the ``Parameters`` section.

            Parameters
            ----------
            param1
                The first parameter.
            param2
                The second parameter.

            Returns
            -------
            bool
                True if successful, False otherwise.

            """
            return True

        def __special__(self):
            """By default special members with docstrings are not included.

            Special members are any methods or attributes that start with and
            end with a double underscore. Any special member with a docstring
            will be included in the output, if
            ``napoleon_include_special_with_doc`` is set to True.

            This behavior can be enabled by changing the following setting in
            Sphinx's conf.py::

                napoleon_include_special_with_doc = True

            """
            pass

        def __special_without_docstring__(self):
            pass

        def _private(self):
            """By default private members are not included.

            Private members are any methods or attributes that start with an
            underscore and are *not* special. By default they are not included
            in the output.

            This behavior can be changed such that private members *are* included
            by changing the following setting in Sphinx's conf.py::

                napoleon_include_private_with_doc = True

            """
            pass

        def _private_without_docstring(self):
            pass
        '''
            
       
    template_dict ={}
    template_dict['google']={}
    template_dict['google']['url']="https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google"
    template_dict['google']['function'] = '''
    Example function with types documented in the docstring.

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    '''

    template_dict['google']['module_function'] = r'''
    def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    if param1 == param2:
        raise ValueError('param1 may not be equal to param2')
    return True
    '''
    
    
    template_dict['google']['class'] = '''
    class ExampleClass(object):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list of str: Doc comment *before* attribute, with type specified
        self.attr4 = ['attr4']

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self):
        """str: Properties should be documented in their getter method."""
        return 'readonly_property'

    @property
    def readwrite_property(self):
        """:obj:`list` of :obj:`str`: Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ['readwrite_property']

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    def example_method(self, param1, param2):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return True

    def __special__(self):
        """By default special members with docstrings are not included.

        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.

        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::

            napoleon_include_special_with_doc = True

        """
        pass

    def __special_without_docstring__(self):
        pass

    def _private(self):
        """By default private members are not included.

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self):
        pass
    '''
    
    
    ### Select output
    style_dict = template_dict[style]
    print_template = style_dict[object_type]
    url = style_dict['url']
    
    if show_url:
        print(f'Template source for {style} style docstrings: {url} ')

    if to_clipboard==False:        
        print(print_template)
    else:
        import pyperclip
        print('Template copied to clipboard.')
        return pyperclip.copy(print_template)
    
    
    
    """A collection of function to change the aesthetics of Pandas DataFrames using CSS, html, and pandas styling."""
# from IPython.display import HTML
# import pandas as pd
def hover(hover_color="gold"):
    """DataFrame Styler: Called by highlight to highlight row below cursor.
        Changes html background color.

        Parameters:

        hover_Color
    """
    from IPython.display import HTML
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])


def highlight(df,hover_color="gold"):
    """DataFrame Styler:
        Highlight row when hovering.
        Accept and valid CSS colorname as hover_color.
    """
    styles = [
        hover(hover_color),
        dict(selector="th", props=[("font-size", "115%"),
                                   ("text-align", "center")]),
        dict(selector="caption", props=[("caption-side", "bottom")])
    ]
    html = (df.style.set_table_styles(styles)
              .set_caption("Hover to highlight."))
    return html


def color_true_green(val):
    """DataFrame Styler:
    Changes text color to green if value is True
    Ex: style_df = df.style.applymap(color_true_green)
        style_df #to display"""
    color='green' if val==True else 'black'
    return f'color: {color}'

# Style dataframe for easy visualization


def color_scale_columns(df,matplotlib_cmap = "Greens",subset=None,):
    """DataFrame Styler:
    Takes a df, any valid matplotlib colormap column names
    (matplotlib.org/tutorials/colors/colormaps.html) and
    returns a dataframe with a gradient colormap applied to column values.

    Example:
    df_styled = color_scale_columns(df,cmap = "YlGn",subset=['Columns','to','color'])

    Parameters:
    -----------
        df:
            DataFrame containing columns to style.
    subset:
         Names of columns to color-code.
    cmap:
        Any matplotlib colormap.
        https://matplotlib.org/tutorials/colors/colormaps.html

    Returns:
    ----------
        df_style:
            styled dataframe.

    """
    from IPython.display import display
    import seaborn as sns
    cm = matplotlib_cmap
    #     cm = sns.light_palette("green", as_cmap=True)
    df_style = df.style.background_gradient(cmap=cm,subset=subset)#,low=results.min(),high=results.max())
    # Display styled dataframe
#     display(df_style)
    return df_style

def make_CSS(show=False):
    """Makes default CSS for html_on function."""
    CSS="""
        table td{
        text-align: center;
        }
        table th{
        background-color: black;
        color: white;
        font-family:serif;
        font-size:1.2em;
        }
        table td{
        font-size:1.05em;
        font-weight:75;
        }
        table td, th{
        text-align: center;
        }
        table caption{
        text-align: center;
        font-size:1.2em;
        color: black;
        font-weight: bold;
        font-style: italic
        }
    """
    if show==True:
        from pprint import pprint
        pprint(CSS)
    return CSS



# -*- coding: utf-8 -*-
"""A collection of function to change the aesthetics of Pandas DataFrames using CSS, html, and pandas styling."""
# from IPython.display import HTML
# import pandas as pd


# def hover(hover_color="gold"):
#     """DataFrame Styler: Called by highlight to highlight row below cursor.
#         Changes html background color.

#         Parameters:

#         hover_Color
#     """
#     from IPython.display import HTML
#     return dict(selector="tr:hover",
#                 props=[("background-color", "%s" % hover_color)])


# def highlight(df,hover_color="gold"):
#     """DataFrame Styler:
#         Highlight row when hovering.
#         Accept and valid CSS colorname as hover_color.
#     """
#     styles = [
#         hover(hover_color),
#         dict(selector="th", props=[("font-size", "115%"),
#                                    ("text-align", "center")]),
#         dict(selector="caption", props=[("caption-side", "bottom")])
#     ]
#     html = (df.style.set_table_styles(styles)
#               .set_caption("Hover to highlight."))
#     return html


# def color_true_green(val):
#     """DataFrame Styler:
#     Changes text color to green if value is True
#     Ex: style_df = df.style.applymap(color_true_green)
#         style_df #to display"""
#     color='green' if val==True else 'black'
#     return f'color: {color}'

# # Style dataframe for easy visualization


# def color_scale_columns(df,matplotlib_cmap = "Greens",subset=None,):
#     """DataFrame Styler:
#     Takes a df, any valid matplotlib colormap column names
#     (matplotlib.org/tutorials/colors/colormaps.html) and
#     returns a dataframe with a gradient colormap applied to column values.

#     Example:
#     df_styled = color_scale_columns(df,cmap = "YlGn",subset=['Columns','to','color'])

#     Parameters:
#     -----------
#         df:
#             DataFrame containing columns to style.
#     subset:
#          Names of columns to color-code.
#     cmap:
#         Any matplotlib colormap.
#         https://matplotlib.org/tutorials/colors/colormaps.html

#     Returns:
#     ----------
#         df_style:
#             styled dataframe.

#     """
#     from IPython.display import display
#     import seaborn as sns
#     cm = matplotlib_cmap
#     #     cm = sns.light_palette("green", as_cmap=True)
#     df_style = df.style.background_gradient(cmap=cm,subset=subset)#,low=results.min(),high=results.max())
#     # Display styled dataframe
# #     display(df_style)
#     return df_style

# def make_CSS(show=False):
#     CSS="""
#         table td{
#         text-align: center;
#         }
#         table th{
#         background-color: black;
#         color: white;
#         font-family:serif;
#         font-size:1.2em;
#         }
#         table td{
#         font-size:1.05em;
#         font-weight:75;
#         }
#         table td, th{
#         text-align: center;
#         }
#         table caption{
#         text-align: center;
#         font-size:1.2em;
#         color: black;
#         font-weight: bold;
#         font-style: italic
#         }
#     """
#     if show==True:
#         from pprint import pprint
#         pprint(CSS)
#     return CSS


# CSS="""
#     .{
#     text-align: center;
#     }
#     th{
#     background-color: black;
#     color: white;
#     font-family:serif;
#     font-size:1.2em;
#     }
#     td{
#     font-size:1.05em;
#     font-weight:75;
#     }
#     td, th{
#     text-align: center;
#     }
#     caption{
#     text-align: center;
#     font-size:1.2em;
#     color: black;
#     font-weight: bold;
#     font-style: italic
#     }
# """
# HTML(f"<style>{CSS}</style>")
# CSS = """
# table.dataframe td, table.dataframe th { /* This is for the borders for columns)*/
#     border: 2px solid black
#     border-collapse:collapse;
#     text-align:center;
# }
# table.dataframe th {
#     /*padding:1em 1em;*/
#     background-color: #000000;
#     color: #ffffff;
#     text-align: center;
#     font-weight: bold;
#     font-size: 12pt
#     font-weight: bold;
#     padding: 0.5em 0.5em;
# }
# table.dataframe td:not(:th){
#     /*border: 1px solid ##e8e8ea;*/
#     /*background-color: ##e8e8ea;*/
#     background-color: gainsboro;
#     text-align: center;
#     vertical-align: middle;
#     font-size:10pt;
#     padding: 0.7em 1em;
#     /*padding: 0.1em 0.1em;*/
# }
# table.dataframe tr:not(:last-child) {
#     border-bottom: 1px solid gainsboro;
# }
# table.dataframe {
#     /*border-collapse: collapse;*/
#     background-color: gainsboro; /* This is alternate rows*/
#     text-align: center;
#     border: 2px solid black;
# }
# table.dataframe th:not(:empty), table.dataframe td{
#     border-right: 1px solid white;
#     text-align: center;
# }
# # """

def html_off():
    from IPython.display import HTML
    return HTML('<style>{}</style>'.format(''))

def html_on(CSS=None, verbose=False):
    """Applies HTML/CSS styling to all dataframes. 'CSS' variable is created by make_CSS() if not supplied.
    Verbose =True will display the default CSS code used. Any valid CSS key: value pair can be passed."""
    from IPython.display import HTML
    if CSS is None:
        CSS = make_CSS()
    if verbose==True:
        from pprint import pprint
        pprint(CSS)

    return HTML("<style>{}</style>".format(CSS))


def display_side_by_side(*args):
    """Display all input dataframes side by side. Also accept captioned styler df object (df_in = df.style.set_caption('caption')
    Modified from Source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side"""
    from IPython.display import display_html
    import pandas
    html_str=''
    for df in args:
        if type(df) == pandas.io.formats.style.Styler:
            html_str+= '&nbsp;'
            html_str+=df.render()
        else:
            html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)




def plot_auc_roc_curve(y_test, y_test_pred):
    """ Takes y_test and y_test_pred from a ML model and uses sklearn roc_curve to plot the AUC-ROC curve."""
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    auc = roc_auc_score(y_test, y_test_pred[:,1])

    FPr, TPr, _  = roc_curve(y_test, y_test_pred[:,1])
    auc()
    plt.plot(FPr, TPr,label=f"AUC for Classifier:\n{round(auc,2)}" )

    plt.plot([0, 1], [0, 1],  lw=2,linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None,
                          print_matrix=True):
    """Check if Normalization Option is Set to True. If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function."""
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    if cmap==None:
        cmap = plt.get_cmap("Blues")

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



## Finding outliers and statistics
# Tukey's method using IQR to eliminate
def detect_outliers(df, n, features):
    """Uses Tukey's method to return outer of interquartile ranges to return indices if outliers in a dataframe.
    Parameters:
    df (DataFrame): DataFrame containing columns of features
    n: default is 0, multiple outlier cutoff

    Returns:
    Index of outliers for .loc

    Examples:
    Outliers_to_drop = detect_outliers(data,2,["col1","col2"]) Returning value
    df.loc[Outliers_to_drop] # Show the outliers rows
    data= data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
"""
    import numpy as np
    import pandas as pd
    # Drop outliers

    outlier_indices = []
    # iterate over features(columns)
    for col in features:

        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        from collections import Counter
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers



# Plots histogram and scatter (vs price) side by side
def plot_hist_scat(df, target=None, figsize=(12,9),fig_style='dark_background',font_dict=None,plot_kwds=None):
    """EDA: Great summary plots of all columns of a df vs target columne.
    Shows distplots and regplots for columns im datamframe vs target.
    Parameters:
        df (DataFrame):
            DataFrame.describe() columns will be plotted.
        target (string):
            Name of column containing target variable.assume first column.
        figsize (tuple):
            Tuple for figsize. Default=(12,9).
        fig_style:
            Figure style to use (in this context, will not change others in notebook).
            Default is 'dark_background'.
        font_dict:
            A keywork dictionry containing values for font properties under the following keys:
            - "fontTitle": font dictioanry for titles
            , fontAxis, fontTicks

    **plot_kwds:
        A kew_word dictionary containing any of the following keys for dictionaries containing
        any valid matplotlib key:value pairs for plotting:
            "hist_kws, kde_kws, line_kws,scatter_kws"
        Accepts any valid matplotlib key:value pairs passed by searborn to matplotlib.
        Subplot 1: hist_kws, kde_kws
        Subplot 2: line_kws,scatter_kws

    Returns:
        fig:
            Figure object.
        ax:
            Subplot axes with format ax[row,col].
            Subplot 1 = ax[0,0]; Subplot 2 = ax[0,1]
   """
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set target as first column if not specified
    if target==None:
        target= df.iloc[:,0]

    ###  DEFINE AESTHETIC CUSTOMIZATIONS  -------------------------------##
    # Checking for user font_dict, if not setting defaults:
    if font_dict == None:
        # Axis Label fonts
        fontTitle = {'fontsize': 16,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontAxis = {'fontsize': 14,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontTicks = {'fontsize': 12,
                   'fontweight':'bold',
                    'fontfamily':'serif'}

    else:

        if 'fontTitle' in font_dict.keys():
            fontTitle = font_dict['fontTitle']
        else:
            fontTitle = {'fontsize': 16, 'fontweight': 'bold','fontfamily':'serif'}

        if 'fontAxis' in font_dict.keys():
            fontAxis = font_dict['fontAxis']
        else:
            fontAxis = {'fontsize': 14,'fontweight': 'bold', 'fontfamily':'serif'}

        if 'fontTicks' in font_dict.keys():
            fontTicks = font_dict['fontTicks']
        else:
            fontTicks = {'fontsize': 12,'fontweight':'bold','fontfamily':'serif'}

    # Checking for user plot_kwds
    if plot_kwds == None:
        hist_kws = {"linewidth": 1, "alpha": 1, "color": 'steelblue','edgecolor':'w','hatch':'\\'}
        kde_kws = {"color": "white", "linewidth": 3, "label": "KDE",'alpha':0.7}
        line_kws={"color":"white","alpha":0.5,"lw":3,"ls":":"}
        scatter_kws={'s': 2, 'alpha': 0.8,'marker':'.','color':'steelblue'}

    else:
        kwds = plot_kwds
        # Define graphing keyword dictionaries for distplot (Subplot 1)
        if 'hist_kws' in kwds.keys():
            hist_kws = kwds['hist_kws']
        else:
            hist_kws = {"linewidth": 1, "alpha": 1, "color": 'steelblue','edgecolor':'w','hatch':'\\'}

        if 'kde_kws' in kwds.keys():
            kde_kws = kwds['kde_kws']
        else:
            kde_kws = {"color": "white", "linewidth": 3, "label": "KDE",'alpha':0.7}

        # Define the kwd dictionaries for scatter and regression line (subplot 2)
        if 'line_kws' in kwds.keys():
            line_kws = kwds['line_kws']
        else:
            line_kws={"color":"white","alpha":0.5,"lw":3,"ls":":"}

        if 'scatter_kws' in kwds.keys():
            scatter_kws = kwds['scatter_kws']
        else:
            scatter_kws={'s': 2, 'alpha': 0.8,'marker':'.','color':'steelblue'}


    with plt.style.context(fig_style):
        # Formatting dollar sign labels
        # fmtPrice = '${x:,.0f}'
        # tickPrice = mtick.StrMethodFormatter(fmtPrice)

        ###  PLOTTING ----------------------------- ------------------------ ##

        # Loop through dataframe to plot
        for column in df.describe():

            # Create figure with subplots for current column
            fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=2)

            ##  SUBPLOT 1 --------------------------------------------------##
            i,j = 0,0
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)


            # Plot distplot on ax[i,j] using hist_kws and kde_kws
            sns.distplot(df[column], norm_hist=True, kde=True,
                         hist_kws = hist_kws, kde_kws = kde_kws,
                         label=column+' histogram', ax=ax[i,j])


            # Set x axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)

            # Get x-ticks, rotate labels, and return
            xticklab1 = ax[i,j].get_xticklabels(which = 'both')
            ax[i,j].set_xticklabels(labels=xticklab1, fontdict=fontTicks, rotation=0)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())


            # Set y-label
            ax[i,j].set_ylabel('Density',fontdict=fontAxis)
            yticklab1=ax[i,j].get_yticklabels(which='both')
            ax[i,j].set_yticklabels(labels=yticklab1,fontdict=fontTicks)
            ax[i,j].yaxis.set_major_formatter(mtick.ScalarFormatter())


            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')




            ##  SUBPLOT 2-------------------------------------------------- ##
            i,j = 0,1
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)



            # Plot regplot on ax[i,j] using line_kws and scatter_kws
            sns.regplot(df[column], df[target],
                        line_kws = line_kws,
                        scatter_kws = scatter_kws,
                        ax=ax[i,j])

            # Set x-axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)

             # Get x ticks, rotate labels, and return
            xticklab2=ax[i,j].get_xticklabels(which='both')
            ax[i,j].set_xticklabels(labels=xticklab2,fontdict=fontTicks, rotation=0)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())

            # Set  y-axis label
            ax[i,j].set_ylabel(target.title(),fontdict=fontAxis)

            # Get, set, and format y-axis Price labels
            yticklab = ax[i,j].get_yticklabels()
            ax[i,j].set_yticklabels(yticklab,fontdict=fontTicks)
            ax[i,j].yaxis.set_major_formatter(mtick.ScalarFormatter())

            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')

            ## ---------- Final layout adjustments ----------- ##
            # Deleted unused subplots
            fig.delaxes(ax[1,1])
            fig.delaxes(ax[1,0])

            # Optimizing spatial layout
            fig.tight_layout()
            # figtitle=column+'_dist_regr_plots.png'
            # plt.savefig(figtitle)
    return fig, ax


def big_pandas(user_options=None,verbose=0):
    """Changes the default pandas display setttings to show all columns and all rows.
    User may replace settings with a kwd dictionary matching available options.
    
    Args:
        user_options(dict) :  Pandas size parameters for pd.set_options = {
            'display' : {
                'max_columns' : None,
                'expand_frame_repr':False,
                'max_rows':None,
                'max_info_columns':500,
                'precision' : 4,
            }
    """
    import pandas as pd
    if user_options==None:
        options = {
            'display' : {
                'max_columns' : None,
                'expand_frame_repr':False,
                'max_rows':None,
                'max_info_columns':500,
                'precision' : 4,
            }
        }
    else:
        options = user_options

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+
            if verbose>0:
                print(f'{category}.{op}={value}')
    return options 

def reset_pandas():
    """Resets all pandas options back to default state."""
    import pandas as pd
    return pd.reset_option('all')


def ignore_warnings():
    """Ignores all deprecation warnings (future,and pending categories too)."""
    import warnings
    return warnings.simplefilter(action='ignore', category=(FutureWarning,DeprecationWarning,PendingDeprecationWarning))

def reset_warnings():
    """Restore the default warnings settings"""
    import warnings
    return warnings.simplefilter(action='default', category=(FutureWarning,DeprecationWarning,PendingDeprecationWarning))



# def check_column(panda_obj, columns=None,nlargest='all'):
#     """
#     Prints column name, dataype, # and % of null values, and unique values for the nlargest # of rows (by valuecount_.
#     it will only print results for those columns
#     ************
#     Params:
#     panda_object: pandas DataFrame or Series
#     columns: list containing names of columns (strings)

#     Returns: None
#         prints values only
#     """
#     import numpy as np
#     import pandas as pd
#     # Check for DF vs Series
#     if type(panda_obj)==pd.core.series.Series:
#         series=panda_obj
#         print(f'\n----------------------------\n')
#         print(f"Column: df['{series.name}']':")
#         print(f"dtype: {series.dtype}")
#         print(f"isna: {series.isna().sum()} out of {len(series)} - {round(series.isna().sum()/len(series)*100,3)}%")

#         print(f'\nUnique non-na values:')
#         if nlargest =='all':
#             print(series.value_counts())
#         else:
#             print(series.value_counts().nlargest(nlargest))


#     elif type(panda_obj)==pd.core.frame.DataFrame:
#         df = panda_obj
#         for col_name in df.columns:
#             col = df[col_name]
#             print("\n-----------------------------------------------")
#             print(f"Column: df['{col_name}']':")
#             print(f"dtype: {col.dtypes}")
#             print(f"isna: {col.isna().sum()} out of {len(col)} - {round(col.isna().sum()/len(col)*100,3)}%")

#             print(f'\nUnique non-na values:\nnlargest={nlargest}\n-----------------')
#             if nlargest =='all':
#                 print(col.value_counts())
#             else:
#                 print(col.value_counts().nlargest(nlargest))



# def check_df_for_columns(df, columns=None):

#     """
#     Checks df for presence of columns.

#     args:
#     **********
#     df: pd.DataFrame to find columns in
#     columns: str or list of str. column names
#     """
#     if not columns:
#         print('check_df_for_columns expected to be passed a list of column names.')
#     else:
#         for column in columns:
#             if not column in df.columns:
#                 continue
#             else:
#                 print(f'{column} is a valid column name')
#     pass


# def check_unique(df, columns=None):
#     """
#     Prints unique values for all columns in dataframe. If passed list of columns,
#     it will only print results for those columns
#     8************  >
#     Params:
#     df: pandas DataFrame, or pd.Series
#     columns: list containing names of columns (strings)

#     Returns: None
#         prints values only
#     """
#     from IPython.display import display
#     import pandas as pd
#     # check for columns
# #     if columns is None:
#         # Check if series, even though this is unnecesary because you could simply
#         # Call pd.series.sort_values()
#     if isinstance(df, pd.Series):
#         # display all the value counts
#         nunique = df.nunique()
#         print(f'\n---------------------------\n')
#         print(f"{df.name} Type: {df.dtype}\nNumber unique values: {nunique}")
#         return pd.DataFrame(df.value_counts())

#     else:
#         if columns is None:
#             columns = df.columns

#         for col in columns:
#             nunique = df[col].nunique()
#             unique_df = pd.DataFrame(df[col].value_counts())
#             print(f'\n---------------------------')
#             print(f"\n{col} Type: {df[col].dtype}\nNumber unique values: {nunique}.")
#             display(unique_df)
#         pass


# def check_numeric(df, columns=None, unique_check=False, return_list=False, show_df=False):

#     """
#     Iterates through columns and checks for possible numeric features labeled as objects.
#     Params:
#     ******************
#     df: pandas DataFrame

#     unique_check: bool. (default=True)
#         If true, distplays interactive interface for checking unique values in columns.

#     return_list: bool, (default=False)
#         If True, returns a list of column names with possible numeric types.
#     **********>
#     Returns: dataframe displayed (always), list of column names if return_list=True
#     """
#     # from .bs_ds import list2df
#     from IPython.display import display
#     display_list = [['Column', 'Numeric values','Total Values', 'Percent']]
#     outlist = []
#     # print(f'\n---------------------------------------------------\n')
#     # print(f'# of Identified Numeric Values in "Object" columns:')

#     # Check for user column list
#     columns_to_check = []
#     if columns == None:
#         columns_to_check = df.columns
#     else:
#         columns_to_check = columns
#     # Iterate through columns

#     for col in columns_to_check:

#         # Check for object dtype,
#         if df[col].dtype == 'object':

#             # If object, check for numeric
#             if df[col].str.isnumeric().any():

#                 # If numeric, get counts
#                 vals = df[col].str.isnumeric().sum()
#                 percent = round((df[col].str.isnumeric().sum()/len(df[col]))*100, 2)
#                 display_list.append([col, vals,len(df[col]), percent])
#                 outlist.append(col)

#     list2show = list2df(display_list)
#     list2show.set_index('Column',inplace=True)

#     styled_list2show = list2show.style.set_caption('# of Detected Numeric Values in "Object" columns:')
#     if show_df==True:
#         display(styled_list2show)

#     if unique_check:
#         unique = input("display unique values? (Enter 'y' for all columns, a column name, or 'n' to quit):")

#         while unique != 'n':

#             if unique == 'y':
#                 check_unique(df, outlist)
#                 break

#             elif unique in outlist:
#                 name = [unique]
#                 check_unique(df, name)

#             unique = input('Enter column name or n to quit:')

#     if return_list==True:
#         return styled_list2show, outlist
#     else:
#         return styled_list2show


# def check_null(df, columns=None,show_df=False):
#     """
#     Iterates through columns and checks for null values and displays # and % of column.
#     Params:
#     ******************
#     df: pandas DataFrame

#     columns: list of columns to check
#     **********>
#     Returns: displayed dataframe
#     """
#     from IPython.display import display
#     # from .bs_ds import list2df
#     display_list = [['Column', 'Null values', 'Total Values','Percent']]
#     outlist = []
#     # print(f'\n----------------------------\n')
#     # print(f'# of Identified Null Values:')

#     # Check for user column list
#     columns_to_check = []
#     if columns==None:
#         columns_to_check = df.columns
#     else:
#         columns_to_check = columns
#     # Iterate through columns

#     for col in columns_to_check:

#         # Check for object dtype,
#         # if df[col].dtype == 'object':

#         # If object, check for numeric


#         # If numeric, get counts
#         vals = df[col].isna().sum()
#         percent = round((vals/len(df[col]))*100, 3)
#         display_list.append([col, vals, len(df[col]), percent])
#         outlist.append(col)

#     list2show=list2df(display_list)
#     list2show.set_index('Column',inplace=True)

#     styled_list2show = list2show.style.set_caption('# of Identified Null Values:')
#     if show_df==True:
#         display(styled_list2show)

#     return styled_list2show






# def compare_duplicates(df1, df2, to_drop=True, verbose=True, return_names_list=False):
#     """
#     Compare two dfs for duplicate columns, drop if to_drop=True, useful
#     to us before concatenating when dtypes are different between matching column names
#     and df.drop_duplicates is not an option.
#     Params:
#     --------------------
#     df1, df2 : pandas dataframe suspected of having matching columns
#     to_drop : bool, (default=True)
#         If True will give the option of dropping columns one at a time from either column.
#     verbose: bool (default=True)
#         If True prints column names and types, set to false and return_names list=True
#         if only desire a list of column names and no interactive interface.
#     return_names_list: bool (default=False),
#         If True, will return a list of all duplicate column names.
#     --------------------
#     Returns: List of column names if return_names_list=True, else nothing.
#     """
#     catch = []
#     dropped1 = []
#     dropped2 = []
#     if verbose:
#         print("Column |   df1   |   df2   ")
#         print("*----------------------*")

#     # Loop through columns, inspect for duplicates
#     for col in df1.columns:
#         if col in df2.columns:
#             catch.append(col)

#             if verbose:
#                 print(f"{col}   {df1[col].dtype}   {df2[col].dtype}")

#             # Accept user input and drop columns one by one
#             if to_drop:
#                 choice = input("\nDrop this column? Enter 1. df1, 2. df2 or n for neither")

#                 if choice ==  "1":
#                     df1.drop(columns=col, axis=1, inplace=True)
#                     dropped1.append(col)

#                 elif choice == "2":
#                     df2.drop(columns=col, axis=1, inplace=True)
#                     dropped2.append(col)
#                 else:

#                     continue
#     # Display dropped columns and orignating df
#     if to_drop:
#         if len(dropped1) >= 1:
#             print(f"\nDropped from df1:\n{dropped1}")
#         if len(dropped2) >= 1:
#             print(f"\nDropped from df1:\n{dropped2}")

#     if return_names_list:
#         return catch
#     else:
#         pass


# # ## Dataframes styling
# # def check_column(panda_obj, columns=None,nlargest='all'):
# #     """
# #     Prints column name, dataype, # and % of null values, and unique values for the nlargest # of rows (by valuecount_.
# #     it will only print results for those columns
# #     ************
# #     Params:
# #     panda_object: pandas DataFrame or Series
# #     columns: list containing names of columns (strings)

# #     Returns: None
# #         prints values only
# #     """
# #     import pandas as pd
# #     # Check for DF vs Series
# #     if type(panda_obj)==pd.core.series.Series:
# #         series=panda_obj
# #         print(f'\n----------------------------\n')
# #         print(f"Column: df['{series.name}']':")
# #         print(f"dtype: {series.dtype}")
# #         print(f"isna: {series.isna().sum()} out of {len(series)} - {round(series.isna().sum()/len(series)*100,3)}%")

# #         print(f'\nUnique non-na values:')
# #         if nlargest =='all':
# #             print(series.value_counts())
# #         else:
# #             print(series.value_counts().nlargest(nlargest))


# #     elif type(panda_obj)==pd.core.frame.DataFrame:
# #         df = panda_obj
# #         for col_name in df.columns:
# #             col = df[col_name]
# #             print("\n-----------------------------------------------")
# #             print(f"Column: df['{col_name}']':")
# #             print(f"dtype: {col.dtypes}")
# #             print(f"isna: {col.isna().sum()} out of {len(col)} - {round(col.isna().sum()/len(col)*100,3)}%")

# #             print(f'\nUnique non-na values:\nnlargest={nlargest}\n-----------------')
# #             if nlargest =='all':
# #                 print(col.value_counts())
# #             else:
# #                 print(col.value_counts().nlargest(nlargest))



#     ## DataFrame Creation, Inspection, and Exporting
# def inspect_df(df, n_rows=3, verbose=True):
#     """ EDA:
#     Show all pandas inspection tables.
#     Displays df.head(), df.info(), df.describe().
#     By default also runs check_null and check_numeric to inspect
#     columns for null values and to check string columns to detect
#     numeric values. (If verbose==True)
#     Parameters:
#         df(dataframe):
#             dataframe to inspect
#         n_rows:
#             number of header rows to show (Default=3).
#         verbose:
#             If verbose==True (default), check_null and check_numeric.
#     Ex: inspect_df(df,n_rows=4)
#     """
#     # from bs_ds.bamboo import check_column, check_null, check_numeric, check_unique
#     # from bs_ds.prettypandas import display_side_by_side
#     import pandas as pd
#     from IPython.display import display

#     with pd.option_context("display.max_columns", None ,'display.precision',4):
#         display(df.info()) #, display(df.describe())

#         if verbose == True:

#             df_num = check_numeric(df,unique_check=False, show_df=False)
#             # sdf_num = df_num.style.set_caption('Detected Numeric Values')

#             df_null = check_null(df, show_df=False)
#             # sdf_null = df_null.style.set_caption('Detected Null values')

#             display_side_by_side(df_null, df_num,df.describe())
#         else:
#             display(df.describe())

#         display(df.head(n_rows))





def drop_cols(df, list_of_strings_or_regexp,verbose=0):#,axis=1):
    """EDA: Take a df, a list of strings or regular expression and recursively
    removes all matching column names containing those strings or expressions.
    # Example: if the df_in columns are ['price','sqft','sqft_living','sqft15','sqft_living15','floors','bedrooms']
    df_out = drop_cols(df_in, ['sqft','bedroom'])
    df_out.columns # will output: ['price','floors']

    Parameters:
        DF --
            Input dataframe to remove columns from.
        regex_list --
            list of string patterns or regexp to remove.

    Returns:
        df_dropped -- input df without the dropped columns.
    """
    regex_list=list_of_strings_or_regexp
    df_cut = df.copy()
    for r in regex_list:
        df_cut = df_cut[df_cut.columns.drop(list(df_cut.filter(regex=r)))]
        if verbose>0:
            print(f'Removed {r}.')
    df_dropped = df_cut
    return df_dropped



    ## DataFrame Creation, Inspection, and Exporting
# def inspect_df(df, n_rows=3, verbose=True):
#     """ EDA:
#     Show all pandas inspection tables.
#     Displays df.head(), df.info(), df.describe().
#     By default also runs check_null and check_numeric to inspect
#     columns for null values and to check string columns to detect
#     numeric values. (If verbose==True)
#     Parameters:
#         df(dataframe):
#             dataframe to inspect
#         n_rows:
#             number of header rows to show (Default=3).
#         verbose:
#             If verbose==True (default), check_null and check_numeric.
#     Ex: inspect_df(df,n_rows=4)
#     """
#     # from ..
#     # from bs_ds.bamboo import check_column, check_null, check_numeric, check_unique
#     # from bs_ds.prettypandas import display_side_by_side
#     import pandas as pd
#     from IPython.display import display
#     with pd.option_context("display.max_columns", None ,'display.precision',4):
#         display(df.info()) #, display(df.describe())

#         if verbose == True:

#             df_num = check_numeric(df,unique_check=False, show_df=False)
#             # sdf_num = df_num.style.set_caption('Detected Numeric Values')

#             df_null = check_null(df, show_df=False)
#             # sdf_null = df_null.style.set_caption('Detected Null values')

#             display_side_by_side(df_null, df_num,df.describe())
#         else:
#             display(df.describe())

#         display(df.head(n_rows))





# def drop_cols(df, list_of_strings_or_regexp,verbose=0):#,axis=1):
#     """EDA: Take a df, a list of strings or regular expression and recursively
#     removes all matching column names containing those strings or expressions.
#     # Example: if the df_in columns are ['price','sqft','sqft_living','sqft15','sqft_living15','floors','bedrooms']
#     df_out = drop_cols(df_in, ['sqft','bedroom'])
#     df_out.columns # will output: ['price','floors']

#     Parameters:
#         DF --
#             Input dataframe to remove columns from.
#         regex_list --
#             list of string patterns or regexp to remove.

#     Returns:
#         df_dropped -- input df without the dropped columns.
#     """
#     regex_list=list_of_strings_or_regexp
#     df_cut = df.copy()
#     for r in regex_list:
#         df_cut = df_cut[df_cut.columns.drop(list(df_cut.filter(regex=r)))]
#         if verbose>0:
#             print(f'Removed {r}.')
#     df_dropped = df_cut
#     return df_dropped


def add_filtered_col_to_df(df_source, df_to_add_to, list_of_exps, return_filtered_col_names =False):
    """Takes a dataframe source with columns to copy using df.filter(regexp=(list_of_exps)),
    with list_of_exps being a list of text expressions to find inside column names."""
    # import bs_ds as bs
    import pandas as pd
    filtered_col_list = {}
    for exp in list_of_exps:
        df_temp_filtered = df_source.filter(regex=(exp),axis=1).copy()
        filtered_col_list[exp]= list(df_temp_filtered.columns)

        df_to_add_to = pd.concat([df_to_add_to, df_temp_filtered])

    if return_filtered_col_names == False:
        return df_to_add_to
    else:
        print(filtered_col_list)
        return df_to_add_to, filtered_col_list


##
# EDA / Plotting Functions
def multiplot(df,annot=True,fig_size=None):
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

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    idx = np.triu_indices_from(mask)
    mask[idx] = True

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





# def save_ihelp_to_file(function,save_help=False,save_code=True, 
#                         as_md=False,as_txt=True,
#                         folder='readme_resources/ihelp_outputs/',
#                         filename=None,file_mode='w'):
#     """Saves the string representation of the ihelp source code as markdown. 
#     Filename should NOT have an extension. .txt or .md will be added based on
#     as_md/as_txt.
#     If filename is None, function name is used."""

#     if as_md & as_txt:
#         raise Exception('Only one of as_md / as_txt may be true.')

#     import sys
#     from io import StringIO
#     ## save original output to restore
#     orig_output = sys.stdout
#     ## instantiate io stream to capture output
#     io_out = StringIO()
#     ## Redirect output to output stream
#     sys.stdout = io_out
    
#     if save_code:
#         print('### SOURCE:')
#         help_md = get_source_code_markdown(function)
#         ## print output to io_stream
#         print(help_md)
        
#     if save_help:
#         print('### HELP:')
#         help(function)
        
#     ## Get printed text from io stream
#     text_to_save = io_out.getvalue()
    

#     ## MAKE FULL FILENAME
#     if filename is None:

#         ## Find the name of the function
#         import re
#         func_names_exp = re.compile('def (\w*)\(')
#         func_name = func_names_exp.findall(text_to_save)[0]    
#         print(f'Found code for {func_name}')

#         save_filename = folder+func_name#+'.txt'
#     else:
#         save_filename = folder+filename

#     if as_md:
#         ext = '.md'
#     elif as_txt:
#         ext='.txt'

#     full_filename = save_filename + ext
    
#     with open(full_filename,file_mode) as f:
#         f.write(text_to_save)
        
#     print(f'Output saved as {full_filename}')
    
#     sys.stdout = orig_output



# def get_source_code_markdown(function):
#     """Retrieves the source code as a string and appends the markdown
#     python syntax notation"""
#     import inspect
#     from IPython.display import display, Markdown
#     source_DF = inspect.getsource(function)            
#     output = "```python" +'\n'+source_DF+'\n'+"```"
#     return output

def save_ihelp_menu_to_file(function_list, filename,save_help=False,save_code=True, 
    folder='readme_resources/ihelp_outputs/',as_md=True, as_txt=False,verbose=1):
    """Accepts a list of functions and uses save_ihelp_to_file with mode='a' 
    to combine all outputs. Note: this function REQUIRES a filename"""
    from ..ds import save_ihelp_to_file
    if as_md:
        ext='.md'
    elif as_txt:
        ext='.txt'

    for function in function_list:
        save_ihelp_to_file(function=function,save_help=save_help, save_code=save_code,
                              as_md=as_md, as_txt=as_txt,folder=folder,
                              filename=filename,file_mode='a')

    if verbose>0:
        print(f'Functions saved as {folder+filename+ext}')




def auto_filename_time(prefix='',sep=' ',suffix='',ext='',fname_friendly=True,timeformat='%m-%d-%Y %T'):
    '''Generates a filename with a  base string + sep+ the current datetime formatted as timeformat.
     filename = f"{prefix}{sep}{suffix}{sep}{timesuffix}{ext}
    '''
    if prefix is None:
        prefix=''
    timesuffix=get_time(timeformat=timeformat, filename_friendly=fname_friendly)

    filename = f"{prefix}{sep}{suffix}{sep}{timesuffix}{ext}"
    return filename



def disp_df_head_tail(df,n_head=3, n_tail=3,head_capt='df.head',tail_capt='df.tail'):
    """Displays the df.head(n_head) and df.tail(n_tail) and sets captions using df.style"""
    from IPython.display import display
    import pandas as pd
    df_h = df.head(n_head).style.set_caption(head_capt)
    df_t = df.tail(n_tail).style.set_caption(tail_capt)
    display(df_h, df_t)


def create_required_folders(full_filenamepath,folder_delim='/',verbose=1):
    """Accepts a full file name path include folders with '/' as default delimiter.
    Recursively checks for all sub-folders in filepath and creates those that are missing."""
    import os
    ## Creating folders needed
    check_for_folders = full_filenamepath.split(folder_delim)#'/')

    # if the splits creates more than 1 filepath:
    if len(check_for_folders)==1:
        return print('[!] No folders detected in provided full_filenamepath')
    else:# len(check_for_folders) >1:

        # set first foler to check
        check_path = check_for_folders[0]

        if check_path not in os.listdir():
            if verbose>0:
                print(f'\t- creating folder "{check_path}"')
            os.mkdir(check_path)

        ## handle multiple subfolders
        if len(check_for_folders)>2:

            ## for each subfolder:
            for folder in check_for_folders[1:-1]:
                base_folder_contents = os.listdir(check_path)

                # add the subfolder to prior path
                check_path = check_path + '/' + folder

                if folder not in base_folder_contents:#os.listdir():
                    if verbose>0:
                        print(f'\t- creating folder "{check_path}"')
                    os.mkdir(check_path)
        if verbose>1:
            print('Finished. All required folders have been created.')
        else:
            return
        
        

def dict_dropdown(dict_to_display,title='Dictionary Contents'):
    """Display the model_params dictionary as a dropdown menu."""
    from ipywidgets import interact
    from IPython.display import display
    from pprint import pprint

    dash='---'
    print(f'{dash*4} {title} {dash*4}')

    @interact(dict_to_display=dict_to_display)
    def display_params(dict_to_display=dict_to_display):
        # # if the contents of the first level of keys is dicts:, display another dropdown
        # if dict_to_display.values()
        display(pprint(dict_to_display))
        return #params.values();


# def dict_of_df_dropdown(dict_to_display, selected_key=None):
#     import ipywidgets as widgets
#     from IPython.display import display
#     from ipywidgets import interact, interactive
#     import pandas as pd

#     key_list = list(dict_to_display.keys())
#     key_list.append('_All_')

#     if selected_key is not None:
#         selected_key = selected_key

#     def view(eval_dict=dict_to_display,selected_key=''):

#         from IPython.display import display
#         from pprint import pprint

#         if selected_key=='_All_':

#             key_list = list(eval_dict.keys())
#             outputs=[]

#             for k in key_list:

#                 if type(eval_dict[k]) == pd.DataFrame:
#                     outputs.append(eval_dict[k])
#                     display(eval_dict[k].style.set_caption(k).hide_index())
#                 else:
#                     outputs.append(f"{k}:\n{eval_dict[k]}\n\n")
#                     pprint('\n',eval_dict[k])

#             return outputs#pprint(outputs)

#         else:
#                 k = selected_key
# #                 if type(eval_dict(k)) == pd.DataFrame:
#                 if type(eval_dict[k]) == pd.DataFrame:
#                      display(eval_dict[k].style.set_caption(k))
#                 else:
#                     pprint(eval_dict[k])
#                 return [eval_dict[k]]

#     w= widgets.Dropdown(options=key_list,value='_All_', description='Key Word')

#     # old, simple
#     out = widgets.interactive_output(view, {'selected_key':w})


#     # new, flashier
#     output = widgets.Output(layout={'border': '1px solid black'})
#     if type(out)==list:
#         output.append_display_data(out)
# #         out =widgets.HBox([x for x in out])
#     else:
#         output = out
# #     widgets.HBox([])
#     final_out =  widgets.VBox([widgets.HBox([w]),output])
#     display(final_out)
#     return final_out#widgets.VBox([widgets.HBox([w]),output])#out])


def display_dict_dropdown(dict_to_display ):
    """Display the model_params dictionary as a dropdown menu."""
    from ipywidgets import interact
    from IPython.display import display
    from pprint import pprint

    dash='---'
    print(f'{dash*4} Dictionary Contents {dash*4}')

    @interact(dict_to_display=dict_to_display)
    def display_params(dict_to_display):
        # print(dash)
        pprint(dict_to_display)
        return #params.values();



def get_time(timeformat='%m-%d-%y_%T%p',raw=False,filename_friendly= False,replacement_seperator='-'):
    """
    Gets current time in local time zone.
    if raw: True then raw datetime object returned without formatting.
    if filename_friendly: replace ':' with replacement_separator
    """
    from datetime import datetime
    from pytz import timezone
    from tzlocal import get_localzone

    now_utc = datetime.now(timezone('UTC'))
    now_local = now_utc.astimezone(get_localzone())

    if raw == True:
        return now_local

    else:
        now = now_local.strftime(timeformat)

    if filename_friendly==True:
        return now.replace(':',replacement_seperator).lower()
    else:
        return now
    
    

def print_array_info(X, name='Array'):
    """Test function for verifying shapes and data ranges of input arrays"""
    Xt=X
    print('X type:',type(Xt))
    print(f'X.shape = {Xt.shape}')
    print(f'\nX[0].shape = {Xt[0].shape}')
    print(f'X[0] contains:\n\t',Xt[0])

# from ..ds import arr2series
# def arr2series(array,series_index=[],series_name='predictions'):
#     """Accepts an array, an index, and a name. If series_index is longer than array:
#     the series_index[-len(array):] """
#     import pandas as pd
#     if len(series_index)==0:
#         series_index=list(range(len(array)))

#     if len(series_index)>len(array):
#         new_index= series_index[-len(array):]
#         series_index=new_index

#     preds_series = pd.Series(array.ravel(), index=series_index, name=series_name)
#     return preds_series





class Clock(object):
    """A clock meant to be used as a timer for functions using local time.
    Clock.tic() starts the timer, .lap() adds the current laps time to clock._list_lap_times, .toc() stops the timer.
    If user initiializes with verbose =0, only start and final end times are displays.
        If verbose=1, print each lap's info at the end of each lap.
        If verbose=2 (default, display instruction line, return datafarme of results.)
    """

    from datetime import datetime
    from pytz import timezone
    from tzlocal import get_localzone
    # from bs_ds import list2df

    # from bs_ds import list2df

    def get_time(self,local=True):
        """Returns current time, in local time zone by default (local=True)."""
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        _now_utc_=datetime.now(timezone('UTC'))
        _now_local_=_now_utc_.astimezone(self._timezone_)
        if local==True:
            time_now = _now_local_

            return time_now#_now_local_
        else:
            return _now_utc_


    def __init__(self, display_final_time_as_minutes=True, verbose=2):

        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        self._strformat_ = []
        self._timezone_ = []
        self._timezone_ = get_localzone()
        self._start_time_ = []
        self._lap_label_ = []
        self._lap_end_time_ = []
        self._verbose_ = verbose
        self._lap_duration_ = []
        self._verbose_ = verbose
        self._prior_start_time_ = []
        self._display_as_minutes_ = display_final_time_as_minutes

        strformat = "%m/%d/%y - %I:%M:%S %p"
        self._strformat_ = strformat

    def mark_lap_list(self, label=None):
        """Used internally, appends the current laps' information when called by .lap()
        self._lap_times_list_ = [['Lap #' , 'Start Time','Stop Time', 'Stop Label', 'Duration']]"""
        # import bs_ds as bs
#         print(self._prior_start_time_, self._lap_end_time_)

        if label is None:
            label='--'

        duration = self._lap_duration_.total_seconds()
        self._lap_times_list_.append([ self._lap_counter_ , # Lap #
                                      (self._prior_start_time_).strftime(self._strformat_), # This Lap's Start Time
                                      self._lap_end_time_,#.strftime(self._strformat_), # stop clock time
                                      label,#self._lap_label_, # The Label passed with .lap()
                                      f'{duration:.3f} sec']) # the lap duration


    def tic(self, label=None ):
        "Start the timer and display current time, appends label to the _list_lap_times."
        from datetime import datetime
        from pytz import timezone

        self._start_time_ = self.get_time()
        self._start_label_ = label
        self._lap_counter_ = 0
        self._prior_start_time_=self._start_time_
        self._lap_times_list_=[]

        # Initiate lap counter and list
        self._lap_times_list_ = [['Lap #','Start Time','Stop Time', 'Label', 'Duration']]
        self._lap_counter_ = 0
        self._decorate_ = '--- '
        decorate=self._decorate_
        base_msg = f'{decorate}CLOCK STARTED @: {self._start_time_.strftime(self._strformat_):>{25}}'

        if label == None:
            display_msg = base_msg+' '+ decorate
            label='--'
        else:
            spacer = ' '
            display_msg = base_msg+f'{spacer:{10}} Label: {label:{10}} {decorate}'
        if self._verbose_>0:
            print(display_msg)#f'---- Clock started @: {self._start_time_.strftime(self._strformat_):>{25}} {spacer:{10}} label: {label:{20}}  ----')

    def toc(self,label=None, summary=True):
        """Stop the timer and displays results, appends label to final _list_lap_times entry"""
        if label == None:
            label='--'
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone
        from fsds_100719.ds import list2df
        if label is None:
            label='--'

        _final_end_time_ = self.get_time()
        _total_time_ = _final_end_time_ - self._start_time_
        _end_label_ = label

        self._lap_counter_+=1
        self._final_end_time_ = _final_end_time_
        self._lap_label_=_end_label_
        self._lap_end_time_ = _final_end_time_.strftime(self._strformat_)
        self._lap_duration_ = _final_end_time_ - self._prior_start_time_
        self._total_time_ = _total_time_

        decorate=self._decorate_
        # Append Summary Line
        if self._display_as_minutes_ == True:
            total_seconds = self._total_time_.total_seconds()
            total_mins = int(total_seconds // 60)
            sec_remain = total_seconds % 60
            total_time_to_display = f'{total_mins} min, {sec_remain:.3f} sec'
        else:

            total_seconds = self._total_time_.total_seconds()
            sec_remain = round(total_seconds % 60,3)

            total_time_to_display = f'{sec_remain} sec'
        self._lap_times_list_.append(['TOTAL',
                                      self._start_time_.strftime(self._strformat_),
                                      self._final_end_time_.strftime(self._strformat_),
                                      label,
                                      total_time_to_display]) #'Total Time: ', total_time_to_display])

        if self._verbose_>0:
            print(f'--- TOTAL DURATION   =  {total_time_to_display:>{15}} {decorate}')

        if summary:
            self.summary()

    def lap(self, label=None):
        """Records time, duration, and label for current lap. Output display varies with clock verbose level.
        Calls .mark_lap_list() to document results in clock._list_lap_ times."""
        from datetime import datetime
        if label is None:
            label='--'
        _end_time_ = self.get_time()

        # Append the lap attribute list and counter
        self._lap_label_ = label
        self._lap_end_time_ = _end_time_.strftime(self._strformat_)
        self._lap_counter_+=1
        self._lap_duration_ = (_end_time_ - self._prior_start_time_)
        # Now update the record
        self.mark_lap_list(label=label)

        # Now set next lap's new _prior_start
        self._prior_start_time_=_end_time_
        spacer = ' '

        if self._verbose_>0:
            print(f'       - Lap # {self._lap_counter_} @:  \
            {self._lap_end_time_:>{25}} {spacer:{5}} Dur: {self._lap_duration_.total_seconds():.3f} sec.\
            {spacer:{5}}Label:  {self._lap_label_:{20}}')

    def summary(self):
        """Display dataframe summary table of Clock laps"""
        from fsds_100719.ds import list2df
        import pandas as pd
        from IPython.display import display
        df_lap_times = list2df(self._lap_times_list_)#,index_col='Lap #')
        df_lap_times.drop('Stop Time',axis=1,inplace=True)
        df_lap_times = df_lap_times[['Lap #','Start Time','Duration','Label']]
        dfs = df_lap_times.style.hide_index().set_caption('Summary Table of Clocked Processes').set_properties(subset=['Start Time','Duration'],**{'width':'140px'})
        display(dfs.set_table_styles([dict(selector='table, th', props=[('text-align', 'center')])]))






# def plot_confusion_matrix(conf_matrix, classes = None, normalize=False,
#                           title='Confusion Matrix', cmap=None,
#                           print_raw_matrix=False,fig_size=(5,5), show_help=False):
#     """Check if Normalization Option is Set to True. If so, normalize the raw confusion matrix before visualizing
#     #Other code should be equivalent to your previous function.
#     Note: Taken from bs_ds and modified"""
#     import itertools
#     import numpy as np
#     import matplotlib.pyplot as plt

#     cm = conf_matrix
#     ## Set plot style properties
#     if cmap==None:
#         cmap = plt.get_cmap("Blues")

#     ## Text Properties
#     fmt = '.2f' if normalize else 'd'

#     fontDict = {
#         'title':{
#             'fontsize':16,
#             'fontweight':'semibold',
#             'ha':'center',
#             },
#         'xlabel':{
#             'fontsize':14,
#             'fontweight':'normal',
#             },
#         'ylabel':{
#             'fontsize':14,
#             'fontweight':'normal',
#             },
#         'xtick_labels':{
#             'fontsize':10,
#             'fontweight':'normal',
#             'rotation':45,
#             'ha':'right',
#             },
#         'ytick_labels':{
#             'fontsize':10,
#             'fontweight':'normal',
#             'rotation':0,
#             'ha':'right',
#             },
#         'data_labels':{
#             'ha':'center',
#             'fontweight':'semibold',

#         }
#     }


#     ## Normalize data
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     # Create plot
#     fig,ax = plt.subplots(figsize=fig_size)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title,**fontDict['title'])
#     plt.colorbar()

#     if classes is None:
#         classes = ['negative','positive']

#     tick_marks = np.arange(len(classes))


#     plt.xticks(tick_marks, classes, **fontDict['xtick_labels'])
#     plt.yticks(tick_marks, classes,**fontDict['ytick_labels'])


#     # Determine threshold for b/w text
#     thresh = cm.max() / 2.

#     # fig,ax = plt.subplots()
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), color='darkgray',**fontDict['data_labels'])#color="white" if cm[i, j] > thresh else "black"

#     plt.tight_layout()
#     plt.ylabel('True label',**fontDict['ylabel'])
#     plt.xlabel('Predicted label',**fontDict['xlabel'])
#     fig = plt.gcf()
#     plt.show()

#     if print_raw_matrix:
#         print_title = 'Raw Confusion Matrix Counts:'
#         print('\n',print_title)
#         print(conf_matrix)

#     if show_help:
#         print('''For binary classifications:
#         [[0,0(true_neg),  0,1(false_pos)]
#         [1,0(false_neg), 1,1(true_pos)] ]

#         to get vals as vars:
#         >>  tn,fp,fn,tp=confusion_matrix(y_test,y_hat_test).ravel()
#                 ''')

#     return fig





def evaluate_regression(y_true, y_pred, metrics=None, show_results=False, display_thiels_u_info=False):
    """Calculates and displays any of the following evaluation metrics: (passed as strings in metrics param)
    r2, MAE,MSE,RMSE,U
    if metrics=None:
        metrics=['r2','RMSE','U']
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np
    import inspect

    idx_true_null = find_null_idx(y_true)
    idx_pred_null = find_null_idx(y_pred)
    if all(idx_true_null == idx_pred_null):
        y_true.dropna(inplace=True)
        y_pred.dropna(inplace=True)
    else:
        raise Exception('There are non-overlapping null values in y_true and y_pred')

    results=[['Metric','Value']]
    metric_list = []
    if metrics is None:
        metrics=['r2','rmse','u']

    else:
        for metric in metrics:
            if isinstance(metric,str):
                metric_list.append(metric.lower())
            elif inspect.isfunction(metric):
                custom_res = metric(y_true,y_pred)
                results.append([metric.__name__,custom_res])
                metric_list.append(metric.__name__)
        metrics=metric_list

    # metrics = [m.lower() for m in metrics]

    if any(m in metrics for m in ('r2','r squared','R_squared')): #'r2' in metrics: #any(m in metrics for m in ('r2','r squared','R_squared'))
        r2 = r2_score(y_true, y_pred)
        results.append(['R Squared',r2])##f'R\N{SUPERSCRIPT TWO}',r2])

    if any(m in metrics for m in ('RMSE','rmse','root_mean_squared_error','root mean squared error')): #'RMSE' in metrics:
        RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
        results.append(['Root Mean Squared Error',RMSE])

    if any(m in metrics for m in ('MSE','mse','mean_squared_error','mean squared error')):
        MSE = mean_squared_error(y_true,y_pred)
        results.append(['Mean Squared Error',MSE])

    if any(m in metrics for m in ('MAE','mae','mean_absolute_error','mean absolute error')):#'MAE' in metrics or 'mean_absolute_error' in metrics:
        MAE = mean_absolute_error(y_true,y_pred)
        results.append(['Mean Absolute Error',MAE])


    if any(m in metrics for m in ('u',"thiel's u")):# in metrics:
        if display_thiels_u_info is True:
            show_eqn=True
            show_table=True
        else:
            show_eqn=False
            show_table=False

        U = thiels_U(y_true, y_pred,display_equation=show_eqn,display_table=show_table )
        results.append(["Thiel's U", U])
    from fsds_100719.ds import list2df
    results_df = list2df(results)#, index_col='Metric')
    results_df.set_index('Metric', inplace=True)
    if show_results:
        from IPython.display import display
        dfs = results_df.round(3).reset_index().style.hide_index().set_caption('Evaluation Metrics')
        display(dfs)
    return results_df.round(4)


def thiels_U(ys_true=None, ys_pred=None,display_equation=True,display_table=True):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Returns Thiel's U"""


    from IPython.display import Markdown, Latex, display
    import numpy as np
    display(Markdown(""))
    eqn=" $$U = \\sqrt{\\frac{ \\sum_{t=1 }^{n-1}\\left(\\frac{\\bar{Y}_{t+1} - Y_{t+1}}{Y_t}\\right)^2}{\\sum_{t=1 }^{n-1}\\left(\\frac{Y_{t+1} - Y_{t}}{Y_t}\\right)^2}}$$"

    # url="['Explanation'](https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html)"
    markdown_explanation ="|Thiel's U Value | Interpretation |\n\
    | --- | --- |\n\
    | <1 | Forecasting is better than guessing| \n\
    | 1 | Forecasting is about as good as guessing| \n\
    |>1 | Forecasting is worse than guessing| \n"


    if display_equation and display_table:
        display(Latex(eqn),Markdown(markdown_explanation))#, Latex(eqn))
    elif display_equation:
        display(Latex(eqn))
    elif display_table:
        display(Markdown(markdown_explanation))

    if ys_true is None and ys_pred is None:
        return

    # sum_list = []
    num_list=[]
    denom_list=[]
    for t in range(len(ys_true)-1):
        num_exp = (ys_pred[t+1] - ys_true[t+1])/ys_true[t]
        num_list.append([num_exp**2])
        denom_exp = (ys_true[t+1] - ys_true[t])/ys_true[t]
        denom_list.append([denom_exp**2])
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U

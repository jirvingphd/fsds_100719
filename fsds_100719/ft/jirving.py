"""My Template Module 
Name: James M. Irving
Email: james.irving.phd@gmail.com
GitHub Profile: https://github.com/jirvingphd
"""
# import fsds_100719 as fs 
from fsds_100719.ds import ihelp,ihelp_menu, reload
from fsds_100719.jmi import print_docstring_template
# print(f"[i] You're using V {fs.__version__} of fsds.")

def testing():
    print('Huzzah!')
    
def check_column(df, col_name, n_unique=10):
    """Displays info on null values, datatype, unqiue values
    and displays .describe()
    
    Args:
        df (df): contains the columns
        col_name (str): name of the df column to show
        n_unique (int): Number of unique values top show.
    
    Return:
        fig, ax (Matplotlib Figure and Axes)
    """
    import matplotlib.pyplot as plt
    from IPython.display import display
    print('DataType:')
    print('\t',df[col_name].dtypes)
    
    num_nulls = df[col_name].isna().sum()
    print(f'Null Values Present = {num_nulls}')
    
    display(df[col_name].describe().round(3))
    
    print('\nValue Counts:')
    display(df[col_name].value_counts(n_unique))
    
    ## Add some EDA figures
    fig, ax = plt.subplots(ncols=2)
    
    return fig,ax
    
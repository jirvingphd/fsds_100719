__version__ = '0.2.3'
from .imports import *

print(f"fsds_1007219  v{__version__} loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ ")
print(f"> For convenient loading of standard modules use: `>> from fsds_100719.imports import *`\n")


def ihelp(function_or_mod, show_help=True, show_code=True,return_code=False,markdown=True,file_location=False):
    """Call on any module or functon to display the object's
    help command printout AND/OR soruce code displayed as Markdown
    using Python-syntax"""

    import inspect
    from IPython.display import display, Markdown
    page_header = '---'*28
    footer = '---'*28+'\n'
    if show_help:
        print(page_header)
        banner = ''.join(["---"*2,' HELP ',"---"*24,'\n'])
        print(banner)
        help(function_or_mod)
        # print(footer)
        
    import sys
    if "google.colab" in sys.modules:
        markdown=False

    if show_code:
        print(page_header)

        banner = ''.join(["---"*2,' SOURCE -',"---"*23])
        print(banner)
        try:
            import inspect
            source_DF = inspect.getsource(function_or_mod)

            if markdown == True:
                
                output = "```python" +'\n'+source_DF+'\n'+"```"
                display(Markdown(output))
            else:
                print(source_DF)

        except TypeError:
            pass
            # display(Markdown)


    if file_location:
        file_loc = inspect.getfile(function_or_mod)
        banner = ''.join(["---"*2,' FILE LOCATION ',"---"*21])
        print(page_header)
        print(banner)
        print(file_loc)

    # print(footer)

    if return_code:
        return source_DF


def list2df(list, index_col=None, set_caption=None, return_df=True,df_kwds=None): #, sort_values='index'):
    
    """ Quick turn an appened list with a header (row[0]) into a pretty dataframe.

        
        Args
            list (list of lists):
            index_col (string): name of column to set as index; None (Default) has integer index.
            set_caption (string):
            show_and_return (bool):
    
    EXAMPLE USE:
    >> list_results = [["Test","N","p-val"]] 
    
    # ... run test and append list of result values ...
    
    >> list_results.append([test_Name,length(data),p])
    
    ## Displays styled dataframe if caption:
    >> df = list2df(list_results, index_col="Test",
                     set_caption="Stat Test for Significance")
    

    """
    from IPython.display import display
    import pandas as pd
    df_list = pd.DataFrame(list[1:],columns=list[0],**df_kwds)
    
        
    if index_col is not None:
        df_list.reset_index(inplace=True)
        df_list.set_index(index_col, inplace=True)
        
    if set_caption is not None:
        dfs = df_list.style.set_caption()
        display(dfs)
    return df_list

# def column_report(df,index_col='iloc', sort_column='iloc', ascending=True,name_for_notes_col = 'Notes',notes_by_dtype=False,
#  decision_map=None, format_dict=None,   as_qgrid=True, qgrid_options=None, qgrid_column_options=None,qgrid_col_defs=None, qgrid_callback=None,
#  as_df = False, as_interactive_df=False, show_and_return=True):

#     """
#     Returns a datafarme summary of the columns, their dtype,  a summary dataframe with the column name, column dtypes, and a `decision_map` dictionary of
#     datatype.
#     [!] Please note if qgrid does not display properly, enter this into your terminal and restart your temrinal.
#         'jupyter nbextension enable --py --sys-prefix qgrid'# required for qgrid
#         'jupyter nbextension enable --py --sys-prefix widgetsnbextension' # only required if you have not enabled the ipywidgets nbextension yet
    
#     Default qgrid options:
#        default_grid_options={
#         # SlickGrid options
#         'fullWidthRows': True,
#         'syncColumnCellResize': True,
#         'forceFitColumns': True,
#         'defaultColumnWidth': 50,
#         'rowHeight': 25,
#         'enableColumnReorder': True,
#         'enableTextSelectionOnCells': True,
#         'editable': True,
#         'autoEdit': False,
#         'explicitInitialization': True,

#         # Qgrid options
#         'maxVisibleRows': 30,
#         'minVisibleRows': 8,
#         'sortable': True,
#         'filterable': True,
#         'highlightSelectedCell': True,
#         'highlightSelectedRow': True
#     }
#     """
#     from ipywidgets import interact
#     import pandas as pd
#     from IPython.display import display
#     import qgrid
#     small_col_width = 20

#     # default_col_options={'width':20}

#     default_column_definitions={'column name':{'width':60}, '.iloc[:,i]':{'width':small_col_width}, 'dtypes':{'width':30}, '# zeros':{'width':small_col_width},
#                     '# null':{'width':small_col_width},'% null':{'width':small_col_width}, name_for_notes_col:{'width':100}}

#     default_grid_options={
#         # SlickGrid options
#         'fullWidthRows': True,
#         'syncColumnCellResize': True,
#         'forceFitColumns': True,
#         'defaultColumnWidth': 50,
#         'rowHeight': 25,
#         'enableColumnReorder': True,
#         'enableTextSelectionOnCells': True,
#         'editable': True,
#         'autoEdit': False,
#         'explicitInitialization': True,

#         # Qgrid options
#         'maxVisibleRows': 30,
#         'minVisibleRows': 8,
#         'sortable': True,
#         'filterable': True,
#         'highlightSelectedCell': True,
#         'highlightSelectedRow': True
#     }

#     ## Set the params to defaults, to then be overriden
#     column_definitions = default_column_definitions
#     grid_options=default_grid_options
#     # column_options = default_col_options

#     if qgrid_options is not None:
#         for k,v in qgrid_options.items():
#             grid_options[k]=v

#     if qgrid_col_defs is not None:
#         for k,v in qgrid_col_defs.items():
#             column_definitions[k]=v
#     else:
#         column_definitions = default_column_definitions


#     # format_dict = {'sum':'${0:,.0f}', 'date': '{:%m-%Y}', 'pct_of_total': '{:.2%}'}
#     # monthly_sales.style.format(format_dict).hide_index()
#     def count_col_zeros(df, columns=None):
#         import pandas as pd
#         import numpy as np
#         # Make a list of keys for every column  (for series index)
#         zeros = pd.Series(index=df.columns)
#         # use all cols by default
#         if columns is None:
#             columns=df.columns

#         # get sum of zero values for each column
#         for col in columns:
#             zeros[col] = np.sum( df[col].values == 0)
#         return zeros


#     ##
#     df_report = pd.DataFrame({'.iloc[:,i]': range(len(df.columns)),
#                             'column name':df.columns,
#                             'dtypes':df.dtypes.astype('str'),
#                             '# zeros': count_col_zeros(df),
#                             '# null': df.isna().sum(),
#                             '% null':df.isna().sum().divide(df.shape[0]).mul(100).round(2)})
#     ## Sort by index_col
#     if 'iloc' in index_col:
#         index_col = '.iloc[:,i]'

#     df_report.set_index(index_col ,inplace=True)

#     ## Add additonal column with notes
#     # decision_map_keys = ['by_name', 'by_dtype','by_iloc']
#     if decision_map is None:
#         decision_map ={}
#         decision_map['by_dtype'] = {'object':'Check if should be one hot coded',
#                         'int64':'May be  class object, or count of a ',
#                         'bool':'one hot',
#                         'float64':'drop and recalculate'}

#     if notes_by_dtype:
#         df_report[name_for_notes_col] = df_report['dtypes'].map(decision_map['by_dtype'])#column_list
#     else:
#         df_report[name_for_notes_col] = ''
# #     df_report.style.set_caption('DF Columns, Dtypes, and Course of Action')

#     ##  Sort column
#     if sort_column is None:
#         sort_column = '.iloc[:,i]'


#     if 'iloc' in sort_column:
#         sort_column = '.iloc[:,i]'

#     df_report.sort_values(by =sort_column,ascending=ascending, axis=0, inplace=True)

#     if as_df:
#         if show_and_return:
#             display(df_report)
#         return df_report

#     elif as_qgrid:
#         print('[i] qgrid returned. Use gqrid.get_changed_df() to get edited df back.')
#         qdf = qgrid.show_grid(df_report,grid_options=grid_options, column_options=qgrid_column_options, column_definitions=column_definitions,row_edit_callback=qgrid_callback  )
#         if show_and_return:
#             display(qdf)
#         return qdf

#     elif as_interactive_df:

#         @interact(column= df_report.columns,direction={'ascending':True,'descending':False})
#         def sort_df(column, direction):
#             return df_report.sort_values(by=column,axis=0,ascending=direction)
#     else:
#         raise Exception('One of the output options must be true: `as_qgrid`,`as_df`,`as_interactive_df`')

def inspect_variables(local_vars = None,sort_col='size',exclude_funcs_mods=True, top_n=10,return_df=False,always_display=True,
show_how_to_delete=False,print_names=False):
    """Displays a dataframe of all variables and their size in memory, with the
    largest variables at the top."""
    import sys
    import inspect
    import pandas as pd
    from IPython.display import display
    if local_vars is None:
        raise Exception('Must pass "locals()" in function call. i.e. inspect_variables(locals())')


    glob_vars= [k for k in globals().keys()]
    loc_vars = [k for k in local_vars.keys()]

    var_list = glob_vars+loc_vars

    var_df = pd.DataFrame(columns=['variable','size','type'])

    exclude = ['In','Out']
    var_list = [x for x in var_list if (x.startswith('_') == False) and (x not in exclude)]

    i=0
    for var in var_list:#globals().items():#locals().items():

        if var in loc_vars:
            real_var = local_vars[var]
        elif var in glob_vars:
            real_var = globals()[var]
        else:
            print(f"{var} not found.")

        var_size = sys.getsizeof(real_var)

        var_type = []
        if inspect.isfunction(real_var):
            var_type = 'function'
            if exclude_funcs_mods:
                continue
        elif inspect.ismodule(real_var):
            var_type = 'module'
            if exclude_funcs_mods:
                continue
        elif inspect.isbuiltin(real_var):
            var_type = 'builtin'
        elif inspect.isclass(real_var):
            var_type = 'class'
        else:

            var_type = real_var.__class__.__name__


        var_row = pd.Series({'variable':var,'size':var_size,'type':var_type})
        var_df.loc[i] = var_row#pd.concat([var_df,var_row],axis=0)#.join(var_row,)
        i+=1

    # if exclude_funcs_mods:
    #     var_df = var_df.loc[var_df['type'] not in ['function', 'module'] ]

    var_df.sort_values(sort_col,ascending=False,inplace=True)
    var_df.reset_index(inplace=True,drop=True)
    var_df.set_index('variable',inplace=True)
    var_df = var_df[['type','size']]

    if top_n is not None:
        var_df = var_df.iloc[:top_n]



    if always_display:
        display(var_df.style.set_caption('Current Variables by Size in Memory'))

    if show_how_to_delete:
        print('---'*15)
        print('## CODE TO DELETE MANY VARS AT ONCE:')
        show_del_me_code(called_by_inspect_vars=True)


    if print_names ==False:
        print('#[i] set `print_names=True` for var names to copy/paste.')
        print('---'*15)
    else:
        print('---'*15)
        print('Variable Names:\n')
        print_me = [f"{str(x)}" for x in var_df.index]
        print(print_me)
    
        
    if show_del_me_code == False:
        print("[i] set `show_del_me_code=True prints copy/paste var deletion code.")
        

    if return_df:
        return var_df




def show_del_me_code(called_by_inspect_vars=False):
    """Prints code to copy and paste into a cell to delete vars using a list of their names.
    Companion function inspect_variables(locals(),print_names=True) will provide var names tocopy/paste """
    from pprint import pprint
    if called_by_inspect_vars==False:
        print("#[i]Call: `inspect_variables(locals(), print_names=True)` for list of var names")

    del_me = """
    del_me= []#list of variable names
    for me in del_me:
        try:
            exec(f'del {me}')
            print(f'del {me} succeeded')
        except:
            print(f'del {me} failed')
            continue
        """
    print(del_me)


def reload(mod):
    """Reloads the module from file without restarting kernel.
        Args:
            mod (loaded mod or list of mod objects): name or handle of package (i.e.,[ pd, fs,np])
        Returns:
            reload each model.
    Example:
    # You pass in whatever name you imported as.
    import my_functions_from_file as mf
    # after editing the source file:
    # mf.reload(mf)"""
    from importlib import reload
    print(f'Reloading...\n')
    
    ## 
    if isinstance(mod,list):
        return [reload(m) for m in mod]
    else:
        return  reload(mod)
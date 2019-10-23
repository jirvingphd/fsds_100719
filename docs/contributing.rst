
TO CONTRIBUTE::
==============

1) Fork the project repo to your github account and clone it to your computer.
 https://github.com/jirvingphd/fsds_100719

2) To add your own submodule with your code, 
navigate to the `ft` or `pt` folders (full or part time).


3) Add a new `module_name.py` file, where `module_name` is what your submodule will be called.  
I recommended either your github username or you intials. 
    i.e. for me (github=jirvingphd), if I were a full time student,
I would create  `fsds_100719/ft/jirvingphd.py`.

4) Open your .py file (recommended editor is Visual Studio Code) and add a module docstring contact info. 
i.e.:
""" A collection of functions by James M. Irving, PhD from Flatiron ds bootcamp
""" GitHub: jirvingphd, email: james.irving@flatironschool.com

5) Add whatever functions and classes you'd like!
> NOTE: Unlike functions inside notebooks, you *must import 
every package* used in a function *INSIDE* oF the function.
i.e.:
def combine_dfs(df1,df2):
    """Concatenate 2 dataframes."""
    import pandas as pd
    comb_df = pd.concat([df1,df1],axis=1)]
    return comb_df


6) Commit your changes and push them back to github.

7) Open your forked fsds_100719 repo on github and click `New Pull Request` (next to branch dropdown at top of repo file list).  Fill out the form and click `Create pull request`

8) Travis-CI.org will automatically check your pull reqest for bugs and I will notify you if there is an error to resolve (will notify via contact info in header of your submodule.)


.. include:: ../CONTRIBUTING.rst

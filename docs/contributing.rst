.. include:: ../CONTRIBUTING.rst

To contribute:
-------------

1) Fork the project repo to your github account and clone it to your computer.

2) To add your own code, navigate to the `ft` oe `pt` folders, 
    based on if you're a full-time or part-time student, respectively.

3) Add a new .py file using your github name (preferably, not required)
i.e. for me (github=jirvingphd), if I were a full time student,
I would create  "fsds_100719/ft/jirvingphd.py".

4) Open your .py file (recommended editor is Visual Studio Code).
- Add a beginning docstring with your contact info.
i.e.:
""" A collection of functions by James M. Irving, PhD from Flatiron ds bootcamp
""" GitHub: jirvingphd, email: james.irving@flatironschool.com

5) Add whatever functions and classes you'd like!

NOTE: Unlike functions inside notebooks, you must import 
every package used in a function INSIDE oF the function.
i.e.:
def combine_dfs(df1,df2):
    import pandas as pd
    comb_df = pd.concat([df1,df1],axis=1)]
    return comb_df

6) Commit your changes and push them back to github.

7) Go to the package's repo on your github account and click `New Pull Request`.
The button above file list, next to the branch button on the repo's page. 
Fill out the form and click `Create pull request`

8) Travis-CI.org will automatically check your pull reqest for bugs and I will notify you 
if there is an error to resolve (will notify via contact info in header of your submodule.)
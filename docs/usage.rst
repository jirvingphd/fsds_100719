=====
Usage
=====

To use fsds_100719 in a project::

    import fsds_100719 as fs

To import common modules as their usual handles
e.g. pandas as pd, numpy as np,etc.::

    from fsds_100719.imports import *

Functions worth importing by name::
    
    # To easily inspect help and source code
    from fsds_100719 import ihelp
    
    #If you're import funcs from a local file.
    from fsds_100719 import reload 


You can load just your cohort or your own module as fs::
    import fsds_100719.ft.jirvingphd as fs
    # or
    import fsds_100719.ft as fs

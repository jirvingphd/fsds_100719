from .jmi import *
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



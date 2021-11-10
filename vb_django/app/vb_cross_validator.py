import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


class RegressorQStratifiedCV:

    def __init__(self, n_splits=5, n_repeats=2, groupcount=3, random_state=0, strategy='quantile'):
        self.groupcount = groupcount
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)

    def split(self, X, y, groups=None):
        if len(y.shape) > 1:
            if type(y) is pd.DataFrame:
                y_vec = y.to_numpy()[:, 0]
            else:
                y_vec = y[:, 0]
        else:
            y_vec = y
        ysort_order = np.argsort(y_vec)
        y1 = np.ones(y_vec.shape)
        y1split = np.array_split(y1, self.groupcount)
        kgroups = np.empty(y_vec.shape)
        kgroups[ysort_order] = np.concatenate([y1split[i] * i for i in range(self.groupcount)], axis=0)
        return self.cv.split(X, kgroups)

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)


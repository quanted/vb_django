import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer


class RegressorQStratifiedCV:

    def __init__(self, n_splits=5, n_repeats=2, groupcount=3, random_state=0, strategy='quantile'):
        self.groupcount = groupcount
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer = KBinsDiscretizer(n_bins=self.groupcount, encode='ordinal', strategy=self.strategy)

    def split(self, X, y, groups=None):
        if len(y.shape) > 1:
            if type(y) is pd.DataFrame:
                y_vec = y.to_numpy()[:, 0]
            else:
                y_vec = y[:, 0]
        else:
            y_vec = y
        kgroups = self.discretizer.fit_transform(y_vec[:, None])[:, 0]
        return self.cv.split(X, kgroups, groups)

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)


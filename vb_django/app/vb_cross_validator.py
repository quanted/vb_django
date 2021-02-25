import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


class RegressorQStratifiedCV:

    def __init__(self, n_splits=10, n_repeats=2, groupcount=10, random_state=0, strategy='quantile'):
        self.groupcount = groupcount
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)

    def split(self, X, y):
        ysort_order = np.argsort(y)
        y1 = np.ones(y.shape)
        y1split = np.array_split(y1, self.groupcount)
        kgroups = np.empty(y.shape)
        kgroups[ysort_order] = np.concatenate([y1split[i] * i for i in range(self.groupcount)], axis=0)
        return self.cv.split(X, kgroups)

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)


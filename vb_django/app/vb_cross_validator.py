from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer


class RegressorQStratifiedCV:
    def __init__(self, n_splits=10, n_repeats=2, group_count=10, random_state=0, strategy='quantile'):
        self.group_count = group_count
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer = KBinsDiscretizer(n_bins=self.group_count, encode='ordinal', strategy=self.strategy)

    def split(self, X, y, groups=None):
        kgroups = self.discretizer.fit_transform(y[:, None])[:, 0]
        return self.cv.split(X, kgroups, groups)

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)

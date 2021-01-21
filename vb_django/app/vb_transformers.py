import logging
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import ElasticNet, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression


class ColumnBestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, float_k=None):
        self.logger = logging.getLogger()
        self.transform_funcs = {
            'abs_ln': lambda x: np.log(np.abs(x) + .0000001),
            'exp': lambda x: np.exp(x / 100),
            'recip': lambda x: (x + .000000001) ** -1,
            'none': lambda x: x,
            'exp_inv': lambda x: np.exp(x) ** -1
        }
        self.float_k = float_k

    def fit(self, X, y=None):
        if not self.float_k is None:
            Xn = X[:, :self.float_k]
        else:
            Xn = X
        self.k_ = Xn.shape[1]
        pvals = []
        self.logger.info(f'Xn.shape:{Xn.shape},Xn:{Xn}')
        for fn in self.transform_funcs.values():
            TXn = fn(Xn)
            try:
                F, p = f_regression(TXn, y)
            except:
                self.logger.exception(f'error doing f_regression')
                p = np.array([10000.] * TXn.shape[1])
            pvals.append(p[None, :])

        pval_stack = np.concatenate(pvals, axis=0)  # each row is a transform
        bestTloc = np.argsort(pval_stack, axis=0)[0, :]
        Ts = list(self.transform_funcs.keys())
        self.bestTlist = [Ts[i] for i in bestTloc]
        self.logger.info(f'bestTlist:{self.bestTlist},')
        T_s = list(self.transform_funcs.keys())
        self.best_T_ = [T_s[loc] for loc in bestTloc]
        return self

    def transform(self, X):
        for c, t in enumerate(self.best_T_):
            X[:, c] = self.transform_funcs[t](X[:, c])
        return X


class DropConst(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.logger = logging.getLogger()
        pass

    def fit(self, X, y=None):
        if type(X) is np.ndarray:
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        self.unique_ = X_df.apply(pd.Series.nunique)
        return self

    def transform(self, X):
        if type(X) is pd.DataFrame:
            return X.loc[:, self.unique_ > 1]
        else:
            return X[:, self.unique_ > 1]

    def get_feature_name(self, input_features=None):
        if input_features is None:
            input_features = [f'var_{i}' for i in range(len(self.unique_))]
        return [input_features[i] for i, count in enumerate(self.unique_) if count > 1]


class ShrinkBigKTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_k=None, k_share=None, selector=None):
        self.logger = logging.getLogger()
        self.max_k = max_k
        self.k_share = k_share
        self.selector = selector

    def get_feature_name(self, input_features=None):
        if input_features is None:
            input_features = [f'var_{i}' for i in range(len(self.k_))]
        return [input_features[i] for i in self.col_select_]

    def fit(self, X, y):
        assert not y is None, f'y:{y}'
        k = X.shape[1]
        self.k_ = k
        if self.max_k is None:
            if self.k_share is None:
                self.max_k = 500
            else:
                self.max_k = int(k * self.k_share)

        if self.selector is None:
            self.selector = 'Lars'
        if self.selector == 'Lars':
            selector = Lars(fit_intercept=1, normalize=1, n_nonzero_coefs=self.max_k)
        elif self.selector == 'elastic-net':
            selector = ElasticNet(fit_intercept=True, selection='random', tol=0.001, max_iter=5000, warm_start=1,
                                  random_state=0)
        else:
            selector = self.selector

        selector.fit(X, y)
        self.col_select_ = np.arange(k)[np.abs(selector.coef_) > 0.0001]
        if self.col_select_.size < 1:
            self.col_select_ = np.arange(1)
        return self

    def transform(self, X):
        return X[:, self.col_select_]


class LogMinPlus1_T(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.x_min_ = np.min(X)
        return self

    def transform(self, X, y=None):
        return np.log(X - self.x_min_ + 1)

    def inverse_transform(self, X, y=None):
        return np.exp(X) - 1 + self.x_min_


class LogP1_T(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.logger = logging.getLogger()
        pass

    def fit(self, X, y=None):
        xmin = X.min()
        if xmin < 0:
            self.min_shift_ = -xmin
        else:
            self.min_shift_ = 0
        self.logger.debug(f'logp1_T fitting with self.min_shift:{self.min_shift_}')
        return self

    def transform(self, X, y=None):
        X[X < -self.min_shift_] = -self.min_shift_  # added to avoid np.log(neg), really np.log(<1) b/c 0+1=1
        XT = np.log1p(X + self.min_shift_)
        return XT

    def inverse_transform(self, X, y=None):
        XiT = np.expm1(X) - self.min_shift_
        try:
            infinites = XiT.size - np.isfinite(XiT).sum()
        except:
            self.logger.exception(f'type(XiT):{type(XiT)}')
        XiT[~np.isfinite(XiT)] = 10 ** 50
        return XiT


class Log_T(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        XT = np.zeros(X.shape)
        XT[X > 0] = np.log(X[X > 0])
        return XT

    def inverse_transform(self, X, y=None):
        return np.exp(X)


class LogMinus_T(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.sign(X) * np.log(np.abs(X))

    def inverse_transform(self, X, y=None):
        return np.exp(X)


class Exp_T(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return np.exp(X)

    def inverse_transform(self, X, y=None):
        xout = np.zeros(X.shape)
        xout[X > 0] = np.log(X[X > 0])
        return xout


class None_T(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.k_ = X.shape[1]
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

    def get_feature_name(self, input_features=None):
        if input_features is None:
            input_features = [f'var_{i}' for i in range(len(self.k_))]
        return input_features

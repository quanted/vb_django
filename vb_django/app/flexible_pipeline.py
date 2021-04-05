import numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from vb_django.app.vb_transformers import ShrinkBigKTransformer, ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.vb_helper import VBLogger
from vb_django.app.base_helper import BaseHelper
from sklearn.pipeline import Pipeline


class FlexibleEstimator(BaseEstimator, RegressorMixin, VBLogger):
    def __init__(self, form='expXB', robust=False, shift=True, scale=True):
        self.form = form
        self.robust = robust
        self.shift = shift
        self.scale = scale

    def linear(self, B, X):
        Bconst = B[0]
        Betas = B[1:]
        y = Bconst + (X @ Betas)
        return np.nan_to_num(y, nan=1e298)

    def powXB(self, B, X):
        param_idx = 0
        if self.shift:
            Bshift = B[param_idx]
            param_idx += 1
        else:
            Bshift = 0
        if self.scale:
            Bscale = B[param_idx]
            param_idx += 1
        else:
            Bscale = 1
        Bexponent = B[param_idx]
        param_idx += 1
        Bconst = B[param_idx]
        param_idx += 1
        Betas = B[param_idx:]

        y = Bshift + Bscale * (Bconst + (X @ Betas)) ** (int(Bexponent))
        return np.nan_to_num(y, nan=1e290)

    def expXB(self, B, X):
        param_idx = 0
        if self.shift:
            Bshift = B[param_idx]
            param_idx += 1
        else:
            Bshift = 0
        if self.scale:
            Bscale = B[param_idx]
            param_idx += 1
        else:
            Bscale = 1
        Bconst = B[param_idx]
        param_idx += 1
        Betas = B[param_idx:]
        y = Bshift + Bscale * np.exp(Bconst + (X @ Betas))
        return np.nan_to_num(y, nan=1e298)

    def pipe_residuals(self, B, X, y):
        return self.pipe_(B, X) - y

    def fit(self, X, y):
        if self.form == 'expXB':
            self.pipe_ = self.expXB
            self.k = X.shape[1] + 1  # constant
        elif self.form == 'powXB':
            self.pipe_ = self.powXB
            self.k = X.shape[1] + 2  # constant & exponent
        elif self.form == 'linear':
            self.pipe_ = self.linear
            self.k = X.shape[1] + 1  # constant
        if not self.form == 'linear':
            if self.scale:
                self.k += 1
            if self.shift:
                self.k += 1
        # https://scipy-cookbook.readthedocs.io/items/robust_regression.html
        if self.robust:
            self.fit_est_ = least_squares(self.pipe_residuals, np.ones(self.k), args=(X, y), loss='soft_l1',
                                          f_scale=0.1, )  #
        else:
            self.fit_est_ = least_squares(self.pipe_residuals, np.ones(self.k), args=(X, y))  #
        return self

    """def score(self,X,y):
        #negative mse
        return mean_squared_error(self.predict(X),y)"""

    def predict(self, X):
        B = self.fit_est_.x
        return self.pipe_(B, X)


class FlexiblePipe(BaseEstimator, RegressorMixin, BaseHelper):
    name = "Flexible Pipeline"
    ptype = "flexpipe"
    description = "placeholder description for the flexible pipeline"
    hyper_parameters = {
        "do_prep": {
            "type": "str",
            "options": ['True', 'False'],
            "value": 'True'
        },
        "impute_strategy": {
            "type": "str",
            "options": ['impute_knn5'],
            "value": "impute_knn5"
        },
        "gridpoints": {
            "type": "int",
            "options": "1:8",
            "value": 4
        },
        "cv_strategy": {
            "type": "str",
            "options": ['quantile'],
            "value": 'quantile'
        },
        "groupcount": {
            "type": "int",
            "options": "1:5",
            "value": 5
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]
    def __init__(
            self, pipeline_id, do_prep='True', functional_form_search=False,
            prep_dict={'impute_strategy': 'impute_knn5'}, gridpoints=4,
            inner_cv=None, groupcount=None, bestT=False,
            cat_idx=None, float_idx=None, flex_kwargs={}
    ):
        self.pid = pipeline_id
        self.do_prep = do_prep == 'True'
        self.functional_form_search = functional_form_search
        self.gridpoints = gridpoints
        self.inner_cv = inner_cv
        self.groupcount = groupcount
        self.bestT = bestT
        self.cat_idx = cat_idx
        self.float_idx = float_idx
        self.prep_dict = prep_dict
        self.flex_kwargs = flex_kwargs
        BaseHelper.__init__(self)

    def get_pipe(self, ):
        if self.inner_cv is None:
            inner_cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        else:
            inner_cv = self.inner_cv

        steps = [
            ('scaler', StandardScaler()),
            ('select', ShrinkBigKTransformer(max_k=4)),
            ('reg', FlexibleEstimator(**self.flex_kwargs))
        ]
        if self.bestT:
            steps.insert(0, 'xtransform', ColumnBestTransformer(float_k=len(self.float_idx)))

        pipe = Pipeline(steps=steps)
        param_grid = {'select__k_share': np.linspace(0.2, 1, self.gridpoints * 2)}
        if self.functional_form_search:
            param_grid['reg__form'] = ['powXB', 'expXB']  # ,'linear']

        outerpipe = GridSearchCV(pipe, param_grid=param_grid)
        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)),
                     ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)

        return outerpipe

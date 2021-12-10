import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import SVR, LinearSVR
from vb_django.app.vb_transformers import ShrinkBigKTransformer, ColumnBestTransformer, LogMinus_T, Exp_T, LogMinPlus1_T, None_T, Log_T, LogP1_T, DropConst
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.base_helper import BaseHelper
from sklearn.pipeline import Pipeline
from dask_ml.model_selection import GridSearchCV, KFold


class RBFSVR(BaseEstimator, RegressorMixin, BaseHelper):
    name = "RBF SVR Pipeline"
    ptype = "rbfsvr"
    description = "placeholder description for the RBF SVR pipeline"
    hyper_parameters = {
        "do_prep": {
            "type": "str",
            "options": ['True', 'False'],
            "value": 'True'
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

    def __init__(self, pipeline_id=None, do_prep='True', prep_dict={'impute_strategy': 'impute_knn5'},
                 gridpoints=4, inner_cv=None, groupcount=None, impute_strategy=None, float_idx=None, cat_idx=None,
                 bestT=False, cv_splits=5, cv_repeats=2):
        self.pipeline_id = pipeline_id
        self.do_prep = do_prep == 'True' if type(do_prep) != bool else do_prep
        self.gridpoints = gridpoints
        self.inner_cv = inner_cv
        self.groupcount = groupcount
        self.bestT = bestT
        self.cat_idx = cat_idx
        self.float_idx = float_idx
        self.prep_dict = prep_dict
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.impute_strategy = impute_strategy
        if impute_strategy:
            self.prep_dict["impute_strategy"] = self.impute_strategy
        BaseHelper.__init__(self)

    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv = KFold(n_splits=self.cv_splits, random_state=0)
        else:
            inner_cv = self.inner_cv

        gridpoints = self.gridpoints
        param_grid = {'C': np.logspace(-2, 2, gridpoints), 'gamma': np.logspace(-2, 0.5, gridpoints)}
        steps = [
            ('scaler', StandardScaler()),
            ('reg', GridSearchCV(SVR(kernel='rbf', cache_size=10000, tol=1e-4, max_iter=5000), param_grid=param_grid))]
        if self.bestT:
            steps.insert(0, ('xtransform', ColumnBestTransformer(float_k=len(self.float_idx))))
        outerpipe = Pipeline(steps=steps)
        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)), ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe


class LinSVR(BaseEstimator, RegressorMixin, BaseHelper):
    name = "Linear SVR Pipeline"
    ptype = "linsvr"
    description = "placeholder description for the Linear SVR pipeline"
    hyper_parameters = {
        "do_prep": {
            "type": "str",
            "options": ['True', 'False'],
            "value": 'True'
        },
        "gridpoints": {
            "type": "int",
            "options": "1:8",
            "value": 4
        },
        "groupcount": {
            "type": "int",
            "options": "1:5",
            "value": 5
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id=None, do_prep='True', prep_dict={'impute_strategy': 'impute_knn5'},
                 gridpoints=4, inner_cv=None, groupcount=None, impute_strategy=None, bestT=False, cat_idx=None,
                 float_idx=None, cv_splits=5, cv_repeats=2):
        self.pipeline_id = pipeline_id
        self.do_prep = do_prep == 'True' if type(do_prep) != bool else do_prep
        self.gridpoints = gridpoints
        self.inner_cv = inner_cv
        self.groupcount = groupcount
        self.bestT = bestT
        self.cat_idx = cat_idx
        self.float_idx = float_idx
        self.prep_dict = prep_dict
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.impute_strategy = impute_strategy
        if impute_strategy:
            self.prep_dict["impute_strategy"] = self.impute_strategy
        BaseHelper.__init__(self)

    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv = KFold(n_splits=self.cv_splits, random_state=0)
        else:
            inner_cv = self.inner_cv

        gridpoints = self.gridpoints
        param_grid = {'C': np.logspace(-2, 4, gridpoints)}
        steps = [
            ('polyfeat', PolynomialFeatures(interaction_only=0, degree=2)), # create interactions among them

            ('drop_constant', DropConst()),
            ('shrink_k2', ShrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv, max_iter=64))),
            ('scaler', StandardScaler()),
            ('reg', GridSearchCV(LinearSVR(random_state=0, tol=1e-4, max_iter=1000), param_grid=param_grid))]
        if self.bestT:
            steps = [steps[0], ('xtransform', ColumnBestTransformer(float_k=len(self.float_idx))), *steps[1:]]
        outerpipe = Pipeline(steps=steps)
        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)), ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe

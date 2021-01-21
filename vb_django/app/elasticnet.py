import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from vb_django.app.vb_transformers import ColumnBestTransformer
from missing_val_transformer import missingValHandler
from vb_django.app.vb_cross_validator import RegressorQStratifiedCV
from vb_django.app.base_helper import BaseHelper


class ENet(BaseEstimator, TransformerMixin, BaseHelper):
    def __init__(self, pipeline_id):
        self.gridpoints = None
        self.cv_strategy = None
        self.groupcount = None
        self.float_idx = None
        self.cat_idx = None
        self.bestT = None
        self.impute_strategy = None
        self.flags = None
        BaseHelper.__init__(self, pipeline_id)

    def set_params(self, impute_strategy='impute_knn5', gridpoints=4, cv_strategy='quantile', groupcount=5,
                   float_idx=None, cat_idx=None, bestT=False, flags=None):
        self.impute_strategy = impute_strategy
        self.gridpoints = gridpoints
        self.cv_strategy = cv_strategy
        self.groupcount = groupcount
        self.float_idx = float_idx
        self.cat_idx = cat_idx
        self.bestT = bestT
        self.flags = flags

    def get_estimator(self):
        if self.cv_strategy:
            inner_cv = RegressorQStratifiedCV(n_splits=10, n_repeats=5, strategy=self.cv_strategy, random_state=0,
                                                 groupcount=self.groupcount)
        else:
            inner_cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        gridpoints = self.gridpoints
        param_grid = {'l1_ratio': np.logspace(-2, -.03, gridpoints)}
        steps = [
            ('prep', missingValHandler(strategy=self.impute_strategy, cat_idx=self.cat_idx)),
            ('scaler', StandardScaler()),
            ('reg', GridSearchCV(ElasticNetCV(cv=inner_cv, normalize=False), param_grid=param_grid))]

        if self.bestT:
            steps = [steps[0], ('xtransform', ColumnBestTransformer(float_k=len(self.float_idx))), *steps[1:]]
        pipe = Pipeline(steps=steps)
        return pipe

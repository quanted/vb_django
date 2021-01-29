import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from vb_django.app.vb_transformers import ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.vb_cross_validator import RegressorQStratifiedCV
from vb_django.app.base_helper import BaseHelper


class GBR(BaseEstimator, TransformerMixin, BaseHelper):
    name = "Gradient Boosting Regressor"
    ptype = "gbr"
    description = "GB builds an additive model in a forward stage-wise fashion; it allows for the optimization " \
                  "of arbitrary differentiable loss functions. In each stage a regression tree is fit on the " \
                  "negative gradient of the given loss function."
    hyper_parameters = {
        "impute_strategy": {
            "type": "str",
            "options": ['impute_knn5'],
            "value": "impute_knn5"
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id):
        self.bestT = False
        self.cat_idx = None
        self.float_idx = None
        self.impute_strategy = None
        super().__init__(pipeline_id)

    def set_params(self, hyper_parameters):
        # Validation of user specified impute_strategy
        if "impute_strategy" in hyper_parameters.keys():
            if hyper_parameters["impute_strategy"] in self.hyper_parameters["impute_strategy"]["options"]:
                self.impute_strategy = hyper_parameters["impute_strategy"]
            else:
                self.impute_strategy = self.hyper_parameters["impute_strategy"]["value"]
        else:
            self.impute_strategy = self.hyper_parameters["impute_strategy"]["value"]

    def get_estimator(self):
        steps = [
            ('prep', MissingValHandler(strategy=self.impute_strategy, cat_idx=self.cat_idx)),
            ('reg', GradientBoostingRegressor())
        ]
        if self.bestT:
            steps = [steps[0], ('xtransform', ColumnBestTransformer(float_k=len(self.float_idx))), *steps[1:]]
        return Pipeline(steps=steps)


class HGBR(BaseEstimator, TransformerMixin, BaseHelper):
    name = "Histogram Gradient Boosting Regressor"
    ptype = "hgbr"
    description = "Histogram-based Gradient Boosting Regression Tree."
    hyper_parameters = {
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id):
        self.cat_idx = None
        self.float_idx = None
        super().__init__(pipeline_id)

    def set_params(self, hyper_parameters):
        self.cat_idx = None
        self.float_idx = None

    def get_estimator(self):
        steps = [
            ('prep', MissingValHandler(strategy='pass-through', cat_idx=self.cat_idx)),
            ('reg', HistGradientBoostingRegressor())
        ]
        return Pipeline(steps=steps)

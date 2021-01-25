import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from vb_django.app.vb_transformers import ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.vb_cross_validator import RegressorQStratifiedCV
from vb_django.app.base_helper import BaseHelper


class ENet(BaseEstimator, TransformerMixin, BaseHelper):
    name = "Elastic Net Pipeline"
    ptype = "enet"
    description = "placeholder description for the enet pipeline"
    hyper_parameters = {
        "impute_strategy": {
            "type": "list",
            "options": ['impute_knn5'],
            "value": "impute_knn5"
        },
        "gridpoints": {
            "type": "int",
            "options": "1:5",
            "value": 4
        },
        "cv_strategy": {
            "type": "list",
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

    def __init__(self, pipeline_id):
        self.gridpoints = None
        self.cv_strategy = None
        self.groupcount = None
        self.float_idx = None
        self.cat_idx = None
        self.bestT = False
        self.impute_strategy = None
        self.flags = None
        super().__init__(pipeline_id)

    def set_params(self, hyper_parameters):
        # TODO: Update output to display the hyper-parameter values used.
        # Validation of user specified impute_strategy
        if "impute_strategy" in hyper_parameters.keys():
            if hyper_parameters["impute_strategy"] in self.hyper_parameters["impute_strategy"]["options"]:
                self.impute_strategy = hyper_parameters["impute_strategy"]
            else:
                self.impute_strategy = self.hyper_parameters["impute_strategy"]["value"]
        else:
            self.impute_strategy = self.hyper_parameters["impute_strategy"]["value"]
        # Validation of user specified gridpoints
        if "gridpoints" in hyper_parameters.keys():
            gp_range = self.hyper_parameters["gridpoints"]["options"].split(":")
            if int(gp_range[0]) <= int(hyper_parameters["gridpoints"]) <= int(gp_range[1]):
                self.gridpoints = int(hyper_parameters["gridpoints"])
            else:
                self.gridpoints = self.hyper_parameters["gridpoints"]["value"]
        else:
            self.gridpoints = self.hyper_parameters["gridpoints"]["value"]
        # Validation of user specified cv_strategy
        if "cv_strategy" in hyper_parameters.keys():
            if hyper_parameters["cv_strategy"] in self.hyper_parameters["cv_strategy"]["options"]:
                self.cv_strategy = hyper_parameters["cv_strategy"]
            else:
                self.cv_strategy = self.hyper_parameters["cv_strategy"]["value"]
        else:
            self.cv_strategy = self.hyper_parameters["cv_strategy"]["value"]
        # Validation of user specified groupcount
        if "groupcount" in hyper_parameters.keys():
            gp_range = self.hyper_parameters["groupcount"]["options"].split(":")
            if int(gp_range[0]) <= int(hyper_parameters["groupcount"]) <= int(gp_range[1]):
                self.groupcount = int(hyper_parameters["groupcount"])
            else:
                self.groupcount = self.hyper_parameters["groupcount"]["value"]
        else:
            self.groupcount = self.hyper_parameters["groupcount"]["value"]

    def get_estimator(self):
        if self.cv_strategy:
            inner_cv = RegressorQStratifiedCV(n_splits=10, n_repeats=5, strategy=self.cv_strategy, random_state=0,
                                                 groupcount=self.groupcount)
        else:
            inner_cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        gridpoints = self.gridpoints
        param_grid = {'l1_ratio': np.logspace(-2, -.03, gridpoints)}
        steps = [
            ('prep', MissingValHandler(strategy=self.impute_strategy, cat_idx=self.cat_idx)),
            ('scaler', StandardScaler()),
            ('reg', GridSearchCV(ElasticNetCV(cv=inner_cv, normalize=False), param_grid=param_grid))]

        if self.bestT:
            steps = [steps[0], ('xtransform', ColumnBestTransformer(float_k=len(self.float_idx))), *steps[1:]]
        pipe = Pipeline(steps=steps)
        return pipe

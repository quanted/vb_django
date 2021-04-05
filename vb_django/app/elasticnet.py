import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from vb_django.app.vb_transformers import ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.base_helper import BaseHelper


class ENet(BaseEstimator, TransformerMixin, BaseHelper):
    name = "Elastic Net Pipeline"
    ptype = "enet"
    description = "placeholder description for the enet pipeline"
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
        "groupcount": {
            "type": "int",
            "options": "1:5",
            "value": 5
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id, do_prep='True', prep_dict=None, impute_strategy=None,
                 gridpoints=4, inner_cv=None, groupcount=None,
                 float_idx=None, cat_idx=None, bestT=False):
        self.pid = pipeline_id
        self.do_prep = do_prep == 'True'
        self.gridpoints = gridpoints
        self.groupcount = groupcount
        self.float_idx = float_idx
        self.cat_idx = cat_idx
        self.bestT = bestT
        self.inner_cv = inner_cv
        self.prep_dict = {'impute_strategy': self.hyper_parameters["impute_strategy"]["value"]} if prep_dict is None else prep_dict
        if impute_strategy:
            self.prep_dict["impute_strategy"] = impute_strategy
        self.flags = None
        BaseHelper.__init__(self)

    def set_params(self, hyper_parameters):
        if hyper_parameters is None:
            return
        # TODO: Update output to display the hyper-parameter values used.
        # Validation of user specified impute_strategy
        if "impute_strategy" in hyper_parameters.keys():
            if hyper_parameters["impute_strategy"] in self.hyper_parameters["impute_strategy"]["options"]:
                self.impute_strategy = hyper_parameters["impute_strategy"]
        # Validation of user specified gridpoints
        if "gridpoints" in hyper_parameters.keys():
            gp_range = self.hyper_parameters["gridpoints"]["options"].split(":")
            if int(gp_range[0]) <= int(hyper_parameters["gridpoints"]) <= int(gp_range[1]):
                self.gridpoints = int(hyper_parameters["gridpoints"])
        # Validation of user specified cv_strategy
        if "cv_strategy" in hyper_parameters.keys():
            if hyper_parameters["cv_strategy"] in self.hyper_parameters["cv_strategy"]["options"]:
                self.cv_strategy = hyper_parameters["cv_strategy"]
        # Validation of user specified groupcount
        if "groupcount" in hyper_parameters.keys():
            gp_range = self.hyper_parameters["groupcount"]["options"].split(":")
            if int(gp_range[0]) <= int(hyper_parameters["groupcount"]) <= int(gp_range[1]):
                self.groupcount = int(hyper_parameters["groupcount"])
        if "prep_dict" in hyper_parameters.keys():
            self.prep_dict = hyper_parameters["prep_dict"]
        if "inner_cv" in hyper_parameters.keys():
            self.inner_cv = hyper_parameters["inner_cv"]

    def get_pipe(self, ):
        if self.inner_cv is None:
            inner_cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        else:
            inner_cv = self.inner_cv
        gridpoints = self.gridpoints
        l1_ratio = 1 - np.logspace(-2, -.03, gridpoints)
        steps = [
            ('scaler', StandardScaler()),
            ('reg', ElasticNetCV(cv=inner_cv, normalize=False, l1_ratio=l1_ratio))]

        if self.bestT:
            steps.insert(0, ('xtransform', ColumnBestTransformer(float_k=len(self.float_idx))))
        outerpipe = Pipeline(steps=steps)
        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)),
                     ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe

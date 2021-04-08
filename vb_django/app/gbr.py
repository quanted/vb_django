from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from vb_django.app.vb_transformers import ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.base_helper import BaseHelper


class GBR(BaseEstimator, TransformerMixin, BaseHelper):
    name = "Gradient Boosting Regressor"
    ptype = "gbr"
    description = "GB builds an additive model in a forward stage-wise fashion; it allows for the optimization " \
                  "of arbitrary differentiable loss functions. In each stage a regression tree is fit on the " \
                  "negative gradient of the given loss function."
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
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id=None, prep_dict=None, do_prep='True', impute_strategy=None,
                 inner_cv=None, bestT=False, cat_idx=None, float_idx=None):
        self.pid = pipeline_id
        self.do_prep = do_prep == 'True'
        self.bestT = bestT
        self.cat_idx = cat_idx
        self.float_idx = float_idx
        self.inner_cv = inner_cv
        self.prep_dict = {'impute_strategy': self.hyper_parameters["impute_strategy"]["value"]} if prep_dict is None else prep_dict
        if impute_strategy:
            self.prep_dict["impute_strategy"] = impute_strategy
        BaseHelper.__init__(self)

    def set_params(self, hyper_parameters):
        if hyper_parameters is None:
            return
        # Validation of user specified impute_strategy
        if "impute_strategy" in hyper_parameters.keys():
            if hyper_parameters["impute_strategy"] in self.hyper_parameters["impute_strategy"]["options"]:
                self.prep_dict["impute_strategy"] = hyper_parameters["impute_strategy"]

    def get_pipe(self):
        if self.inner_cv is None:
            inner_cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv = self.inner_cv

        param_grid = {'max_depth': list(range(1, 3)),
                      'n_estimators': [75, 100]
                      }
        steps = [('reg', GridSearchCV(GradientBoostingRegressor(random_state=0), param_grid=param_grid, cv=inner_cv, n_jobs=1))]
        # steps = [('reg', GradientBoostingRegressor(random_state=0))]

        if self.bestT:
            steps.insert(0, 'xtransform', ColumnBestTransformer(float_k=len(self.float_idx)))
        outerpipe = Pipeline(steps=steps)
        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)),
                     ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe


class HGBR(BaseEstimator, TransformerMixin, BaseHelper):
    name = "Histogram Gradient Boosting Regressor"
    ptype = "hgbr"
    description = "Histogram-based Gradient Boosting Regression Tree."
    hyper_parameters = {
        "do_prep": {
            "type": "str",
            "options": ['True', 'False'],
            "value": 'True'
        },
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id, do_prep='True', prep_dict=None):
        self.pid = pipeline_id
        self.do_prep = do_prep == 'True'
        self.cat_idx = None
        self.float_idx = None
        self.prep_dict = prep_dict
        BaseHelper.__init__(self)

    def set_params(self, hyper_parameters):
        if hyper_parameters is None:
            return
        self.cat_idx = None
        self.float_idx = None
        self.prep_dict = hyper_parameters["prep_dict"]

    def get_pipe(self):
        steps = [
            ('reg', HistGradientBoostingRegressor())
        ]
        outerpipe = Pipeline(steps=steps)
        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=dict(impute_strategy='pass-through',cat_idx=self.prep_dict['cat_idx']))),
                     ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe

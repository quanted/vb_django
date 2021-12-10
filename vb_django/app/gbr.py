from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from vb_django.app.vb_transformers import ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.base_helper import BaseHelper
import dask_ml.model_selection as dms


class GBR(BaseEstimator, RegressorMixin, BaseHelper):
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
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id=None, do_prep=True, prep_dict={'impute_strategy': 'impute_knn5'}, inner_cv=None,
                 bestT=False, cat_idx=None, float_idx=None, est_kwargs=None, cv_splits=5, cv_repeats=2):

        self.pipeline_id = pipeline_id
        self.do_prep = do_prep == 'True' if type(do_prep) != bool else do_prep
        self.bestT = bestT
        self.cat_idx = cat_idx
        self.float_idx = float_idx
        self.inner_cv = inner_cv
        self.prep_dict = prep_dict
        self.est_kwargs = est_kwargs
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        # self.impute_strategy = impute_strategy
        # if impute_strategy:
        #     self.prep_dict["impute_strategy"] = self.impute_strategy
        BaseHelper.__init__(self)

    def set_params(self, hyper_parameters):
        pass
        # if hyper_parameters is None:
        #     return
        # # Validation of user specified impute_strategy
        # if "impute_strategy" in hyper_parameters.keys():
        #     if hyper_parameters["impute_strategy"] in self.hyper_parameters["impute_strategy"]["options"]:
        #         self.prep_dict["impute_strategy"] = hyper_parameters["impute_strategy"]

    def get_pipe(self):
        if self.inner_cv is None:
            inner_cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv = self.inner_cv
        if self.est_kwargs is None:
            self.est_kwargs = {'max_depth': [3, 4], 'n_estimators': [64, 128]}
        hyper_param_dict, gbr_params = self.extractParams(self.est_kwargs)
        if not 'random_state' in gbr_params:
            gbr_params['random_state'] = 0
        steps = [('reg', dms.GridSearchCV(GradientBoostingRegressor(**gbr_params), param_grid=hyper_param_dict, cv=inner_cv))]
        if self.bestT:
            steps.insert(0, 'xtransform', ColumnBestTransformer(float_k=len(self.float_idx)))
        outerpipe = Pipeline(steps=steps)
        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)), ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe


class HGBR(BaseEstimator, RegressorMixin, BaseHelper):
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

    def __init__(self, pipeline_id=None, do_prep='True', prep_dict=None):
        self.pipeline_id = pipeline_id
        self.do_prep = do_prep == 'True' if type(do_prep) != bool else do_prep
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
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)),
                     ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe

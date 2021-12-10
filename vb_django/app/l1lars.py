from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoLarsCV
from vb_django.app.vb_transformers import ShrinkBigKTransformer, ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.base_helper import BaseHelper
from sklearn.pipeline import Pipeline
import dask_ml.model_selection as dms


class L1Lars(BaseEstimator, RegressorMixin, BaseHelper):
    name = "L1Lars Pipeline"
    ptype = "l1lars"
    description = "placeholder description for the l1lars pipeline"
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
        },
        "max_n_alphas": {
            "type": "int",
            "options": "1:2000",
            "value": 1000
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id=None, do_prep='True', prep_dict={'impute_strategy': 'impute_knn5'},
                 gridpoints=4, inner_cv=None, groupcount=None, impute_strategy=None,
                 bestT=False, cat_idx=None, float_idx=None, max_n_alphas=1000, cv_splits=5, cv_repeats=2):
        self.pipeline_id = pipeline_id
        self.do_prep = do_prep == 'True' if type(do_prep) != bool else do_prep
        self.gridpoints = gridpoints
        self.inner_cv = inner_cv
        self.groupcount = groupcount
        self.bestT = bestT
        self.cat_idx = cat_idx
        self.float_idx = float_idx
        self.prep_dict = prep_dict
        self.max_n_alphas = max_n_alphas
        self.impute_strategy = impute_strategy
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        if impute_strategy:
            self.prep_dict["impute_strategy"] = self.impute_strategy
        BaseHelper.__init__(self)

    def get_pipe(self):
        if self.inner_cv is None:
            inner_cv = RepeatedKFold(n_splits=self.cv_splits, n_repeats=self.cv_repeats, random_state=0)
        else:
            inner_cv = self.inner_cv

        steps = [('reg', LassoLarsCV(cv=inner_cv, max_n_alphas=self.max_n_alphas))]
        if self.bestT:
            steps.insert(0, 'xtransform', ColumnBestTransformer(float_k=len(self.float_idx)))
        outerpipe = Pipeline(steps=steps)

        if self.do_prep:
            steps = [('prep', MissingValHandler(prep_dict=self.prep_dict)), ('post', outerpipe)]
            outerpipe = Pipeline(steps=steps)
        return outerpipe

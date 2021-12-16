# from vb_django.utilities import update_status, save_model, update_pipeline_metadata
# from sklearn import metrics as skm
# import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
# from vb_django.app.vb_transformers import ColumnBestTransformer
from vb_django.app.missing_val_transformer import MissingValHandler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import StackingRegressor
from dask.distributed import Client

import logging
import warnings
# import joblib
# import os
# import socket
import django
django.setup()
from sklearn.utils import parallel_backend

warnings.simplefilter('ignore')
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)


class BaseHelper:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.n_, self.k_ = X.shape
        self.pipe_ = self.get_pipe()
        try:
            with parallel_backend('loky'):  # test case 30min with dask backend in vb_helper
                self.pipe_.fit(X, y)
        except Exception as e:
            print(f'error fitting pipeline, error: {e}')
        return self

    def transform(self, X, y=None):
        return self.pipe_.transform(X, y)

    def score(self, X, y):
        return self.pipe_.score(X, y)

    def predict(self, X):
        return self.pipe_.predict(X)

    def extractParams(self, param_dict, prefix=''):
        hyper_param_dict = {}
        static_param_dict = {}
        for param_name, val in param_dict.items():
            if type(val) is list:
                hyper_param_dict[prefix + param_name] = val
            else:
                static_param_dict[param_name] = val
        return hyper_param_dict, static_param_dict


class NullModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, x, y, w=None):
        pass

    def predict(self, x, ):
        if len(x.shape) > 1:
            return np.mean(x, axis=1)
        return x


class MultiPipe(BaseEstimator, RegressorMixin, BaseHelper):
    def __init__(self, pipeline_id=None, pipelist=None, prep_dict=None, stacker_estimator=None):
        self.pipeline_id = pipeline_id
        self.pipelist = pipelist
        self.prep_dict = self.getPrepDict(prep_dict)
        self.stacker_estimator = stacker_estimator

        self.pipe_ = self.get_pipe()  # formerly inside basehelper
        BaseHelper.__init__(self)

    def getPrepDict(self, prep_dict):
        if self.pipelist is None:
            print('empty MultiPipe!')
            return None
        if prep_dict is None:
            for pname, pdict in self.pipelist:
                if 'prep_dict' in pdict['pipe_kwargs']:
                    return pdict['pipe_kwargs']['prep_dict']
            assert False, f'no prep_dict found in any pipedict within self.pipelist:{self.pipelist}'
        else:
            return prep_dict

    def get_pipe(self):
        try:
            pipe_n = len(self.pipelist)
            est_pipes = [(p[0], p[1]['pipe'](**p[1]['pipe_kwargs'])) for i, p in enumerate(self.pipelist)]
            final_e = self.stacker_estimator
            steps = [
                ('prep', MissingValHandler(prep_dict=self.prep_dict)),
                ('post',
                 make_pipeline(StackingRegressor(est_pipes, passthrough=False, final_estimator=final_e, n_jobs=-1, verbose=5), verbose=True))]
            return Pipeline(steps=steps)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_pipe_names(self):
        pipe_names = [pipe_tup[0] for pipe_tup in self.pipelist]
        return pipe_names

    def get_individual_post_pipes(self, names=None):
        if names is None:
            names = self.get_pipe_names()
        if type(names) is str:
            names = [names]
        pipe_dict = {}
        # logger.warn(f"Pipeline ID: {self.pipeline_id}, Attributes of stackingregressor: {self.pipe_['post']['stackingregressor']}")
        # for estimator_o in self.pipe_['post']['stackingregressor'].estimators:
        #     name = estimator_o[0]
        #     estimator = estimator_o[1]
        #     # logger.warn(f"Pipeline ID: {self.pipeline_id}, estimator name: {name}, estimator: {estimator}")
        #     pipe_dict[name] = estimator
        for name in names:
            pipe_dict[name] = self.pipe_['post']['stackingregressor'].named_estimators_[name]
        return pipe_dict

    def get_prep(self):
        return self.pipe_['prep']

    def build_individual_fitted_pipelines(self, names=None):
        pipe_dict = self.get_individual_post_pipes(names=names)
        prep = self.get_prep()
        fitted_ipipe_dict = {}  # i for individual
        for pname, pipe in pipe_dict.items():
            fitted_steps = [('prep', prep), ('post', pipe)]
            fitted_ipipe_dict[pname] = FCombo(fitted_steps)
        return fitted_ipipe_dict


class FCombo(BaseEstimator, RegressorMixin):
    # Frankenstein fitted combos
    def __init__(self, fitted_steps):
        self.fitted_steps = fitted_steps

    def fit(self, X, y):
        assert False, 'fit called! this is a fitted combo!'

    def predict(self, X):
        step_n = len(self.fitted_steps)
        step_names = [step_tup[0] for step_tup in self.fitted_steps]
        Xt = X.copy()
        for s in range(0, step_n - 1):
            Xt = self.fitted_steps[s][1].transform(Xt)
        return self.fitted_steps[-1][1].predict(Xt)

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.compose import TransformedTargetRegressor
from vb_django.app.vb_transformers import ShrinkBigKTransformer, LogMinus_T, Exp_T, LogMinPlus1_T, None_T, LogP1_T, DropConst
from vb_django.app.missing_val_transformer import MissingValHandler
from vb_django.app.vb_cross_validator import RegressorQStratifiedCV

import pandas as pd
import numpy as np
import warnings
import time
import logging

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

logger = logging.getLogger("vb_dask")
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')


class LinearRegressionVB:

    def __init__(self):
        pass


class LinearRegressionAutomatedVB:
    name = "Automated Linear Regression"
    id = "lra"
    description = "Automated pipeline with feature evaluation and selection for a linear regression estimator."

    def __init__(self, test_split=0.2, cv_folds=10, cv_reps=10, seed=42, one_out=False, cv_strategy='q-balanced', group_count=5):
        self.hyperparameters = {
            'test_split': 0.2,
            'cv_folds': 10,
            'cv_reps': 10,
            'random_seed': 42,
            'one_out': False,
            'cv_strategy': 'q-balanced',
            'group_count': 5
        }
        self.start_time = time.time()
        self.test_split = test_split
        self.cv_folds = cv_folds
        self.cv_reps = cv_reps
        self.gridpoints = 3
        self.seed = seed
        self.one_out = one_out
        self.cv_strategy = cv_strategy
        self.group_count = group_count

        self.k = None
        self.n = None
        self.max_k = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.train_score = None
        self.test_score = None

        self.lr_estimator = None
        self.attr = None
        self.results = None
        self.residuals = None

    def validate_h_params(self, parameters):
        for h in self.hyperparameters.keys():
            if h in parameters.keys():
                self.hyperparameters[h] = parameters[h]
        self.test_split = float(self.hyperparameters['test_split'])
        self.cv_folds = int(self.hyperparameters['cv_folds'])
        self.cv_reps = int(self.hyperparameters['cv_reps'])
        self.seed = int(self.hyperparameters['random_seed'])
        self.one_out = bool(self.hyperparameters['one_out'])
        self.cv_strategy = str(self.hyperparameters['cv_strategy'])
        self.group_count = int(self.hyperparameters['group_count'])

    def set_data(self, x, y):
        if self.one_out:
            self.x_train, self.x_test, self.y_train, self.y_test = (x, x, y, y)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x, y,
                test_size=self.test_split,
                random_state=self.seed
            )
        self.n, self.k = self.x_train.shape
        self.max_k = min([self.n // 2, int(1.5 * self.k)])

    @ignore_warnings(category=ConvergenceWarning)
    def set_pipeline(self):
        warnings.filterwarnings('ignore')
        gridpoints = self.gridpoints
        transformer_list = [None_T(), LogP1_T()]
        steps = [
            ('prep', MissingValHandler()),
            ('scaler', StandardScaler()),
            ('shrink_k1', ShrinkBigKTransformer()),  # retain a subset of the best original variables
            ('polyfeat', PolynomialFeatures(interaction_only=0)),  # create interactions among them
            ('drop_constant', DropConst()),
            ('shrink_k2', ShrinkBigKTransformer(selector=ElasticNet())),  # pick from all of those options
            ('reg', LinearRegression(fit_intercept=1))]

        X_T_pipe = Pipeline(steps=steps)
        if self.cv_strategy == 'q-balanced':
            inner_cv = RegressorQStratifiedCV(n_splits=10, n_repeats=2, random_state=0, group_count=self.group_count)

        else:
            inner_cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)

        Y_T_X_T_pipe = Pipeline(steps=[('ttr', TransformedTargetRegressor(regressor=X_T_pipe))])
        Y_T__param_grid = {
            'ttr__transformer': transformer_list,
            'ttr__regressor__polyfeat__degree': [2],
            'ttr__regressor__shrink_k2__selector__alpha': np.logspace(-2, 2, gridpoints),
            'ttr__regressor__shrink_k2__selector__l1_ratio': np.linspace(0, 1, gridpoints),
            'ttr__regressor__shrink_k1__max_k': [self.k // gp for gp in range(1, gridpoints + 1, 2)],
            'ttr__regressor__prep__strategy': ['impute_middle', 'impute_knn_10']
        }
        lin_reg_Xy_transform = GridSearchCV(Y_T_X_T_pipe, param_grid=Y_T__param_grid, cv=inner_cv, n_jobs=11)

        self.lr_estimator = lin_reg_Xy_transform
        self.lr_estimator.fit(self.x_train, self.y_train)
        self.train_score = self.lr_estimator.score(self.x_train, self.y_train)
        self.test_score = self.lr_estimator.score(self.x_test, self.y_test)
        self.attr = pd.DataFrame(self.lr_estimator.cv_results_)
        # generates the model that is saved
        logger.info("Total execution time: {} sec".format(round(time.time() - self.start_time, 3)))

    def predict(self, x_test=None):
        # obsolete within dask stack
        x_test = x_test if x_test else self.x_test
        self.results = self.lr_estimator.predict(x_test)
        self.residuals = self.results - self.y_test.to_numpy().flatten()

    def get_info(self):
        details = {
            "name": LinearRegressionAutomatedVB.name,
            "id": LinearRegressionAutomatedVB.id,
            "description": LinearRegressionAutomatedVB.description,
            "hyperparameters": self.hyperparameters
        }
        return details

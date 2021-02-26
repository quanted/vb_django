from vb_django.app.vb_cross_validator import RegressorQStratifiedCV
from vb_django.utilities import update_status, save_model, update_pipeline_metadata
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from vb_django.app.base_helper import BaseHelper
import numpy as np
import time
import copy


# TODO: Add logging and model saving


class CrossValidatePipeline:
    name = "Cross Validator Pipeline"
    ptype = "cvpipe"
    description = "The cross-validate pipeline takes one or more estimators and performs cross-validate on the possible input parameters of the pipeline."
    hyper_parameters = {
        "test_share": {
            "type": "float",
            "options": "0.0:1.0",
            "value": 0.2
        },
        "cv_folds": {
            "type": "int",
            "options": "2:10",
            "value": 5
        },
        "cv_reps": {
            "type": "int",
            "options": "1:10",
            "value": 2
        },
        "cv_strategy": {
            "type": "str",
            "options": ['quantile'],
            "value": None
        },
        "groupcount": {
            "type": "int",
            "options": "1:5",
            "value": None
        },
        "seed": {
            "type": "int",
            "options": "0:",
            "value": 42
        },
        "scorer_list": {
            "type": "list",
            "options": ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            "value": ['neg_mean_squared_error', 'neg_mean_absolute_error']
        },
        "max_k": {
            "type": "int",
            "options": "0:10",
            "value": None
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id):
        update_status(
            pipeline_id,
            "Initializing pipeline",
            "1/7",
            log="Pipeline: {}, Initializing pipeline. Step: 1/7".format(pipeline_id)
        )
        self.time0 = time.time()
        self.pid = pipeline_id
        self.test_share = self.hyper_parameters["test_share"]["value"]
        self.cv_folds = int(self.hyper_parameters["cv_folds"]["value"])
        self.cv_reps = int(self.hyper_parameters["cv_reps"]["value"])
        self.cv_count = self.cv_folds * self.cv_reps
        self.cv_strategy = self.hyper_parameters["cv_strategy"]["value"]
        self.cv_groupcount = self.hyper_parameters["groupcount"]["value"]
        self.seed = self.hyper_parameters["seed"]["value"]
        self.scorer_list = self.hyper_parameters["scorer_list"]["value"]
        self.max_k = self.hyper_parameters["max_k"]["value"]

        self.estimators = {}
        self.models = {}
        self.results = {}
        self.cv_results = {}
        self.cv_yhat_dict = {}

        self.x = None
        self.y = None
        self.cat_idx, self.cat_vars = None, None
        self.float_idx = None

    def set_params(self, parameters: dict = None):
        update_status(
            self.pid,
            "Settings and validating hyper-parameters",
            "2/7",
            log="Pipeline: {}, Setting hyper-parameters. Step: 2/7".format(self.pid)
        )
        if parameters is None:
            return
        for k, v in parameters.items():
            if k == "test_share":
                r = self.hyper_parameters["test_share"]["options"].split(":")
                if float(r[0]) <= v <= float(r[1]):
                    self.test_share = v
            elif k == "cv_folds":
                r = self.hyper_parameters["cv_folds"]["options"].split(":")
                if int(r[0]) <= v <= int(r[1]):
                    self.cv_folds = int(v)
            elif k == "cv_reps":
                r = self.hyper_parameters["cv_reps"]["options"].split(":")
                if int(r[0]) <= v <= int(r[1]):
                    self.cv_reps = int(v)
            elif k == "cv_strategy":
                r = self.hyper_parameters["cv_strategy"]["options"]
                if v in r:
                    self.cv_strategy = v
            elif k == "groupcount":
                r = self.hyper_parameters["groupcount"]["options"].split(":")
                if v in int(r[0]) <= v <= int(r[1]):
                    self.cv_groupcount = int(v)
            elif k == "seed":
                if v >= 0:
                    self.seed = int(v)
            elif k == "scorer_list":
                scorers = []
                r = self.hyper_parameters["test_share"]["options"]
                for vi in v:
                    if vi in r:
                        scorers.append(vi)
                self.scorer_list = scorers
            elif k == "max_k":
                r = self.hyper_parameters["max_k"]["options"].split(":")
                if v in int(r[0]) <= v <= int(r[1]):
                    self.max_k = int(v)
        self.cv_count = self.cv_reps * self.cv_folds

    def set_data(self, x, y):
        update_status(
            self.pid,
            "Setting features and target datasets",
            "3/7",
            log="Pipeline: {}, Setting features and target datasets. Step: 3/7".format(self.pid)
        )
        self.x = x
        self.y = y
        # self.cat_idx, self.cat_vars = zip(
        #     *[(i, var) for i, (var, dtype) in enumerate(dict(x.dtypes).items()) if dtype == 'object'])
        # self.float_idx = [i for i in range(x.shape[1]) if i not in self.cat_idx]

    def set_estimators(self, estimators):
        update_status(
            self.pid,
            "Settings estimators and their parameters",
            "4/7",
            log="Pipeline: {}, Settings estimators and their parameters. Step: 4/7".format(self.pid)
        )
        self.estimators = estimators
        self.models = {key: val for key, val in estimators.items()}

    def train_test_split(self):
        return train_test_split(
            self.x, self.y,
            test_size=self.test_share, random_state=self.seed)

    def is_valid(self):
        if self.estimators is not None and self.models is not None and self.x is not None and self.y is not None:
            return True
        return False

    def fit(self):
        update_status(
            self.pid,
            "Fitting estimator(s)",
            "5/7",
            log="Pipeline: {}, Fitting estimator(s). Step: 5/7".format(self.pid)
        )
        if not self.is_valid():
            return False
        x_train, x_test, y_train, y_test = self.train_test_split()
        for n, est in self.estimators.items():
            est.fit(x_train, y_train)
            self.results[n + "-train"] = est.scoreXY(x_train, y_train)
            if x_test is not None:
                self.results[n + "-test"] = est.scoreXY(x_test, y_test)
        return True

    def run_cross_validate(self, n_jobs=4):
        update_status(
            self.pid,
            "Executing cross validate on estimator(s)",
            "6/7",
            log="Pipeline: {}, Executing cross validate on estimator(s). Step: 6/7".format(self.pid)
        )
        cv_results = {}
        for estimator_name, model in self.models.items():
            cv = self.get_cv()
            e = model.get_estimator()
            model_i = cross_validate(
                e, self.x, self.y, return_estimator=True,
                scoring=self.scorer_list, cv=cv, n_jobs=n_jobs)
            cv_results[estimator_name] = model_i
        self.cv_results = cv_results

    def get_cv(self):
        if self.cv_strategy is None:
            return RepeatedKFold(
                n_splits=self.cv_folds, n_repeats=self.cv_reps, random_state=self.seed)
        else:
            return RegressorQStratifiedCV(
                n_splits=self.cv_folds, n_repeats=self.cv_reps,
                random_state=self.seed, groupcount=self.cv_groupcount, strategy=self.cv_strategy)

    def save(self):
        n, k = self.x.shape
        runtime = time.time() - self.time0
        update_pipeline_metadata(self, runtime, n)
        m = save_model(self, pipeline_id=self.pid)
        if m:
            update_status(
                self.pid,
                "Completed and model saved",
                "7/7",
                log="Pipeline: {}, Model: {}, Completed and model saved. Step: 7/7".format(self.pid, m.id)
            )
        else:
            update_status(
                self.pid,
                "Completed pipeline, error saving model",
                "-7/7",
                log="Pipeline: {}, Completed pipeline, error saving model. Step: -7/7".format(self.pid)
            )
        # model_n = len(self.models.keys())
        # saved_n = 0
        # for n, m in self.cv_results.items():
        #     model = save_model(m, self.pid, replace=False)
        #     if model is not None:
        #         saved_n += 1
        # update_status(
        #     self.pid,
        #     "Completed and all model(s) saved. {} of {} models saved.".format(saved_n, model_n),
        #     "5/5",
        #     "Pipeline: {}, Multiple Models Saved, Completed and model saved".format(self.pid)
        # )

    def predict(self, features=None):
        train_idx_list, test_idx_list = zip(*list(self.get_cv().split(self.x, self.y)))
        n, k = self.x.shape
        data_idx = np.arange(n)
        yhat_dict = {}
        scores = {}
        for idx, (estimator_name, result) in enumerate(self.cv_results.items()):
            scores[estimator_name] = []
            yhat_dict[estimator_name] = []
            for r in range(self.cv_reps):
                yhat = np.empty([n, ])
                rows = []
                for s in range(self.cv_folds):  # s for split
                    m = r*self.cv_folds+s
                    cv_est = result['estimator'][m]
                    test_rows = test_idx_list[m]
                    yhat[test_rows] = cv_est.predict(self.x.iloc[test_rows])
                    rows.append(BaseHelper.score(self.y[test_rows], yhat[test_rows]))
                scores[estimator_name].append(rows)
                yhat_dict[estimator_name].append(yhat)
        self.cv_yhat_dict = yhat_dict
        return yhat_dict, scores

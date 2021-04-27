import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold
from vb_django.app.base_helper import MultiPipe, FCombo, NullModel
from vb_django.app.vb_cross_validator import RegressorQStratifiedCV
from vb_django.utilities import update_status, save_model, update_pipeline_metadata
import pandas as pd
import logging
import copy
import time

logger = logging.getLogger("vb_dask:task")
logger.setLevel(logging.DEBUG)


class VBLogger:
    def __init__(self, pipeline_id):
        self.id = pipeline_id
        self.i = 0

    def log(self, status, n, i=None, log=None, error: bool = False):
        if not log:
            log = status
        log_i = 0
        if i is None:
            self.i += 1
            log_i = self.i
        if i:
            log_i = i
        if error:
            log_i = -1*self.i
        update_status(
            self.id,
            status,
            "{}/{}".format(log_i, n),
            log="Pipeline: {}, {} Step: {}/{}".format(self.id, log, log_i, n)
        )


class VBHelper:
    name = "VB Helper"
    ptype = "vbhelper"
    description = "Parent pipeline class, containing global CV"
    hyper_parameters = {
        "test_share": {
            "type": "float",
            "options": "0.0, 1.0",
            "value": 0.2
        },
        "cv_folds": {
            "type": "int",
            "options": "1:10",
            "value": 5
        },
        "cv_reps": {
            "type": "int",
            "options": "1:5",
            "value": 2
        },
        "random_state": {
            "type": "int",
            "options": "0:inf",
            "value": 0
        },
        "cv_strategy": {
            "type": "string",
            "options": ['quantile'],
            "value": 'quantile'
        },
        "cn_n_jobs": {
            "type": "int",
            "options": "1:8",
            "value": 4
        },
        "run_stacked": {
            "type": "bool",
            "options": "True,False",
            "value": True
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id, test_share=0.2, cv_folds=5, cv_reps=2, random_state=0, cv_strategy=None, run_stacked='True',
                 cv_n_jobs=4):
        self.id = pipeline_id
        self.logger = VBLogger(self.id)
        self.step_n = 16
        self.logger.log("Initializating global input parameters.", self.step_n)
        self.cv_n_jobs = cv_n_jobs
        self.cv_strategy = cv_strategy
        self.run_stacked = run_stacked == 'True'
        self.test_share = test_share
        self.rs = random_state

        self.scorer_list = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        self.max_k = None
        self.estimator_dict = {}
        self.model_dict = {}

        self.project_CV_dict = {}
        self.X_df = None
        self.y_df = None
        self.X_test = None
        self.y_test = None
        self.cat_idx, self.cat_vars = None, None
        self.float_idx = None
        self.cv_results = None
        self.cv_yhat_dict = None
        self.cv_y_yhat_dict = None
        self.cv_err_dict = {}
        self.cv_score_dict_means = {}
        self.cv_score_dict = {}

        self.predictive_models = {}
        self.predictive_model_type = "single"

        self.setProjectCVDict(cv_folds, cv_reps, cv_strategy)
        self.logger.log("Initialization complete.", self.step_n)

    def setProjectCVDict(self, cv_folds, cv_reps, cv_strategy):
        if cv_folds is None:
            cv_folds = 10
        if cv_reps is None:
            cv_reps = 1
        cv_count = cv_reps * cv_folds
        self.project_CV_dict = {
            'cv_folds': cv_folds,
            'cv_reps': cv_reps,
            'cv_count': cv_count,
            'cv_strategy': cv_strategy
        }

    def setData(self, X_df, y_df):
        self.logger.log("Input data setup started...", self.step_n)

        # Data shuffling
        shuf = np.arange(y_df.shape[0])
        np.random.seed(self.rs)
        np.random.shuffle(shuf)
        X_df = X_df.iloc[shuf]
        y_df = y_df.iloc[shuf]

        if self.test_share > 0:
            self.X_df, self.X_test, self.y_df, self.y_test = train_test_split(
                X_df, y_df, test_size=self.test_share, random_state=self.rs)
        else:
            self.X_df = X_df
            self.y_df = y_df
            self.X_test = None
            self.y_test = None
        try:
            self.cat_idx, self.cat_vars = zip(*[(i, var) for i, (var, dtype) in enumerate(dict(X_df.dtypes).items()) if dtype == 'object'])
        except ValueError as e:
            # No categorical variables
            self.cat_idx = []
            self.cat_vars = []
        self.float_idx = [i for i in range(X_df.shape[1]) if i not in self.cat_idx]
        self.logger.log("Input data setup complete.", self.step_n)

    def setPipeDict(self, pipe_dict):
        self.logger.log("Estimator(s) setup started...", self.step_n)
        if self.run_stacked:
            # logger.info("Running stacked pipeline")
            self.estimator_dict = {'multi_pipe': {'pipe': MultiPipe, 'pipe_kwargs': {
                'pipelist': list(pipe_dict.items())}}}  # list...items() creates a list of tuples...
        else:
            self.estimator_dict = pipe_dict
        self.logger.log("Estimator(s) setup complete.", self.step_n)

    def setModelDict(self, pipe_dict=None):
        self.logger.log("Model(s) initialization started...", self.step_n)
        if pipe_dict is None:
            pipe_dict = {}
            if self.run_stacked:
                e_list = self.estimator_dict['multi_pipe']['pipe_kwargs']['pipelist']
            else:
                e_list = []
                for n, p in self.estimator_dict.items():
                    e_list.append([n, p])
            for e in e_list:
                pipe_dict[e[0]] = e[1]
        model_dict = {}
        for key, val in pipe_dict.items():
            e_args = val['pipe_kwargs']
            e_args["pipeline_id"] = self.id
            initialized = False
            pipe = None
            while not initialized:
                try:
                    pipe = val['pipe'](**e_args)
                    initialized = True
                except TypeError as te:
                    e_name = str(te).split("'")[1]
                    logger.error("Removing '{}' from pipeline arguments".format(e_name))
                    if e_name in e_args.keys() and len(e_args) > 0:
                        del e_args[e_name]
                    else:
                        raise Exception
                except Exception as ex:
                    logger.error("Model Initialization error: {}".format(ex))
                    initialized = True
            model_dict[key] = pipe
        self.model_dict = model_dict
        self.logger.log("Model(s) initialization complete.", self.step_n)
        return self.model_dict

    def fitEstimators(self):
        self.logger.log("Estimator(s) fitting started...", self.step_n)
        for key, model in self.model_dict.items():
            self.model_dict[key].fit(self.X_df, self.y_df)
        self.logger.log("Estimator(s) fitting complete.", self.step_n)

    def runCrossValidate(self, verbose=False):
        self.logger.log("Cross-validate started...", self.step_n)
        n_jobs = self.cv_n_jobs
        if verbose:
            logger.info("CV N-JOBS: {}".format(n_jobs))
        cv_results = {}
        new_cv_results = {}
        cv = self.getCV()
        for estimator_name, model in self.model_dict.items():
            start = time.time()
            model_i = cross_validate(
                model, self.X_df, self.y_df, return_estimator=True,
                scoring=self.scorer_list, cv=cv, n_jobs=n_jobs, error_score='raise')
            end = time.time()
            if verbose:
                logger.info(f"SCORES - {estimator_name},{[(scorer,np.mean(model_i[f'test_{scorer}'])) for scorer in self.scorer_list]}, runtime: {(end-start)/60} min.")
                logger.info(f"MODELS - {estimator_name},{model_i}")
            cv_results[estimator_name] = model_i
        if self.run_stacked:
            for est_name, result in cv_results.items():
                if type(result['estimator'][0]) is MultiPipe:
                    new_results = {}
                    for mp in result['estimator']:
                        for est_n, m in mp.build_individual_fitted_pipelines().items():
                            if not est_n in new_results:
                                new_results[est_n] = []
                            new_results[est_n].append(m)
                            # lil_x = self.X_df.iloc[0:2]
                            # logger.info(f'est_n yhat test: {m.predict(lil_x)}')
                    for est_n in new_results:
                        if est_n in cv_results:
                            est_n += '_fcombo'
                        new_cv_results[est_n] = {'estimator': new_results[est_n]}
            cv_results = {**new_cv_results, **cv_results}
            if verbose:
                logger.info("CV Results: {}".format(cv_results))
        self.cv_results = cv_results
        self.logger.log("Cross-validate complete.", self.step_n)

    def getCV(self, cv_dict=None):
        if cv_dict is None:
            cv_dict = self.project_CV_dict
        cv_reps = cv_dict['cv_reps']
        cv_folds = cv_dict['cv_folds']
        cv_strategy = cv_dict['cv_strategy']
        if cv_strategy is None:
            return RepeatedKFold(
                n_splits=cv_folds, n_repeats=cv_reps, random_state=self.rs)
        else:
            assert type(cv_strategy) is tuple, f'expecting tuple for cv_strategy, got {cv_strategy}'
            cv_strategy, cv_groupcount = cv_strategy
            return RegressorQStratifiedCV(
                n_splits=cv_folds, n_repeats=cv_reps,
                random_state=self.rs, groupcount=cv_groupcount, strategy=cv_strategy)

    def predictCVYhat(self, ):
        self.logger.log("Predicting Y-Hat values...", self.step_n)
        cv_reps = self.project_CV_dict['cv_reps']
        cv_folds = self.project_CV_dict['cv_folds']
        train_idx_list, test_idx_list = zip(*list(self.getCV().split(self.X_df, self.y_df)))
        n, k = self.X_df.shape
        y = self.y_df.to_numpy()
        data_idx = np.arange(n)
        yhat_dict = {}
        err_dict = {}
        cv_y_yhat_dict = {}
        for idx, (estimator_name, result) in enumerate(self.cv_results.items()):
            yhat_dict[estimator_name] = []
            cv_y_yhat_dict[estimator_name] = []
            err_dict[estimator_name] = []
            for r in range(cv_reps):
                yhat = np.empty([n, ])
                err = np.empty([n, ])
                for s in range(cv_folds):  # s for split
                    m = r * cv_folds + s
                    cv_est = result['estimator'][m]
                    # logger.info("ESTIMATOR CHECK: {}".format(cv_est))
                    test_rows = test_idx_list[m]
                    yhat_arr = cv_est.predict(self.X_df.iloc[test_rows])
                    yhat[test_rows] = yhat_arr
                    err[test_rows] = y[test_rows] - yhat[test_rows]
                    cv_y_yhat_dict[estimator_name].append((self.y_df.iloc[test_rows].to_numpy(), yhat_arr))
                yhat_dict[estimator_name].append(yhat)
                err_dict[estimator_name].append(err)
        self.cv_yhat_dict = yhat_dict
        self.cv_y_yhat_dict = cv_y_yhat_dict
        self.cv_err_dict = err_dict
        self.logger.log("Predicting Y-Hat values complete.", self.step_n)

    def evaluate(self):
        full_results = self.arrayDictToListDict(
            {
                'y': self.y_df.to_list(),
                'cv_yhat': self.cv_yhat_dict,
                'cv_score': self.cv_score_dict,
                'project_cv': self.project_CV_dict,
                'cv_model_descrip': {}          #not developed
            }
        )
        return full_results

    def arrayDictToListDict(self, arr_dict):
        assert type(arr_dict) is dict, f'expecting dict but type(arr_dict):{type(arr_dict)}'
        list_dict = {}
        for key, val in arr_dict.items():
            if type(val) is dict:
                list_dict[key] = self.arrayDictToListDict(val)
            elif type(val) is np.ndarray:
                list_dict[key] = val.tolist()
            elif type(val) is list and type(val[0]) is np.ndarray:
                list_dict[key] = [v.tolist() for v in val]
            else:
                list_dict[key] = val
        return list_dict

    def buildCVScoreDict(self):
        self.logger.log("Building CV Score Dict.", self.step_n)
        if self.cv_yhat_dict is None:
            self.predictCVYhat()
        cv_results = self.cv_results
        scorer_list = self.scorer_list
        cv_score_dict = {}
        cv_score_dict_means = {}
        y = self.y_df
        for idx, (pipe_name, result) in enumerate(cv_results.items()):
            model_idx_scoredict = {}
            for scorer in scorer_list:
                scorer_kwarg = f'test_{scorer}'
                a_scorer = lambda y, yhat: get_scorer(scorer)(NullModel(), yhat, y)     #b/c get_scorer wants (est,x,y)
                score = np.array([a_scorer(y, yhat) for yhat in self.cv_yhat_dict[pipe_name]])
                model_idx_scoredict[scorer] = score
            cv_score_dict[pipe_name] = model_idx_scoredict
            model_idx_mean_scores = {scorer: np.mean(scores) for scorer, scores in model_idx_scoredict.items()}
            cv_score_dict_means[pipe_name] = model_idx_mean_scores
        self.cv_score_dict_means = cv_score_dict_means
        self.cv_score_dict = cv_score_dict
        self.logger.log("Building CV Score Dict complete.", self.step_n)

    def refitPredictiveModels(self, selected_models: dict, verbose: bool=False):
        # TODO: Add different process for each possible predictive_model_type
        self.logger = VBLogger(self.id)
        self.logger.log("Refitting specified models for prediction...", 4)

        X_df = self.X_df if self.X_test is None else self.X_test
        y_df = self.y_df if self.y_test is None else self.y_test

        predictive_models = {}
        for name, indx in selected_models.items():
            logger.info(f"Name: {name}, Index: {indx}")
            if name in self.cv_results.keys():
                predictive_models[f"{name}-{indx}"] = copy.copy(self.cv_results[name]["estimator"][0])
        logger.info(f"Models:{predictive_models}")
        for name, est in predictive_models.items():
            predictive_models[name] = est.fit(X_df, y_df)
        self.predictive_models = predictive_models
        self.logger.log("Refitting model for prediction complete.", 4)

    def predict(self, x_df: pd.DataFrame):
        results = {}
        for name, est in self.predictive_models.items():
            results[name] = est.predict(x_df)
        return results

    def save(self, n=None, model_id=None):
        n = self.step_n if n is None else n
        self.logger.log("Saving results...", n)
        m = save_model(self, model_id=model_id, pipeline_id=self.id)
        if m:
            self.logger.log("Saving results complete", n)
        else:
            self.logger.log("Unable to save results", n, error=True)

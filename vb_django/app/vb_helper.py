import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold
from vb_django.app.base_helper import MultiPipe, FCombo, NullModel
from vb_django.app.vb_cross_validator import RegressorQStratifiedCV
from vb_django.app.missing_val_transformer import MissingValHandler
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

    def log(self, status, n, i=None, log=None, error: bool = False, message: str = None):
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
            log="Pipeline: {}, {} Step: {}/{}".format(self.id, log, log_i, n),
            message=message
        )


class VBHelper:
    name = "VB Helper"
    ptype = "vbhelper"
    description = "Parent pipeline class, containing global CV"
    hyper_parameters = {
        "drop_duplicates": {
            "type": "bool",
            "options": "True,False",
            "value": False
        },
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
        },
        "nan_threshold": {
            "type": "float",
            "options": "0.0, 1.0",
            "value": 0.99
        },
        "shuffle": {
            "type": "bool",
            "options": "True,False",
            "value": True
        }
    }
    metrics = ["total_runs", "avg_runtime", "avg_runtime/n"]

    def __init__(self, pipeline_id, test_share=0.2, cv_folds=5, cv_reps=2, random_state=0, cv_strategy=None, run_stacked="True",
                 cv_n_jobs=4, drop_duplicates="False", nan_threshold=0.99, shuffle="True"):
        self.id = pipeline_id
        self.logger = VBLogger(self.id)
        self.step_n = 16
        self.logger.log("Initializating global input parameters.", self.step_n, message="Cross validation")
        self.cv_n_jobs = cv_n_jobs
        self.cv_strategy = cv_strategy
        self.run_stacked = run_stacked == "True"
        self.test_share = test_share
        self.rs = random_state
        self.drop_duplicates = drop_duplicates == "True"
        self.nan_threshold = nan_threshold
        self.shuffle = shuffle == "True"
        # self.predict_n = predict_n
        # self.predict_idx = None

        self.dep_var_name = None
        self.scorer_list = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        self.max_k = None
        self.estimator_dict = {}
        self.model_dict = {}
        self.original_pipe_dict = None

        self.project_CV_dict = {}
        self.X_df_start_order = None
        self.y_df_start_order = None
        self.X_df = None
        self.y_df = None
        self.X_test = None
        self.y_test = None
        self.X_predict = None
        self.y_predict = None
        self.cat_idx, self.cat_vars = None, None
        self.float_idx = None
        self.cv_results = None
        self.cv_yhat_dict = None
        self.cv_y_yhat_dict = None
        self.cv_err_dict = {}
        self.cv_score_dict_means = {}
        self.cv_score_dict = {}

        self.model_averaging_weights = {}
        self.prediction_models = {}
        self.prediction_model_type = "single"

        self.setProjectCVDict(cv_folds, cv_reps, cv_strategy)
        self.logger.log("Initialization complete.", self.step_n, message="Cross validation")

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
        self.logger.log("Input data setup started...", self.step_n, message="Cross validation")
        self.dep_var_name = y_df.columns.to_list()[0]
        X_df, y_df = self.checkData(X_df, y_df)

        self.X_df_start_order = X_df
        self.y_df_start_order = y_df

        # predict_select = np.random.choice(np.arange(y_df.shape[0]), size=self.predict_n, replace=False)
        # self.predict_idx = predict_select
        # self.X_predict = X_df.iloc[predict_select]
        # X_df.drop(index=predict_select, inplace=True)
        # self.y_predict = y_df.iloc[predict_select]
        # y_df.drop(index=predict_select, inplace=True)

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
        if 'object' in list(dict(X_df.dtypes).values()):
            self.cat_idx, self.cat_vars = zip(
                *[(i, var) for i, (var, dtype) in enumerate(dict(X_df.dtypes).items()) if dtype == 'object'])
        else:
            self.cat_idx = []
            self.cat_vars = []
        self.float_idx = [i for i in range(X_df.shape[1]) if i not in self.cat_idx]
        self.logger.log("Input data setup complete.", self.step_n, message="Cross validation")

    def checkData(self, X_df, y_df):
        X_dtype_dict = dict(X_df.dtypes)
        for var, dtype in X_dtype_dict.items():
            if str(dtype)[:3] == 'int':
                X_df.loc[:, var] = X_df.loc[:, var].astype('float')
        data_df = X_df.copy()
        data_df.loc['dependent_variable', :] = y_df.loc[:, self.dep_var_name]
        X_duplicates = X_df.duplicated()
        full_duplicates = data_df.duplicated()
        full_dup_sum = full_duplicates.sum()
        X_dup_sum = X_duplicates.sum()
        logger.info(f"Pipeline ID: {self.id} - Check Data - # of duplicate rows of data: {full_dup_sum}, # of duplicates rows of X: {X_dup_sum}")

        if self.drop_duplicates:
            if self.drop_duplicates.lower() in ['yx', 'xy', 'full']:
                X_df = X_df[~full_duplicates]
                logger.info(f"Pipeline ID: {self.id} - Check Data - # of duplicate Xy rows dropped: {full_dup_sum}")
            elif self.drop_duplicates.lower() == 'x':
                X_df = X_df[~X_duplicates]
                logger.info(f"Pipeline ID: {self.id} - Check Data - # of duplicate X rows dropped: {X_dup_sum}")
            else:
                logger.info(f"Pipeline ID: {self.id} - Check Data - unexpected drop_duplicates:{self.drop_duplicates}")

        drop_cols = X_df.columns[X_df.isnull().sum(axis=0) / X_df.shape[0] > self.nan_threshold]
        if len(drop_cols) > 0:
            logger.info(f"Pipeline ID: {self.id} - Check Data - columns to drop: {drop_cols}")
            X_df.drop(drop_cols, axis=1, inplace=True)

        return X_df, y_df

    @staticmethod
    def saveFullFloatXy(X_df, y_df, X_df_s, y_df_s):
        mvh = MissingValHandler({
            'impute_strategy': 'impute_knn5'  # 'pass-through'
        })
        mvh = mvh.fit(X_df)
        X_float = mvh.transform(X_df_s)
        X_float_df = pd.DataFrame(data=X_float, columns=mvh.get_feature_names(input_features=X_df_s.columns.to_list()))
        X_json_s = X_float_df.to_json()
        y_json_s = y_df_s.to_json()
        X_nan_bool_s = X_df_s.isnull().to_json()
        summary_data = {'full_float_X': X_json_s, 'full_y': y_json_s, 'X_nan_bool': X_nan_bool_s}
        return summary_data

    def setPipeDict(self, pipe_dict):
        self.logger.log("Estimator(s) setup started...", self.step_n, message="Setting pipeline dictionaries")
        self.original_pipe_dict = pipe_dict
        if self.run_stacked:
            # logger.info("Running stacked pipeline")
            self.estimator_dict = {'stacking_reg': {'pipe': MultiPipe, 'pipe_kwargs': {'pipelist': list(pipe_dict.items())}}}
        else:
            self.estimator_dict = pipe_dict
        self.logger.log("Estimator(s) setup complete.", self.step_n, message="Completed setup of pipeline dictionaries")

    def setModelDict(self, pipe_dict=None):
        self.logger.log("Model(s) initialization started...", self.step_n, message="Setting model dictionaries")
        if pipe_dict is None:
            self.model_dict = {key: val['pipe'](**val['pipe_kwargs']) for key, val in self.estimator_dict.items()}
        else:
            self.model_dict = {key: val['pipe'](**val['pipe_kwargs']) for key, val in pipe_dict.items()}
        # if pipe_dict is None:
        #     pipe_dict = {}
        #     if self.run_stacked:
        #         e_list = self.estimator_dict['stacking_reg']['pipe_kwargs']['pipelist']
        #     else:
        #         e_list = []
        #         for n, p in self.estimator_dict.items():
        #             e_list.append([n, p])
        #     for e in e_list:
        #         pipe_dict[e[0]] = e[1]
        # model_dict = {}
        # for key, val in pipe_dict.items():
        #     e_args = val['pipe_kwargs']
        #     e_args["pipeline_id"] = self.id
        #     initialized = False
        #     pipe = None
        #     while not initialized:
        #         try:
        #             pipe = val['pipe'](**e_args)
        #             initialized = True
        #         except TypeError as te:
        #             e_name = str(te).split("'")[1]
        #             print("Removing '{}' from pipeline arguments".format(e_name))
        #             if e_name in e_args.keys() and len(e_args) > 0:
        #                 del e_args[e_name]
        #             else:
        #                 raise Exception
        #         except Exception as ex:
        #             print("Model Initialization error: {}".format(ex))
        #             initialized = True
        #     model_dict[key] = pipe
        # self.model_dict = model_dict
        self.logger.log("Model(s) initialization complete.", self.step_n, message="Completed setup of model dictionaries")
        return self.model_dict

    def fitEstimators(self):
        self.logger.log("Estimator(s) fitting started...", self.step_n, message="Fitting estimators")
        for key, model in self.model_dict.items():
            y = self.y_df.squeeze()
            self.model_dict[key].fit(self.X_df, y)
        self.logger.log("Estimator(s) fitting complete.", self.step_n, message="Completed fitting estimators")

    def runCrossValidate(self, verbose=False):
        self.logger.log("Cross-validate started...", self.step_n, message="Running cross validation")
        n_jobs = self.cv_n_jobs
        if verbose:
            logger.info("CV N-JOBS: {}".format(n_jobs))
        cv_results = {}
        new_cv_results = {}
        cv = self.getCV()
        n_jobs = 1
        for pipe_name, model in self.model_dict.items():
            start = time.time()
            model_i = cross_validate(
                model, self.X_df, self.y_df.iloc[:, 0], return_estimator=True,
                scoring=self.scorer_list, cv=cv, n_jobs=n_jobs)
            end = time.time()
            if verbose:
                logger.info(f"SCORES - {pipe_name},{[(scorer,np.mean(model_i[f'test_{scorer}'])) for scorer in self.scorer_list]}, runtime: {(end-start)/60} min.")
                logger.info(f"MODELS - {pipe_name},{model_i}")
            cv_results[pipe_name] = model_i
        if self.run_stacked:
            for est_name, result in cv_results.items():
                if type(result['estimator'][0]) is MultiPipe:
                    new_results = {}
                    for mp in result['estimator']:
                        for est_n, m in mp.build_individual_fitted_pipelines().items():
                            if not est_n in new_results:
                                new_results[est_n] = []
                            new_results[est_n].append(m)
                    for est_n in new_results:
                        if est_n in cv_results:
                            est_n += '_fcombo'
                        new_cv_results[est_n] = {'estimator': new_results[est_n]}
            cv_results = {**new_cv_results, **cv_results}
            if verbose:
                logger.info("CV Results: {}".format(cv_results))
        self.cv_results = cv_results
        self.logger.log("Cross-validate complete.", self.step_n, message="Completed cross validation")

    def getCV(self, cv_dict=None):
        if cv_dict is None:
            cv_dict = self.project_CV_dict
        cv_reps = cv_dict['cv_reps']
        cv_folds = cv_dict['cv_folds']
        cv_strategy = cv_dict['cv_strategy']
        if cv_strategy is None:
            return RepeatedKFold(n_splits=cv_folds, n_repeats=cv_reps, random_state=self.rs)
        else:
            assert type(cv_strategy) is tuple, f'expecting tuple for cv_strategy, got {cv_strategy}'
            cv_strategy, cv_groupcount = cv_strategy
            return RegressorQStratifiedCV(
                n_splits=cv_folds, n_repeats=cv_reps,
                random_state=self.rs, groupcount=cv_groupcount, strategy=cv_strategy)

    def predictCVYhat(self, ):
        self.logger.log("Predicting Y-Hat values...", self.step_n, message="Predicting Y values from CV")
        cv_reps = self.project_CV_dict['cv_reps']
        cv_folds = self.project_CV_dict['cv_folds']
        train_idx_list, test_idx_list = zip(*list(self.getCV().split(self.X_df, self.y_df)))
        n, k = self.X_df.shape
        y = self.y_df.to_numpy()[:, 0]
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
                    _X_df = self.X_df.iloc[test_rows]
                    yhat_arr = cv_est.predict(_X_df)
                    yhat[test_rows] = yhat_arr
                    err[test_rows] = y[test_rows] - yhat[test_rows]
                    cv_y_yhat_dict[estimator_name].append((self.y_df.iloc[test_rows].to_numpy(), yhat_arr))
                yhat_dict[estimator_name].append(yhat)
                err_dict[estimator_name].append(err)
        self.cv_yhat_dict = yhat_dict
        self.cv_y_yhat_dict = cv_y_yhat_dict
        self.cv_err_dict = err_dict
        self.logger.log("Predicting Y-Hat values complete.", self.step_n, message="Completed Y CV predictions")

    def evaluate(self):
        full_results = self.arrayDictToListDict(
            {
                'y': self.y_df.iloc[:, 0].to_list(),
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
        self.logger.log("Building CV Score Dict.", self.step_n, message="Building CV Score dictionary")
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
        self.logger.log("Building CV Score Dict complete.", self.step_n, message="Completed building CV Score dictionary")

    def refitPredictionModels(self, selected_models: dict, verbose: bool=False):
        self.logger = VBLogger(self.id)
        self.logger.log("Refitting specified models for prediction...", 4, message="Refitting models for prediction")

        X_df = self.X_df
        y_df = self.y_df

        prediction_models = {}
        for name, indx in selected_models.items():
            # logger.info(f"Name: {name}, Index: {indx}")
            try:
                prediction_models[name] = self.model_dict[name]#copy.copy(self.cv_results[name]["estimator"][indx])
            except KeyError:
                pipe_i = self.original_pipe_dict[name]
                prediction_models[name] = pipe_i['pipe'](**pipe_i['pipe_kwargs'])
            # if name in self.cv_results.keys():
            #     prediction_models[f"{name}"] = copy.copy(self.cv_results[name]["estimator"][0])
        for name, est in prediction_models.items():
            prediction_models[name] = est.fit(X_df, y_df)
        self.prediction_models = prediction_models
        self.logger.log("Refitting model for prediction complete.", 4, message="Completed model refitting")

    def setModelAveragingWeights(self):
        pipe_names = list(self.prediction_models.keys())
        model_count = len(self.prediction_models)
        if self.prediction_model_type == "average":
            self.model_averaging_weights = {
                pipe_names[i]: {
                    scorer: 1 / model_count for scorer in self.scorer_list
                } for i in range(model_count)
            }
            return
        elif self.prediction_model_type == "cv-weighted":
            totals = {
                "neg_mean_squared_error": 0,
                "neg_mean_absolute_error": 0,
                "r2": 0
            }
            value = {
                "neg_mean_squared_error": 0,
                "neg_mean_absolute_error": 0,
                "r2": 0
            }
            for name, p in self.cv_score_dict_means.items():
                if name in pipe_names:  # leave out non-selected pipelines
                    totals["neg_mean_squared_error"] += 1 / abs(p["neg_mean_squared_error"])
                    totals["neg_mean_absolute_error"] += 1 / abs(p["neg_mean_absolute_error"])
                    totals["r2"] += p["r2"] if p["r2"] > 0 else 0
            weights = {}
            for pipe_name in pipe_names:
                weights[pipe_name] = {}
                for scorer, score in self.cv_score_dict_means[pipe_name].items():
                    # logger.warning(f"Scorer: {scorer}, Score: {score}")
                    if "neg" == scorer[:3]:
                        w = (1 / (abs(score))) / totals[scorer]
                    elif scorer == "r2":
                        score = score if score > 0 else 0
                        w = score / totals[scorer]
                    else:
                        w = abs(score) / totals[scorer]
                    weights[pipe_name][scorer] = w
            self.model_averaging_weights = weights

    def getPredictionValues(self, x_df):
        prediction_results = self.predict(x_df)
        test_results = self.predict(self.X_test)
        collection = {
            'prediction_results': prediction_results,
            'test_results': test_results,
            'test_y': self.y_test
        }
        return collection

    def predictandSave(self, X_predict=None, scorer=None):
        if scorer is None:
            scorer = self.scorer_list[0]
        if X_predict is None:
            X_predict = self.X_predict
        if not str(X_predict.index.to_list()[0])[:7].lower() == 'predict':
            X_predict.index = [f'predict-{idx}' for idx in X_predict.index]
        yhat = self.predict(X_predict)
        yhat_cv = self.predict(X_predict, model_type='cross_validation')

        predictresult = {
            'yhat': yhat['prediction'][scorer].to_json(),
            'cv_yhat': [yhat_cv_by_scorer[scorer].to_json() for yhat_cv_by_scorer in yhat_cv['prediction']],
            'X_predict': X_predict.to_json(),
            'selected_models': [*self.prediction_models]}
        return predictresult

    def predict(self, X_df_predict: pd.DataFrame, model_type: str = 'predictive'):
        if self.model_averaging_weights is None:
            self.setModelAveragingWeights()
        results = {}

        if model_type == 'cross_validation':
            wtd_yhats = [{scorer: np.zeros(X_df_predict.shape[0]) for scorer in self.scorer_list} for _ in
                         range(self.project_CV_dict['cv_count'])]
        else:
            wtd_yhats = {scorer: np.zeros(X_df_predict.shape[0]) for scorer in self.scorer_list}
        for name, est in self.prediction_models.items():
            if model_type == 'predictive':
                results[name] = est.predict(X_df_predict)
                for scorer, weights in self.model_averaging_weights[name].items():
                    wtd_yhats[scorer] += weights * results[name]
            elif model_type == 'cross_validation':
                results[name] = []
                for cv_i in range(self.project_CV_dict['cv_count']):
                    model_cv_i = self.cv_results[name]['estimator'][cv_i]
                    results[name].append(model_cv_i.predict(X_df_predict))
                    for scorer, weights in self.model_averaging_weights[name].items():
                        wtd_yhats[cv_i][scorer] += weights * results[name][cv_i]
        wtd_yhats_dfs = {}
        if model_type == 'predictive':
            wtd_yhats_dfs = {scorer: pd.DataFrame(data=arr[:, None], index=X_df_predict.index, columns=['yhat']) for
                             scorer, arr in wtd_yhats.items()}
        elif model_type == 'cross_validation':
            wtd_yhats_dfs = [
                {scorer: pd.DataFrame(data=arr[:, None], index=X_df_predict.index, columns=['yhat']) for scorer, arr in
                 cv_dict.items()} for cv_dict in wtd_yhats]
        results["weights"] = self.model_averaging_weights
        results["prediction"] = wtd_yhats_dfs
        return results

    def get_test_predictions(self):
        test_results = {}
        if self.X_test is None:
            return test_results
        for name, est in self.prediction_models.items():
            r = {
                "y": self.y_test.to_list(),
                "yhat": est.predict(self.X_test)
            }
            test_results[name] = r
        return test_results

    def save(self, n=None, model_id=None, message=None):
        n = self.step_n if n is None else n
        self.logger.log("Saving results...", n)
        m = save_model(self, model_id=model_id, pipeline_id=self.id)
        if m:
            self.logger.log("Saving results complete", n, message=message)
        else:
            self.logger.log("Unable to save results", n, error=True)

import pandas as pd
import numpy as np
import re


class VBPlotter:
    def __init__(self):
        pass

    def setData(self, data_dict):
        # data_dict looks like:
        """{
            'y':self.y_df,
            'cv_yhat':self.cv_yhat_dict,
            'cv_score':self.cv_score_dict,
            'project_cv':self.project_CV_dict,
            'cv_model_descrip':None #not developed
        }"""
        self.data_dict = data_dict

        y = np.array(self.data_dict['y'])
        self.y = y
        self.project_CV_dict = self.data_dict['project_cv']
        self.cv_reps = self.project_CV_dict['cv_reps']
        cv_yhat_dict = {}
        for key, val in data_dict['cv_yhat'].items():
            if type(val[0]) is list:
                cv_yhat_dict[key] = [np.array(v) for v in val]
            else:
                cv_yhat_dict[key] = np.array(val)
        self.cv_yhat_dict = cv_yhat_dict
        self.cv_score_dict = {key: [np.array(v) for v in val] for key, val in data_dict['cv_score'].items()}
        yhat_stack_dict = self.stackCVYhat()
        self.yhat_stack_dict = yhat_stack_dict  # stack reps into a single column
        self.y = y
        self.cv_score_dict = data_dict['cv_score']
        self.setScoreDict()

    def setPredictData(self, predictresults, loc_row=None):
        self.ypredict = pd.read_json(predictresults['yhat'])

        self.cv_ypredict = [pd.read_json(cv_i) for cv_i in predictresults['cv_yhat']]
        self.selected_estimators = predictresults['selected_models']
        if not loc_row is None:
            if not re.search('predict-', str(loc_row)):
                loc_row = f'predict-{loc_row}'
            self.ypredict = self.ypredict.loc[[loc_row]]
            self.cv_ypredict = [ser.loc[[loc_row]] for ser in self.cv_ypredict]

    def plotCVYhatVsY(self, single_plot=True, include_all_cv=True, regulatory_standard=False, decision_criteria=False,
                      ypredict=False, cv_ypredict=False, estimators='all', true_y=None):
        # true y on horizontal axis, yhat on vertical axis
        data = {}
        yhat_stack_dict = self.yhat_stack_dict
        y = self.y

        y_stack = np.concatenate([y for _ in range(self.cv_reps)], axis=0)
        if single_plot:
            data["y"] = y
        for e, (est_name, yhat_stack) in enumerate(self.yhat_stack_dict.items()):
            data[est_name] = {}
            if not estimators == 'all':
                if estimators == 'selected':
                    if not est_name in self.selected_estimators: continue
                else:
                    return {}
            if ypredict or cv_ypredict:
                all_y = np.concatenate([y, yhat_stack], axis=0)
                ymin = all_y.min()
                ymax = all_y.max()
            if not single_plot:
                data[est_name]['y'] = y
            if include_all_cv:
                data[est_name]["y_stack"] = y_stack
                data[est_name]["yhat_stack"] = yhat_stack

            if ypredict:
                yhat_df = self.ypredict
                data[est_name]["ymin"] = ymin
                data[est_name]["ymax"] = ymax
            if cv_ypredict:
                cv_yhat_list = self.cv_ypredict
                data["cv_yhat_list"] = cv_yhat_list
            if not true_y is None:
                data["true_y"] = {}
                for i, idx in enumerate(self.ypredict.index):
                    y_idx = ''.join(re.split('-', idx)[1:])
                    y = true_y.loc[y_idx]
                    data["true_y"][y_idx] = y
        return data

    def stackCVYhat(self):
        # makes a single column of all yhats across cv iterations for graphing
        # returns a dictionary with model/estimator/pipeline name as the key
        y = self.y
        yhat_stack_dict = {}
        for e, (est_name, yhat_list) in enumerate(self.cv_yhat_dict.items()):
            yhat_stack = np.concatenate(yhat_list, axis=0)
            yhat_stack_dict[est_name] = yhat_stack
        return yhat_stack_dict

    def setScoreDict(self):
        scorer_score_dict = {}
        for pipe_name, score_dict in self.cv_score_dict.items():
            for scorer, score_arr in score_dict.items():
                if not scorer in scorer_score_dict:
                    scorer_score_dict[scorer] = {}
                scorer_score_dict[scorer][pipe_name] = score_arr
        self.score_dict = scorer_score_dict

    def plotCVYhat(self, single_plot=True, include_all_cv=True):
        data = {}
        y = self.y
        data["y"] = y
        n = y.shape[0]
        y_sort_idx = np.argsort(y)  # other orderings could be added
        y_sort_idx_stack = np.concatenate([y_sort_idx for _ in range(self.cv_reps)], axis=0)
        if single_plot:
            data["y_sort_idx"] = y_sort_idx
        data["y_sort_idx_stack"] = y_sort_idx_stack
        data["yhat_stack"] = {}
        for e, (est_name, yhat_stack) in enumerate(self.yhat_stack_dict.items()):
            data["yhat_stack"][est_name] = yhat_stack.reshape(self.cv_reps, n).mean(axis=0)[y_sort_idx]
        return data

    def plotBoxWhiskerCVScores(self, ):
        data = {}
        scorers = []
        ests = None
        for s_idx, (scorer, pipe_scores) in enumerate(self.score_dict.items()):
            df = pd.DataFrame(pipe_scores)
            s_data = {
                "min": df.min(),
                "f25": df.quantile(q=0.25),
                "f50": df.quantile(q=0.5),
                "f75": df.quantile(q=0.75),
                "max": df.max()
            }
            ests = df.columns
            scorers.append(scorer)
            data[scorer] = s_data
        data["scorers"] = scorers
        data["estimators"] = ests
        return data

    def plotCVScores(self, sort=1):
        data = {}
        for s_idx, (scorer, pipe_scores) in enumerate(self.score_dict.items()):
            data[scorer] = {}
            df = pd.DataFrame(pipe_scores)
            if sort:
                for col in df:
                    data[scorer][col] = df[col].sort_values(ignore_index=True)
        return data

from vb_django.utilities import update_status, save_model, update_pipeline_metadata
from sklearn import metrics as skm
import time
import warnings

warnings.simplefilter('ignore')


class BaseHelper:
    def __init__(self, pipeline_id):
        self._pipeline_id = pipeline_id if pipeline_id is not None else -1
        update_status(
            self._pipeline_id,
            "Initializing pipeline",
            "1/5",
            log="Pipeline: {}, Initializing pipeline. Step: 1/5".format(self._pipeline_id)
        )

    # def get_estimator(self):
    #     return self.est_

    def set_params(self, hyper_parameters):
        update_status(
            self._pipeline_id,
            "Setting hyper-parameters",
            "2/5",
            log="Pipeline: {}, Setting hyper-parameters. Step: 2/5".format(self._pipeline_id)
        )
        self.est_.set_params(hyper_parameters)
        return self

    def fit(self, X, y):
        time0 = time.time()
        update_status(
            self._pipeline_id,
            "Fitting pipeline estimator",
            "3/5",
            log="Pipeline: {}, Fitting pipeline estimator. Step: 3/5".format(self._pipeline_id)
        )
        self.n_, self.k_ = X.shape
        self.est_ = self.get_estimator()
        self.est_.fit(X, y)
        runtime = time.time() - time0
        update_pipeline_metadata(self, runtime, self.n_)
        update_status(
            self._pipeline_id,
            "Completed fitting pipeline estimator",
            "4/5",
            log="Pipeline: {}, Completed fitting pipeline estimator. Step: 4/5".format(self._pipeline_id)
        )
        return self

    def transform(self, X, y=None):
        return self.est_.transform(X, y)

    @staticmethod
    def score(y_true, y_predict):
        metrics = {
            "explained_variance_score": skm.explained_variance_score(y_true, y_predict),
            "max_error": skm.max_error(y_true, y_predict),
            "mean_absolute_error": skm.mean_absolute_error(y_true, y_predict),
            "mean_squared_error": skm.mean_squared_error(y_true, y_predict),
            "mean_squared_log_error": skm.mean_squared_log_error(y_true, y_predict),
            "median_absolute_error": skm.median_absolute_error(y_true, y_predict),
            "r^2": skm.r2_score(y_true, y_predict),
            "mean_poisson_deviance": skm.mean_poisson_deviance(y_true, y_predict),
            "mean_gamma_deviance": skm.mean_gamma_deviance(y_true, y_predict),
            "mean_tweedie_deviance": skm.mean_tweedie_deviance(y_true, y_predict)
        }
        metadata = {
            "explained_variance_score": {
                "description": "Explained variance of the true and predicted values.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score",
                "range": "(0:1], 1 is best score"
            },
            "max_error":  {
                "description": "The max_error function computes the maximum residual error, "
                               "a metric that captures the worst case error between the predicted value and the true value.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#max-error",
                "range": "0 is a perfectly fitted model"
            },
            "mean_absolute_error": {
                "description": "The mean_absolute_error function computes mean absolute error, a risk metric "
                               "corresponding to the expected value of the absolute error loss or -norm loss.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error",
                "range": "0 is a model without error"
            },
            "mean_squared_error": {
                "description": "The mean_squared_error function computes mean square error, a risk metric "
                               "corresponding to the expected value of the squared (quadratic) error or loss.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error",
                "range": "[0:inf), 0 is a model without error"
            },
            "mean_squared_log_error": {
                "description": "The mean_squared_log_error function computes a risk metric corresponding to the "
                               "expected value of the squared logarithmic (quadratic) error or loss.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error",
                "range": "[0:inf), 0 is a model without error"
            },
            "median_absolute_error": {
                "description": "The median_absolute_error is particularly interesting because it is robust to "
                               "outliers. The loss is calculated by taking the median of all absolute differences "
                               "between the target and the prediction.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error",
                "range": "[0:inf), 0 is the best score"
            },
            "r^2": {
                "description": "R^2 (coefficient of determination) regression score function.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score",
                "range": "(-inf:1], 1 is the best score"
            },
            "mean_poisson_deviance": {
                "description": "Mean Poisson deviance regression loss.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance",
                "range": "[0:inf), 0 is the best score"
            },
            "mean_gamma_deviance": {
                "description": "Mean Gamma deviance regression loss.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance",
                "range": "[0:inf), 0 is the best score"
            },
            "mean_tweedie_deviance": {
                "description": "Mean Tweedie deviance regression loss.",
                "ref": "https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance",
                "range": "[0:inf), 0 is the best score"
            },
        }
        return {"metrics": metrics, "metadata": metadata}

    def scoreXY(self, X, y):
        return self.est_.score(X, y)

    def predict(self, X):
        return self.est_.predict(X)

    def save(self):
        m = save_model(self.est_, pipeline_id=self._pipeline_id)
        if m:
            update_status(
                self._pipeline_id,
                "Completed and model saved",
                "5/5",
                log="Pipeline: {}, Model: {}, Completed and model saved. Step: 5/5".format(self._pipeline_id, m.id)
            )
            return True
        else:
            update_status(
                self._pipeline_id,
                "Completed pipeline, error saving model",
                "-5/5",
                log="Pipeline: {}, Completed pipeline, error saving model. Step: -5/5".format(self._pipeline_id)
            )
            return False


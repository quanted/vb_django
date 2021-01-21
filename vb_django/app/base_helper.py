from vb_django.utilities import update_status, save_model
import datetime
import warnings

warnings.simplefilter('ignore')


class BaseHelper:
    def __init__(self, pipeline_id):
        self._pipeline_id = pipeline_id
        update_status(
            self._pipeline_id,
            "Initializing pipeline",
            "1/5",
            "Pipeline: {}, Initializing pipeline".format(self._pipeline_id)
        )
        pass

    def get_estimator(self):
        return self.est_

    def set_params(self, hyper_parameters):
        update_status(
            self._pipeline_id,
            "Setting hyper-parameters",
            "2/5",
            "Pipeline: {}, Setting hyper-parameters".format(self._pipeline_id)
        )
        self.est_.set_params(hyper_parameters)

    def fit(self, X, y):
        time0 = datetime.datetime.now()
        update_status(
            self._pipeline_id,
            "Fitting pipeline estimator",
            "3/5",
            "Pipeline: {}, Fitting pipeline estimator".format(self._pipeline_id)
        )
        self.n_, self.k_ = X.shape
        self.est_ = self.get_estimator()
        self.est_.fit(X, y)
        runtime = datetime.datetime.now() - time0
        # TODO: Update PipelineInstanceMetadata for +1 totalRuns, avgRuntime, avgRuntime/n
        update_status(
            self._pipeline_id,
            "Completed fitting pipeline estimator",
            "4/5",
            "Pipeline: {}, Completed fitting pipeline estimator".format(self._pipeline_id)
        )
        return self

    def transform(self, X, y=None):
        return self.est_.transform(X, y)

    def score(self, X, y):
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
                "Pipeline: {}, Model: {}, Completed and model saved".format(self._pipeline_id, m.id)
            )
            return True
        else:
            update_status(
                self._pipeline_id,
                "Completed pipeline, error saving model",
                "-5/5",
                "Pipeline: {}, Completed pipeline, error saving model".format(self._pipeline_id)
            )
            return False

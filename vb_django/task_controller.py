import vb_django.dask_django
from dask.distributed import Client, fire_and_forget
from vb_django.models import Project, Dataset, Pipeline, Model
from io import StringIO
from vb_django.app.metadata import Metadata
from vb_django.utilities import update_status, load_dataset, load_model
from vb_django.app.elasticnet import ENet
from vb_django.app.gbr import GBR, HGBR
from vb_django.app.vb_helper import VBHelper
from vb_django.app.flexible_pipeline import FlexiblePipe
from vb_django.app.l1lars import L1Lars
from vb_django.app.regressors import LinRegSupreme
from vb_django.app.svr import RBFSVR, LinSVR
from vb_django.app.vb_plotter import VBPlotter
import pandas as pd
import copy
import os
import json
import socket
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dask_scheduler = os.getenv("DASK_SCHEDULER", "tcp://" + socket.gethostbyname(socket.gethostname()) + ":8786")
pre_processing_steps = 3

pipelines = {
    "vbhelper": VBHelper,
    "enet": ENet,
    "gbr": GBR,
    "hgbr": HGBR,
    "flexpipe": FlexiblePipe,
    "l1lars": L1Lars,
    "lrsup": LinRegSupreme,
    "rbfsvr": RBFSVR,
    "linsvr": LinSVR,
}


class DaskTasks:

    @staticmethod
    def setup_task(project_id, dataset_id, pipeline_id):
        docker = bool(os.getenv("IN_DOCKER", False))

        if docker:
            client = Client(dask_scheduler)
            fire_and_forget(client.submit(DaskTasks.execute_task, int(project_id), int(dataset_id), int(pipeline_id)))
        else:
            DaskTasks.execute_task(int(project_id), int(dataset_id), int(pipeline_id))

    @staticmethod
    def refit_task(project_id, model_id, selected_models: dict):
        docker = bool(os.getenv("IN_DOCKER", False))

        if docker:
            client = Client(dask_scheduler)
            fire_and_forget(client.submit(DaskTasks.set_prediction_estimators, int(project_id), int(model_id), selected_models))
        else:
            DaskTasks.set_prediction_estimators(int(project_id), int(model_id), selected_models)

    @staticmethod
    def execute_task(project_id, dataset_id, pipeline_id):
        # STAGE 1 - Data and parameter load from db
        update_status(
            pipeline_id,
            "Data and Model Setup: Retrieving dataset and pipeline", "1/{}".format(pre_processing_steps),
            log="Pipeline: {}, Type: {}, Setup: 1/{}".format(pipeline_id, None, pre_processing_steps),
            message="Cross validation"
        )
        project = Project.objects.get(id=int(project_id))
        dataset = Dataset.objects.get(id=int(dataset_id))
        pipeline = Pipeline.objects.get(id=int(pipeline_id))

        project.dataset = int(dataset_id)
        project.save()

        df = load_dataset(dataset_id, dataset)
        dataset_metadata = Metadata(parent=dataset).get_metadata("DatasetMetadata")
        pipeline_metadata = Metadata(parent=pipeline).get_metadata("PipelineMetadata")
        project_metadata = Metadata(parent=project).get_metadata("ProjectMetadata")

        target_label = None if "target" not in project_metadata.keys() else project_metadata["target"]
        features_label = None if "features" not in project_metadata.keys() else project_metadata["features"]

        target_label = "target" if ("target" not in dataset_metadata.keys() and target_label is None) else dataset_metadata["target"]

        if "features" not in dataset_metadata.keys() and features_label is None:
            features_label = None
        else:
            features_label = dataset_metadata["features"]
        if features_label is None or features_label == "*":
            features_label = list(df.columns)
            features_label.remove(target_label)
        else:
            features_label = json.loads(features_label)
        drop_vars = [] if "drop_features" not in project_metadata.keys() else json.loads(project_metadata["drop_features"].replace("\'", "\""))
        for d in drop_vars:
            features_label.remove(d)

        # STAGE 2 - Data prep
        update_status(
            pipeline_id,
            "Data and Model Setup: Loading data", "2/{}".format(pre_processing_steps),
            log="Pipeline: {}, Type: {}, Setup: 2/{}".format(pipeline_id, pipeline.name, pre_processing_steps),
            message="Cross validation"
        )

        target = df[target_label].to_frame()
        if features_label:
            features = df[features_label]
        else:
            features = df.drop(target_label, axis=1)

        # STAGE 3 - VBHelper execution
        update_status(
            pipeline_id,
            "Data and Model Setup: Loading all parameters and settings", "3/{}".format(pre_processing_steps),
            log="Pipeline: {}, Type: {}, Setup: 3/{}".format(pipeline_id, pipeline.name, pre_processing_steps),
            message="Cross validation"
        )
        if pipeline_metadata:
            vbhelper_parameters = None if "parameters" not in pipeline_metadata.keys() else json.loads(pipeline_metadata["parameters"].replace("'", "\""))
        else:
            vbhelper_parameters = {}

        vbhelper_parameters["pipeline_id"] = pipeline_id
        outer_cv = pipeline_metadata["outer_cv"] if "outer_cv" in pipeline_metadata.keys() else "True"
        try:
            vbhelper = VBHelper(**vbhelper_parameters)
            if "estimators" in pipeline_metadata.keys():
                est_str = pipeline_metadata["estimators"].replace("\'", "\"")
                estimators = json.loads(est_str)
            else:
                update_status(pipeline_id, "Error: VB Helper requires an estimator.",
                              "-1/{}".format(pre_processing_steps),
                              log="Pipeline: {}, Type: {}, Setup: -1/{}".format(pipeline_id, pipeline.name,
                                                                                pre_processing_steps),
                              message="Cross validation"
                              )
                return
            vbhelper.setData(X_df=features, y_df=target)
            inner_cv_dict = {'cv_reps': 1, 'cv_folds': 5, 'cv_strategy': ('quantile', 5)}
            inner_cv = vbhelper.getCV(cv_dict=inner_cv_dict)
            # prep_dict = {'cat_approach': 'together', 'impute_strategy': 'IterativeImputer', 'cat_idx': vbhelper.cat_idx}
            prep_dict = {'cat_approach': 'together', 'impute_strategy': 'impute_middle', 'cat_idx': vbhelper.cat_idx}
            pipe_kwargs = dict(do_prep=not vbhelper.run_stacked, prep_dict=prep_dict, inner_cv=inner_cv,
                               cat_idx=vbhelper.cat_idx, float_idx=vbhelper.float_idx,
                               bestT=False)
            estimators_dict = {}
            e_i = 0
            for e in estimators:
                name = e["name"] if "name" in e.keys() else e["type"] + "-{}".format(e_i)
                n_i = 1
                n_name = name
                while n_name in estimators_dict.keys():
                    n_name = name + "-{}".format(n_i)
                    n_i += 1
                name = n_name
                estimator = DaskTasks.get_estimator(e["type"])
                e_kwargs = copy.copy(pipe_kwargs)
                for k, p in e["parameters"].items():
                    e_kwargs[k] = p
                estimators_dict[name] = {"pipe": estimator, "pipe_kwargs": e_kwargs}
                e_i += 1
            vbhelper.setPipeDict(estimators_dict)
            vbhelper.setModelDict()
            if outer_cv == "True":
                vbhelper.runCrossValidate(verbose=True)
                vbhelper.buildCVScoreDict()
            else:
                #TODO: check processing for non-outer-cv instance for data cleanup
                vbhelper.fitEstimators()
            try:
                model = Model.objects.get(pipeline=pipeline)
                model_id = model.id
            except Model.DoesNotExist:
                model_id = None
            vbhelper.save(message="Completed.")
            del model
        except Exception as e:
            update_status(pipeline_id, "Error: Unknown error executing pipeline",
                          "-0/16",
                          log="Pipeline: {}, Type: {}, Error: {}".format(pipeline_id, pipeline.name, e),
                          message="Cross validation"
                          )
        del vbhelper

    @staticmethod
    def get_estimator(etype):
        if etype in pipelines.keys():
            return pipelines[etype]
        return None

    @staticmethod
    def evaluate(project_id, model_id, flag=None):
        project = Project.objects.get(id=int(project_id))
        model = Model.objects.get(id=int(model_id))
        m = load_model(model.id, model.model)
        flags = ["CVYhatVsY", "CVYhat", "CVScores", "BWCVScores"]
        results = {}
        if flag in flags:
            scores = m.evaluate()
            plotter = VBPlotter()
            plotter.setData(scores)
            if flag == flags[0]:
                results[flag] = plotter.plotCVYhatVsY(single_plot=True, include_all_cv=True)
            elif flag == flags[1]:
                results[flag] = plotter.plotCVYhat(single_plot=True, include_all_cv=True)
            elif flag == flags[2]:
                results[flag] = plotter.plotCVScores(sort=1)
            elif flag == flags[3]:
                results[flag] = plotter.plotBoxWhiskerCVScores()
        else:
            results = m.evaluate()
            results["plot_flags"] = flags
        return results

    @staticmethod
    def set_prediction_estimators(project_id, model_id, selected_models: dict):
        project = Project.objects.get(id=int(project_id))
        model = Model.objects.get(id=int(model_id))
        m = load_model(model.id, model.model)
        model_metadata = Metadata(parent=model).get_metadata("ModelMetadata")
        m.prediction_model_type = model_metadata["prediction_model_type"] if "prediction_model_type" in model_metadata.keys() else "single"
        m.refitPredictionModels(selected_models=selected_models)
        m.save(n=4, model_id=model_id, message="Model selection")

    @staticmethod
    def predict(project_id, model_id, data: str):
        # TODO: Determine need for checking input data structure (labels/types) against the features (labels/types)
        project = Project.objects.get(id=int(project_id))
        model = Model.objects.get(id=int(model_id))
        m = load_model(model.id, model.model)
        try:
            df = pd.read_csv(StringIO(data))
            results = m.getPredictionValues(df)
        except Exception as e:
            results = f"Error attempt to make prediction with data: {data}, error: {e}"
        return results

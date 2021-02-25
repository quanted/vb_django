import vb_django.dask_django
from dask.distributed import Client, fire_and_forget
from vb_django.models import Project, Dataset, Pipeline, Model
from io import StringIO
from vb_django.app.metadata import Metadata
from vb_django.utilities import update_status, load_dataset, load_model
from vb_django.app.elasticnet import ENet
from vb_django.app.gbr import GBR, HGBR
from vb_django.app.cross_validation import CrossValidatePipeline
from vb_django.app.base_helper import BaseHelper
from dask import delayed
import pandas as pd
import os
import json
import socket
import logging


logger = logging.getLogger("vb_dask")
logger.setLevel(logging.INFO)

dask_scheduler = os.getenv("DASK_SCHEDULER", "tcp://" + socket.gethostbyname(socket.gethostname()) + ":8786")
pre_processing_steps = 3

pipelines = {
    "enet": ENet,
    "gbr": GBR,
    "hgbr": HGBR,
    "cvpipe": CrossValidatePipeline
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
    def execute_task(project_id, dataset_id, pipeline_id):
        # STAGE 1
        update_status(
            pipeline_id,
            "Data and Model Setup: Retrieving dataset and pipeline", "1/{}".format(pre_processing_steps),
            log="Pipeline: {}, Type: {}, Setup: 1/{}".format(pipeline_id, None, pre_processing_steps)
        )
        project = Project.objects.get(id=int(project_id))
        dataset = Dataset.objects.get(id=int(dataset_id))
        pipeline = Pipeline.objects.get(id=int(pipeline_id))

        project.dataset = int(dataset_id)
        project.save()

        df = load_dataset(dataset_id, dataset)
        pipeline_metadata = Metadata(parent=Pipeline.objects.get(id=pipeline_id)).get_metadata("PipelineMetadata")
        project_metadata = Metadata(parent=Project.objects.get(id=project_id)).get_metadata("ProjectMetadata")

        target_label = "response" if "target" not in project_metadata.keys() else project_metadata["target"]
        features_label = None if "features" not in project_metadata.keys() else project_metadata["features"]

        # STAGE 2
        update_status(
            pipeline_id,
            "Data and Model Setup: Loading data", "2/{}".format(pre_processing_steps),
            log="Pipeline: {}, Type: {}, Setup: 2/{}".format(pipeline_id, pipeline.name, pre_processing_steps)
        )

        target = df[target_label]
        if features_label:
            features_list = json.loads(features_label.replace("\'", "\""))
            features = df[features_list]
        else:
            features = df.drop(target_label, axis=1)

        # STAGE 3
        update_status(
            pipeline_id,
            "Data and Model Setup: Loading hyper-parameters", "3/{}".format(pre_processing_steps),
            log="Pipeline: {}, Type: {}, Setup: 3/{}".format(pipeline_id, pipeline.name, pre_processing_steps)
        )
        hyper_parameters = None if "hyper_parameters" not in pipeline_metadata.keys() else json.loads(pipeline_metadata["hyper_parameters"].replace("'", "\""))
        parameters = None if "parameters" not in project_metadata.keys() else project_metadata["parameters"]
        # TODO: parameter will contain non-hyper-parameters for the pipeline, that are specified at the project level.

        try:
            if pipeline.type == "enet":
                enet = ENet(pipeline_id)
                enet.set_params(hyper_parameters)
                enet.fit(features, target)
                enet.save()
            elif pipeline.type == "gbr":
                gbr = GBR(pipeline_id)
                gbr.set_params(hyper_parameters)
                gbr.fit(features, target)
                gbr.save()
            elif pipeline.type == "hgbr":
                hgbr = HGBR(pipeline_id)
                hgbr.set_params(hyper_parameters)
                hgbr.fit(features, target)
                hgbr.save()
            elif pipeline.type == "cvpipe":
                if "estimators" in pipeline_metadata.keys():
                    estimators = json.loads(pipeline_metadata["estimators"].replace("\'", "\""))
                else:
                    update_status(pipeline_id, "Error: Pipeline cvpipe requires an estimator.",
                                  "-1/{}".format(pre_processing_steps),
                                  log="Pipeline: {}, Type: {}, Setup: -1/{}".format(pipeline_id, pipeline.name, pre_processing_steps)
                                  )
                    return
                if len(estimators) == 0:
                    update_status(pipeline_id, "Error: Pipeline cvpipe requires an estimator.",
                                  "-1/{}".format(pre_processing_steps),
                                  log="Pipeline: {}, Type: {}, Setup: -1/{}".format(pipeline_id, pipeline.name, pre_processing_steps)
                                  )
                    return
                estimator_dict = {}
                for e in estimators:
                    if e["type"] != pipeline.type:
                        estimator = DaskTasks.get_estimator(e["type"])
                        if estimator:
                            est = estimator(pipeline_id)
                            est.set_params(e["hyper_parameters"])
                            estimator_dict[e["type"]] = est
                cvpipe = CrossValidatePipeline(pipeline_id)
                cvpipe.set_params(hyper_parameters)
                cvpipe.set_data(features, target)
                cvpipe.set_estimators(estimator_dict)
                fitted = cvpipe.fit()
                if not fitted:
                    update_status(pipeline_id, "Error: Pipeline missing required estimator, model or data.",
                                  "-5/7",
                                  log="Pipeline: {}, Type: {}, Setup: -5/7".format(pipeline_id, pipeline.name)
                                  )
                    return
                cvpipe.run_cross_validate()
                cvpipe.save()
        except Exception as e:
            update_status(pipeline_id, "Error: Unknown error executing pipeline",
                          "-0/7",
                          log="Pipeline: {}, Type: {}, Error: {}".format(pipeline_id, pipeline.name, e)
                          )

    @staticmethod
    def get_estimator(etype):
        if etype in pipelines.keys():
            return pipelines[etype]
        return None

    @staticmethod
    def make_prediction(project_id, model_id, data: str = None):
        project = Project.objects.get(id=int(project_id))
        model = Model.objects.get(id=int(model_id))
        try:
            dataset = Dataset.objects.get(id=int(project.dataset))
        except Dataset.DoesNotExist:
            return {"error": "No dataset found for id: {}".format(project.dataset)}
        if data:
            df = pd.read_csv(StringIO(data))
        else:
            df = load_dataset(dataset.id, dataset)
        project_metadata = Metadata(parent=Project.objects.get(id=project_id)).get_metadata("ProjectMetadata")

        target_label = "response" if "target" not in project_metadata.keys() else project_metadata["target"]
        features_label = None if "features" not in project_metadata.keys() else project_metadata["features"]

        target = df[target_label]
        if features_label:
            features_list = json.loads(features_label.replace("\'", "\""))
            features = df[features_list]
        else:
            features = df.drop(target_label, axis=1)

        m = load_model(model.id, model.model)
        predict = m.predict(features)
        if type(predict) == tuple:
            score = predict[1]
            predict = predict[0]
        else:
            score = BaseHelper.score(target, predict)

        response = {
            "predict": predict,
            "train_score": score
        }
        return response

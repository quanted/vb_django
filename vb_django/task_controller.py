from dask.distributed import Client, fire_and_forget
from vb_django.models import Project, Dataset, Pipeline, Model
from io import StringIO
from vb_django.app.metadata import Metadata
from vb_django.utilities import update_status, load_dataset, load_model
from vb_django.app.elasticnet import ENet
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
    # GradientBoosting
    #
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

        dataset = Dataset.objects.get(id=int(dataset_id))
        pipeline = Pipeline.objects.get(id=int(pipeline_id))

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

        if pipeline.type == "enet":
            enet = ENet(pipeline_id)
            enet.set_params(hyper_parameters)
            enet.fit(features, target)
            enet.save()

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
        score = BaseHelper.score(target, predict)

        response = {
            "results": predict,
            "train_score": score
        }
        return response

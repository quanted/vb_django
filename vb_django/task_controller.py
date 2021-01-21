from dask.distributed import Client, fire_and_forget
from vb_django.models import Project, Dataset, Pipeline, Model
from io import StringIO
from vb_django.app.linear_regression import execute_lra
from vb_django.app.metadata import Metadata
from vb_django.utilities import update_status
from vb_django.app.linear_regression import LinearRegressionAutomatedVB
from dask import delayed
import pickle
import pandas as pd
import os
import json
import socket
import logging
import time


# TODO: REFACTOR

logger = logging.getLogger("vb_dask")
logger.setLevel(logging.INFO)

dask_scheduler = os.getenv("DASK_SCHEDULER", "tcp://" + socket.gethostbyname(socket.gethostname()) + ":8786")
pre_processing_steps = 3


class DaskTasks:

    @staticmethod
    def setup_task(project_id, dataset_id, pipeline_id):

        dataset = Dataset.objects.get(id=int(dataset_id))
        pipeline = Pipeline.objects.get(id=int(pipeline_id))
        # pipeline.save()
        client = Client(dask_scheduler)
        df = pd.read_csv(StringIO(bytes(dataset.data).decode())).drop("ID", axis=1)
        fire_and_forget(client.submit(DaskTasks.execute_task, df, int(project_id), int(pipeline.id), str(pipeline.name), int(dataset_id)))
        #DaskTasks.execute_task(df, int(amodel.id), str(amodel.name), int(dataset_id))

    @staticmethod
    def execute_task(df, project_id, pipeline_id, pipeline_name, dataset_id):
        logger.info("Starting Pipeline Pre-processing -------- Pipeline ID: {}; Pipeline Type: {}; step 1/{}".format(pipeline_id, pipeline_name, pre_processing_steps))
        update_status(pipeline_id, "PP: Loading and validating data", "1/{}".format(pre_processing_steps))

        dataset_m = Metadata(parent=Dataset.objects.get(id=dataset_id)).get_metadata("DatasetMetadata")
        pipeline_m = Metadata(parent=Pipeline.objects.get(id=pipeline_id)).get_metadata("PipelineMetadata")
        project_m = Metadata(parent=Project.objects.get(id=project_id)).get_metadata("ProjectMetadata")
        target = "Response" if "target" not in project_m.keys() else project_m["target"]
        attributes = None if "features" not in project_m.keys() else project_m["features"]
        y = df[target]
        if attributes:
            attributes_list = json.loads(attributes.replace("\'", "\""))
            x = df[attributes_list]
        else:
            x = df.drop(target, axis=1)
        logger.info("PP: Model ID: {}, loading hyper-parameters step 2/{}".format(pipeline_id, pre_processing_steps))
        update_status(pipeline_id, "PP: Loading hyper-parameters", "2/{}".format(pre_processing_steps))

        logger.info("PP: Model ID: {}, setup complete step 3/{}".format(pipeline_id, pre_processing_steps))
        update_status(pipeline_id, "PP: setup complete", "3/{}".format(pre_processing_steps))

        if pipeline_name == "lra":
            execute_lra(pipeline_id, pipeline_m, x, y)

    @staticmethod
    def make_prediction(project_id, amodel_id, data=None):
        amodel = Model.objects.get(id=int(amodel_id))
        dataset = Dataset.objects.get(id=int(amodel.dataset))
        y_data = None

        df = pd.read_csv(StringIO(bytes(dataset.data).decode()))
        project_m = Metadata(parent=Project.objects.get(id=project_id)).get_metadata("ProjectMetadata")
        target = "Response" if "target" not in project_m.keys() else project_m["target"]
        attributes = None if "features" not in project_m.keys() else project_m["features"]

        y = df[target]
        if attributes:
            attributes_list = json.loads(attributes.replace("\'", "\""))
            x = df[attributes_list]
        else:
            x = df.drop(target, axis=1)

        t = LinearRegressionAutomatedVB()
        t.set_data(x, y)
        x_train = t.x_train
        y_train = t.y_train
        x_data = t.x_test
        y_test = t.y_test.to_numpy().flatten()

        if data is not None:
            x_data = data
        model = pickle.loads(amodel.model)
        response = {
            "results": model.predict(x_data),
            "train_score": model.score(x_train, y_train)
        }
        if data is None:
            response["residuals"] = y_test - response["results"]
            response["test_score"] = model.score(x_data, y_test)
        return response

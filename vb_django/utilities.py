from vb_django.models import Pipeline, Dataset, Model
from vb_django.app.metadata import Metadata
import json
import logging
import zlib
import pandas as pd
from io import StringIO
import pickle


logger = logging.getLogger("vb_dask")
logger.setLevel(logging.INFO)


def update_status(_id, status, stage, message=None, retry=5, log=None):
    if retry == 0:
        pass
    meta = 'PipelineMetadata'
    try:
        amodel = Pipeline.objects.get(id=int(_id))
        m = Metadata(parent=amodel, metadata=json.dumps({"status": status, "stage": stage, "message": message}))
        m.set_metadata(meta)
        if log:
            logger.info(log)
    except Exception as ex:
        logger.warning("Error attempting to save metadata update: {}".format(ex))
        update_status(_id, status, stage, None, retry - 1)


def save_dataset(data: str, dataset_id=None):
    e_str = data.encode()
    ce_str = zlib.compress(e_str)
    if dataset_id:
        try:
            dataset = Dataset.objects.get(id=int(dataset_id))
        except Dataset.DoesNotExist:
            return None
        dataset.data = ce_str
        dataset.save()
    return ce_str


def load_dataset(dataset_id, dataset=None):
    if not dataset:
        try:
            dataset = Dataset.objects.get(id=int(dataset_id))
        except Dataset.DoesNotExist:
            return None
    ce_df = zlib.decompress(dataset.data)
    df = pd.read_csv(StringIO(bytes(ce_df).decode()))
    return df


def save_model(model: bytes, model_id=None, pipeline_id=None):
    comp_model = zlib.compress(pickle.dumps(model))
    m = None
    if not model_id:
        try:
            o_model = Model.objects.get(id=int(model_id))
        except Model.DoesNotExist:
            return None
        o_model.model = comp_model
        o_model.save()
        m = o_model
    if pipeline_id and not model_id:
        pipeline = Pipeline.objects.get(id=int(pipeline_id))
        existing_m = Model.objects.filter(pipeline=pipeline)
        l = 1 if existing_m is None else len(existing_m) + 1
        name = "{}-{}".format(pipeline.type, l)
        m = Model(pipeline, name, "", comp_model)
    return m


def load_model(model_id, model=None):
    if not model:
        try:
            model = Model.objects.get(id=int(model_id))
        except Model.DoesNotExist:
            return None
    comp_model = zlib.decompress(model.data)
    r_model = pickle.loads(comp_model)
    return r_model

from vb_django.models import Pipeline, Dataset, Model, PipelineInstance, PipelineInstanceMetadata, PipelineLog
from vb_django.app.metadata import Metadata
import json
import logging
import zlib
import datetime
import pandas as pd
from io import StringIO
import pickle
import sys


logger = logging.getLogger("vb_dask")
logger.setLevel(logging.INFO)


def update_status(_id, status, stage, message=None, retry=5, log=None):
    if _id == -1:
        return
    if retry == 0:
        return
    meta = 'PipelineMetadata'
    try:
        amodel = Pipeline.objects.get(id=int(_id))
        m = Metadata(parent=amodel, metadata=json.dumps({"status": status, "stage": stage, "message": message}))
        m.set_metadata(meta)
        if log:
            logger.info(log)
        PipelineLog(parent=amodel, logtype=status, log=f"Stage: {stage}, Message: {message}", timestamp=str(datetime.datetime.now().timestamp()))
    except Exception as ex:
        logger.warning("Error attempting to save status update: {}".format(ex))
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
        logger.info("Dataset ID: {} saved; raw {} bytes; compressed {} bytes".format(dataset_id, sys.getsizeof(e_str), sys.getsizeof(ce_str)))
    return ce_str


def load_dataset(dataset_id, dataset=None, replace_nan=False):
    if not dataset:
        try:
            dataset = Dataset.objects.get(id=int(dataset_id))
        except Dataset.DoesNotExist:
            return None
    ce_df = zlib.decompress(dataset.data)
    df = pd.read_csv(StringIO(bytes(ce_df).decode()))
    if replace_nan:
        df = df.fillna('')
    return df


def save_model(model, model_id=None, pipeline_id=None, replace=True):
    try:
        raw_model = pickle.dumps(model)
    except Exception as e:
        logger.warning("Pipeline ID: {}, Model ID: {}, error attempting to pickle model: {}".format(pipeline_id, model_id, e))
        return None
    comp_model = zlib.compress(raw_model)
    m = None
    if model_id:
        try:
            o_model = Model.objects.get(id=int(model_id))
        except Model.DoesNotExist:
            logger.warning(f"ERROR - Model: {model_id} does not exist")
            return None
        o_model.model = comp_model
        o_model.save()
        m = o_model
        logger.info(f"Model: {model_id} successfully updated")
    if pipeline_id and not model_id:
        pipeline = Pipeline.objects.get(id=int(pipeline_id))
        existing_m = Model.objects.filter(pipeline=pipeline)
        if replace:
            for m in existing_m:
                m.delete()
        l = 1 if existing_m is None else len(existing_m) + 1
        name = "{}-{}".format(pipeline.type, l+1)
        m = Model(pipeline=pipeline, name=name, description="", model=comp_model)
        m.save()
        logger.info(f"Model: {m.id} successfully created.")
    if m:
        base_size = (sys.getsizeof(raw_model)/1024.0**2)
        comp_size = (sys.getsizeof(comp_model)/1024.0**2)
        logger.info("Model ID: {} saved; raw {} MB; compressed {} MB".format(m.id, base_size, comp_size))
    del raw_model
    del comp_model
    return m


def load_model(model_id, model=None):
    if not model:
        try:
            model = Model.objects.get(id=int(model_id)).model
        except Model.DoesNotExist:
            return None
    comp_model = zlib.decompress(model)
    r_model = pickle.loads(comp_model)
    return r_model


def update_pipeline_metadata(pipeline, runtime, n):
    for p in pipeline.metrics:
        try:
            pipelineMetata = PipelineInstanceMetadata.objects.get(parent=PipelineInstance.objects.get(ptype=pipeline.ptype), name=p)
            value = float(pipelineMetata.value)
            if p == "total_runs":
                value = str(int(pipelineMetata.value) + 1)
            elif p == "avg_runtime":
                if value != 0.0:
                    value = str((float(pipelineMetata.value) + runtime)/2.0)
                else:
                    value = runtime
            elif p == "avg_runtime/n":
                if value != 0.0 and n != 0:
                    value = str((float(pipelineMetata.value) + (runtime/float(n))/2.0))
                else:
                    value = (runtime / float(n))
            pipelineMetata.value = value
            pipelineMetata.save()
        except PipelineInstanceMetadata.DoesNotExist:
            continue


def load_request(request):
    data = None
    if "multipart/form-data" in request.content_type:
        data = request.POST.dict()
    elif "application/json" in request.content_type:
        data = request.data

    return data

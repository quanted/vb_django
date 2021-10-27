from vb_django.models import PipelineInstance, PipelineInstanceParameters, PipelineInstanceMetadata
from vb_django.task_controller import pipelines
import json
import logging

logger = logging.getLogger("vb_django")
logger.setLevel(logging.INFO)


def load_pipelines(purge: bool = False):
    pl = pipelines
    if purge and not PipelineInstance.DoesNotExist:
        logger.info("Purging existing pipeline-instance data")
        PipelineInstance.objects.all().delete()
    logger.info("Setting up and loading pipeline instance hyper-parameters and metadata")
    print("Setting up and loading pipeline instance hyper-parameters and metadata")
    for name, p in pl.items():
        try:
            pipeline = PipelineInstance.objects.filter(ptype=name)
            if len(pipeline) > 1:
                for p in pipeline:
                    p.delete()
                pipeline = PipelineInstance(name=p.name, ptype=p.ptype, description=p.description, active=True)
            elif len(pipeline) == 0:
                pipeline = PipelineInstance(name=p.name, ptype=p.ptype, description=p.description, active=True)
            else:
                pipeline = pipeline[0]
                pipeline.description = p.description
                pipeline.name = p.name
                pipeline.active = True
        except Exception as e:
            print("Stopped loading pipelines due to db connection issue")
            logger.info("Stopped loading pipelines due to db connection issue")
            return
        pipeline.save()
        for h, v in p.hyper_parameters.items():
            if type(v["value"]) is dict:
                options = json.dumps(v["options"])
            else:
                options = str(v["options"])
            try:
                pipelineParameter = PipelineInstanceParameters.objects.get(pipeline=pipeline, name=h)
                pipelineParameter.name = h
                pipelineParameter.vtype = v["type"]
                pipelineParameter.value = str(v["value"])
                pipelineParameter.options = options
            except PipelineInstanceParameters.DoesNotExist:
                pipelineParameter = PipelineInstanceParameters(
                    pipeline=pipeline, name=h, vtype=v["type"], value=str(v["value"]), options=options
                )
            pipelineParameter.save()
        for m in p.metrics:
            try:
                pipelineMeta = PipelineInstanceMetadata.objects.get(parent=pipeline, name=m)
                pipelineMeta.save()
            except PipelineInstanceMetadata.DoesNotExist:
                pipelineMeta = PipelineInstanceMetadata(parent=pipeline, name=m, value="0")
                pipelineMeta.save()
        logger.info("Successfully added pipeline: {} to the DB".format(name))
        print("Successfully added pipeline: {} to the DB".format(name))
    logger.info("Completed setting up pipeline instances in DB")
    print("Completed setting up pipeline instances in DB")

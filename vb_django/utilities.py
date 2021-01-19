from vb_django.models import Experiment
from vb_django.app.metadata import Metadata
import json
import logging

logger = logging.getLogger("vb_dask")
logger.setLevel(logging.INFO)


def update_status(_id, status, stage, message=None, retry=5):
    if retry == 0:
        pass
    meta = 'ExperimentMetadata'
    try:
        amodel = Experiment.objects.get(id=int(_id))
        m = Metadata(parent=amodel, metadata=json.dumps({"status": status, "stage": stage, "message": message}))
        m.set_metadata(meta)
    except Exception as ex:
        logger.warning("Error attempting to save metadata update: {}".format(ex))
        update_status(_id, status, stage, None, retry - 1)

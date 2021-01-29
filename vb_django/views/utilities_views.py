from django.http import HttpResponse, JsonResponse
from vb_django.models import PipelineInstance, PipelineInstanceMetadata, PipelineInstanceParameters


def pipeline_details(request):
    """
    Returns the details for each of the implemented pipelines and their corresponding hyper-parameters
    """
    details = []
    for p in PipelineInstance.objects.all():
        pipeline = {}
        if p.active == 0:
            continue
        pipeline["name"] = p.name
        pipeline["ptype"] = p.ptype
        pipeline["description"] = p.description
        pipeline["hyper-parameters"] = []
        params = PipelineInstanceParameters.objects.filter(pipeline=p)
        for hp in params:
            hp_details = {
                "name": hp.name,
                "vtype": hp.vtype,
                "value": hp.value,
                "options": hp.options
            }
            pipeline["hyper-parameters"].append(hp_details)
        pipeline["metadata"] = {}
        pmeta = PipelineInstanceMetadata.objects.filter(parent=p)
        for md in pmeta:
            pipeline["metadata"][md.name] = md.value
        details.append(pipeline)
    return JsonResponse(details, safe=False)

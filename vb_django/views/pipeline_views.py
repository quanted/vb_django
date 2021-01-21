from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from vb_django.models import Pipeline, Project
from vb_django.app.metadata import Metadata
from vb_django.serializers import PipelineSerializer
from vb_django.permissions import IsOwnerOfPipeline, IsOwnerOfProject


class PipelineView(viewsets.ViewSet):
    """
    The Pipeline API endpoint viewset for managing user pipeline in the database.
    """
    serializer_class = PipelineSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsOwnerOfPipeline]

    def list(self, request):
        """
        GET request that lists all the pipeline for a specific project id
        :param request: GET request, containing the project id as 'project'
        :return: List of analytical models
        """
        if 'project' in self.request.query_params.keys():
            pipeline = Pipeline.objects.filter(project=int(self.request.query_params.get('project')))
            serializer = self.serializer_class(pipeline, many=True)
            response_data = serializer.data
            for l in response_data:
                a = Pipeline.objects.get(pk=int(l["id"]))
                m = Metadata(a, None)
                l["metadata"] = m.get_metadata("PipelineMetadata")
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(
            "Required 'project' parameter for the pipeline was not found.",
            status=status.HTTP_400_BAD_REQUEST
        )

    def create(self, request):
        """
        POST request that creates a new Pipeline.
        :param request: POST request
        :return: New pipeline object
        """
        pipeline_inputs = request.data.dict()
        serializer = self.serializer_class(data=pipeline_inputs, context={'request': request})
        try:
            project = Project.objects.get(id=int(pipeline_inputs["project"]))
        except Project.DoesNotExist:
            return Response("No project found for id: {}".format(int(pipeline_inputs["project"])), status=status.HTTP_400_BAD_REQUEST)
        if project.owner != request.user:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        if serializer.is_valid():
            serializer.save()
            pipeline = serializer.data
            if "metadata" not in pipeline_inputs.keys():
                pipeline_inputs["metadata"] = None
            a = Pipeline.objects.get(pk=int(pipeline["id"]))
            m = Metadata(a, pipeline_inputs["metadata"])
            meta = m.set_metadata("PipelineMetadata")
            pipeline["metadata"] = meta
            if pipeline:
                return Response(pipeline, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        pipeline_inputs = request.data.dict()
        serializer = self.serializer_class(data=request.data.dict(), context={'request': request})

        try:
            project = Project.objects.get(id=int(pipeline_inputs["project"]))
        except Project.DoesNotExist:
            return Response("No project found for id: {}".format(int(pipeline_inputs["project"])), status=status.HTTP_400_BAD_REQUEST)
        if project.owner != request.user:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        if serializer.is_valid() and pk is not None:
            try:
                original_pipeline = Pipeline.objects.get(id=int(pk))
            except Pipeline.DoesNotExist:
                return Response(
                    "No pipeline found for id: {}".format(pk),
                    status=status.HTTP_400_BAD_REQUEST
                )
            if IsOwnerOfPipeline().has_object_permission(request, self, original_pipeline):
                experiment = serializer.update(original_pipeline, serializer.validated_data)
                if experiment:
                    response_status = status.HTTP_201_CREATED
                    response_data = serializer.data
                    response_data["id"] = experiment.id
                    if int(pk) == experiment.id:
                        response_status = status.HTTP_200_OK
                    if "metadata" not in pipeline_inputs.keys():
                        pipeline_inputs["metadata"] = None
                    a = Pipeline.objects.get(pk=experiment.id)
                    m = Metadata(a, pipeline_inputs["metadata"])
                    response_data["metadata"] = m.set_metadata("PipelineMetadata")
                    return Response(response_data, status=response_status)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        if pk is not None:
            try:
                amodel = Pipeline.objects.get(id=int(pk))
            except Pipeline.DoesNotExist:
                return Response("No pipeline found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if IsOwnerOfPipeline().has_object_permission(request, self, amodel):
                amodel.delete()
                return Response(status=status.HTTP_200_OK)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response("No pipeline 'id' in request.", status=status.HTTP_400_BAD_REQUEST)

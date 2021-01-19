from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from vb_django.models import Experiment, Project
from vb_django.app.metadata import Metadata
from vb_django.serializers import ExperimentSerializer
from vb_django.permissions import IsOwnerOfExperiment, IsOwnerOfProject


class ExperimentView(viewsets.ViewSet):
    """
    The Experiment API endpoint viewset for managing user experiment in the database.
    """
    serializer_class = ExperimentSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsOwnerOfExperiment]

    def list(self, request):
        """
        GET request that lists all the experiments for a specific project id
        :param request: GET request, containing the project id as 'project'
        :return: List of analytical models
        """
        if 'project' in self.request.query_params.keys():
            experiment = Experiment.objects.filter(project=int(self.request.query_params.get('project')))
            serializer = self.serializer_class(experiment, many=True)
            response_data = serializer.data
            for l in response_data:
                a = Experiment.objects.get(pk=int(l["id"]))
                m = Metadata(a, None)
                l["metadata"] = m.get_metadata("ExperimentMetadata")
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(
            "Required 'project' parameter for the experiment was not found.",
            status=status.HTTP_400_BAD_REQUEST
        )

    def create(self, request):
        """
        POST request that creates a new Experiment.
        :param request: POST request
        :return: New analytical object
        """
        experiment_inputs = request.data.dict()
        serializer = self.serializer_class(data=experiment_inputs, context={'request': request})
        try:
            project = Project.objects.get(id=int(experiment_inputs["project"]))
        except Project.DoesNotExist:
            return Response("No project found for id: {}".format(int(experiment_inputs["project"])), status=status.HTTP_400_BAD_REQUEST)
        if project.owner != request.user:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        if serializer.is_valid():
            serializer.save()
            experiment = serializer.data
            if "metadata" not in experiment_inputs.keys():
                experiment_inputs["metadata"] = None
            a = Experiment.objects.get(pk=int(experiment["id"]))
            m = Metadata(a, experiment_inputs["metadata"])
            meta = m.set_metadata("ExperimentMetadata")
            experiment["metadata"] = meta
            if experiment:
                return Response(experiment, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        experiment_inputs = request.data.dict()
        serializer = self.serializer_class(data=request.data.dict(), context={'request': request})

        try:
            project = Project.objects.get(id=int(experiment_inputs["project"]))
        except Project.DoesNotExist:
            return Response("No project found for id: {}".format(int(experiment_inputs["project"])), status=status.HTTP_400_BAD_REQUEST)
        if project.owner != request.user:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        if serializer.is_valid() and pk is not None:
            try:
                original_experiment = Experiment.objects.get(id=int(pk))
            except Experiment.DoesNotExist:
                return Response(
                    "No experiment found for id: {}".format(pk),
                    status=status.HTTP_400_BAD_REQUEST
                )
            if IsOwnerOfExperiment().has_object_permission(request, self, original_experiment):
                experiment = serializer.update(original_experiment, serializer.validated_data)
                if experiment:
                    response_status = status.HTTP_201_CREATED
                    response_data = serializer.data
                    response_data["id"] = experiment.id
                    if int(pk) == experiment.id:
                        response_status = status.HTTP_200_OK
                    if "metadata" not in experiment_inputs.keys():
                        experiment_inputs["metadata"] = None
                    a = Experiment.objects.get(pk=experiment.id)
                    m = Metadata(a, experiment_inputs["metadata"])
                    response_data["metadata"] = m.set_metadata("ExperimentMetadata")
                    return Response(response_data, status=response_status)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        if pk is not None:
            try:
                amodel = Experiment.objects.get(id=int(pk))
            except Experiment.DoesNotExist:
                return Response("No experiment found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if IsOwnerOfExperiment().has_object_permission(request, self, amodel):
                amodel.delete()
                return Response(status=status.HTTP_200_OK)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response("No experiment 'id' in request.", status=status.HTTP_400_BAD_REQUEST)

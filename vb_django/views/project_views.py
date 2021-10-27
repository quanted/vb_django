from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from vb_django.authentication import ExpiringTokenAuthentication as TokenAuthentication
from vb_django.models import Project
from vb_django.serializers import ProjectSerializer
from vb_django.permissions import IsOwner
from vb_django.app.metadata import Metadata
from vb_django.utilities import load_request
from drf_yasg.utils import swagger_auto_schema


class ProjectView(viewsets.ViewSet):
    """
    The Project API endpoint viewset for managing user projects in the database.
    """
    serializer_class = ProjectSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsOwner]

    def list(self, request, pk=None):
        """
        GET request that lists all the projects
        :param request: GET request
        :return: List of projects
        """
        projects = Project.objects.filter(owner=request.user)
        # TODO: Add ACL access objects
        serializer = self.serializer_class(projects, many=True)
        response_data = serializer.data
        for d in response_data:
            p = Project.objects.get(id=d["id"])
            m = Metadata(p, None)
            meta = m.get_metadata("ProjectMetadata")
            d["metadata"] = meta
        return Response(serializer.data, status=status.HTTP_200_OK)

    @swagger_auto_schema(request_body=ProjectSerializer)
    def create(self, request):
        """
        POST request that creates a new project.
        :param request: POST request
        :return: New project object
        """
        dataset_inputs = load_request(request)
        serializer = self.serializer_class(data=dataset_inputs, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            project = serializer.data
            p = Project.objects.get(id=project["id"])
            if "metadata" not in dataset_inputs.keys():
                dataset_inputs["metadata"] = None
            m = Metadata(p, dataset_inputs["metadata"])
            meta = m.set_metadata("ProjectMetadata")
            project["metadata"] = meta
            return Response(project, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(request_body=ProjectSerializer)
    def update(self, request, pk=None):
        """
        PUT request for updating a project.
        :param request: PUT request
        :return: The updated/200
        """
        dataset_inputs = load_request(request)
        serializer = self.serializer_class(data=dataset_inputs, context={'request': request})
        if serializer.is_valid() and pk is not None:
            try:
                project = Project.objects.get(id=int(pk))
            except Project.DoesNotExist:
                return Response("No project found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if IsOwner().has_object_permission(request, self, project):
                project = serializer.update(project, serializer.validated_data)
                if "metadata" not in dataset_inputs.keys():
                    dataset_inputs["metadata"] = None
                m = Metadata(project, dataset_inputs["metadata"])
                meta = m.set_metadata("ProjectMetadata")
                response_data = serializer.data
                if meta:
                    response_data["metadata"] = meta
                request_status = status.HTTP_200_OK
                return Response(response_data, status=request_status)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        """
        DELETE request for removing the project, cascading deletion for elements in database.
        Front end should require verification of action.
        :param request: DELETE request
        :return:
        """
        if pk is not None:
            try:
                project = Project.objects.get(id=int(pk))
            except Project.DoesNotExist:
                return Response("No project found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if IsOwner().has_object_permission(request, self, project):
                project.delete()
                return Response(status=status.HTTP_200_OK)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response("No project 'id' in request.", status=status.HTTP_400_BAD_REQUEST)

from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from vb_django.authentication import ExpiringTokenAuthentication as TokenAuthentication
from vb_django.models import Dataset, Project
from vb_django.serializers import DatasetSerializer
from vb_django.permissions import IsOwner
from vb_django.app.metadata import Metadata
from vb_django.app.statistics import DatasetStatistics
from vb_django.utilities import load_dataset, load_request
from vb_django.data_exploration import DataExploration
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema


class DatasetView(viewsets.ViewSet):
    """
    The Dataset API endpoint viewset for managing user datasets in the database.
    """
    serializer_class = DatasetSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsOwner]

    dataset_id_p = openapi.Parameter('dataset_id', openapi.IN_QUERY, description="Dataset ID (required)", type=openapi.TYPE_STRING)
    project_id_p = openapi.Parameter('project_id', openapi.IN_QUERY, description="Project ID (required)", type=openapi.TYPE_STRING)
    num_cols_p = openapi.Parameter('num_cols', openapi.IN_QUERY, description="Number of components to get from dataset (optional: default=3)", type=openapi.TYPE_INTEGER)
    keep_cats_p = openapi.Parameter('keep_cats', openapi.IN_QUERY, description="Keep categorical variable columns (optional: default=False)", type=openapi.TYPE_BOOLEAN)
    linkage_p = openapi.Parameter('linkage', openapi.IN_QUERY, description="Linkage method for clustering (optional: default='ward')", type=openapi.TYPE_STRING)
    dist_p = openapi.Parameter('dist', openapi.IN_QUERY, description="Distance heuristic for cluster (optional: default='spearmanr')", type=openapi.TYPE_STRING)

    def list(self, request, pk=None):
        """
        GET request that lists all the Datasets for the user, not containing the data.
        :param request: GET request
        :return: List of datasets
        """
        datasets = Dataset.objects.filter(owner_id=request.user)
        # TODO: Add ACL access objects
        serializer = self.serializer_class(datasets, many=True)
        for d in serializer.data:
            del d["data"]
        return Response(serializer.data, status=status.HTTP_200_OK)

    def retrieve(self, request, pk=None):
        """
        GET request for the data of a dataset, specified by dataset id
        :param request: GET request, containing the dataset id
        :param pk: Dataset id
        :return: Dataset data and relevant statistics
        """
        if pk:
            try:
                dataset = Dataset.objects.get(pk=pk)
            except Dataset.DoesNotExist:
                return Response("No dataset found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, dataset):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            serializer = self.serializer_class(dataset, many=False)
            response_data = serializer.data
            m = Metadata(dataset)
            meta = m.get_metadata("DatasetMetadata")
            response = "Response"
            if meta:
                response_data["metadata"] = meta
                response = meta["target"]
            response_data["data"] = load_dataset(pk)
            if response not in response_data["data"]:
                response = response_data["data"].columns.tolist()[0]
            response_data["statistics"] = DatasetStatistics(response_data["data"]).calculate_statistics(response)
            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(
                "Required id for the dataset was not found.",
                status=status.HTTP_400_BAD_REQUEST
            )

    @swagger_auto_schema(request_body=DatasetSerializer)
    def create(self, request):
        """
        POST request that creates a new Dataset.
        :param request: POST request.
        :return: New dataset
        """
        dataset_inputs = load_request(request)
        serializer = self.serializer_class(data=dataset_inputs, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            dataset = serializer.data
            if dataset:
                d = Dataset.objects.get(id=dataset["id"])
                if "metadata" not in dataset_inputs.keys():
                    dataset_inputs["metadata"] = None
                m = Metadata(d, dataset_inputs["metadata"])
                meta = m.set_metadata("DatasetMetadata")
                response = "Response"
                if meta:
                    dataset["metadata"] = meta
                    response = meta["target"]
                data = load_dataset(d.id)
                if response not in data:
                    response = data.columns.tolist()[0]
                dataset["statistics"] = DatasetStatistics(data).calculate_statistics(response)
                del dataset["data"]
                return Response(dataset, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(request_body=DatasetSerializer)
    def update(self, request, pk=None):
        """
        PUT request to update a dataset
        :param request: PUT request
        :param pk: dataset ID to be updated
        :return: 200/details of updated dataset, 400/bad request, or 401/unauthorized
        """
        dataset_inputs = load_request(request)
        serializer = self.serializer_class(data=dataset_inputs, context={'request': request})
        if serializer.is_valid() and pk is not None:
            try:
                original_dataset = Dataset.objects.get(id=int(pk))
            except Dataset.DoesNotExist:
                return Response(
                    "No dataset found for id: {}".format(pk),
                    status=status.HTTP_400_BAD_REQUEST
                )
            if IsOwner().has_object_permission(request, self, original_dataset):
                amodel = serializer.update(original_dataset, serializer.validated_data)
                m = Metadata(amodel, dataset_inputs["metadata"])
                meta = m.set_metadata("DatasetMetadata")
                if amodel:
                    response_status = status.HTTP_201_CREATED
                    response_data = serializer.data
                    response_data["id"] = amodel.id
                    del response_data["data"]
                    if meta:
                        response_data["metadata"] = meta
                    if int(pk) == amodel.id:
                        response_status = status.HTTP_200_OK
                    return Response(response_data, status=response_status)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        """
        DEL to delete an existing dataset specified by dataset ID
        :param request: DEL request
        :param pk: dataset ID to be deleted
        :return: 200/success, 400/bad request, or 401/unauthorized
        """
        if pk is not None:
            try:
                dataset = Dataset.objects.get(id=int(pk))
            except Dataset.DoesNotExist:
                return Response("No dataset found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if IsOwner().has_object_permission(request, self, dataset):
                m = Metadata(dataset)
                m.delete_metadata("DatasetMetadata")
                dataset.delete()
                return Response(status=status.HTTP_200_OK)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response("No dataset 'id' in request.", status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(manual_parameters=[project_id_p, dataset_id_p])
    @action(detail=False, methods=["GET"], name="Get information about missing data in the dataset")
    def get_missing_vals(self, request):
        """
        GET request to process and retrieve the missing values of a specified dataset. Requires association with a project and a project_id.
        :param request: GET query containing project_id and the dataset_id.
        :return: 401 for unauthorized requests, 200 and error message for an invalid request, 200 with data if valid
        """
        input_data = self.request.query_params
        required_parameters = ["project_id", "dataset_id"]
        if set(required_parameters).issubset(input_data.keys()):
            dataset_id = input_data["dataset_id"]
            project_id = input_data["project_id"]
            try:
                dataset = Dataset.objects.get(pk=dataset_id)
            except Dataset.DoesNotExist:
                return Response("No dataset found for id: {}".format(dataset_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, dataset):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            try:
                project = Project.objects.get(pk=project_id)
            except Project.DoesNotExist:
                return Response("No dataset found for id: {}".format(project_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, project):
                return Response(status=status.HTTP_401_UNAUTHORIZED)

            de = DataExploration(dataset_id=dataset_id, project_id=project_id)
            data = de.get_missing_vals()
            response_status = status.HTTP_200_OK
            return Response(data, status=response_status)
        data = "Missing required parameters: {}".format(", ".join(required_parameters))
        response_status = status.HTTP_200_OK
        return Response(data, status=response_status)

    @swagger_auto_schema(manual_parameters=[project_id_p, dataset_id_p, num_cols_p, keep_cats_p])
    @action(detail=False, methods=["GET"], name="Get component data in the dataset")
    def get_components(self, request):
        """
        GET request to process and retrieve the component data of a specified dataset.
        :param request: GET query containing project_id and the dataset_id.
        Optional num_cols and keep_cats for configuring the components.
        :return: 401 for unauthorized requests, 200 and error message for an invalid request, 200 with data if valid
        """
        input_data = self.request.query_params
        required_parameters = ["project_id", "dataset_id"]
        if set(required_parameters).issubset(input_data.keys()):
            dataset_id = input_data["dataset_id"]
            project_id = input_data["project_id"]
            try:
                dataset = Dataset.objects.get(pk=dataset_id)
            except Dataset.DoesNotExist:
                return Response("No dataset found for id: {}".format(dataset_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, dataset):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            try:
                project = Project.objects.get(pk=project_id)
            except Project.DoesNotExist:
                return Response("No dataset found for id: {}".format(project_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, project):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            de_args = {}
            if "num_cols" in input_data.keys():
                de_args["num_cols"] = input_data["num_cols"]
            if "keep_cats" in input_data.keys():
                de_args["keep_cats"] = "False" != input_data["keep_cats"]
            de = DataExploration(dataset_id=dataset_id, project_id=project_id)
            data = de.get_components(**de_args)
            response_status = status.HTTP_200_OK
            return Response(data, status=response_status)
        data = "Missing required parameters: {}".format(", ".join(required_parameters))
        response_status = status.HTTP_200_OK
        return Response(data, status=response_status)

    @swagger_auto_schema(manual_parameters=[project_id_p, dataset_id_p])
    @action(detail=False, methods=["GET"], name="Get kernel density data in the dataset")
    def get_kernel_densities(self, request):
        """
        GET request for processing and retrieving the gaussian densities of the specified dataset.
        :param request: GET query containing project_id and the dataset_id.
        :return: 401 for unauthorized requests, 200 and error message for an invalid request, 200 with data if valid
        """
        input_data = self.request.query_params
        required_parameters = ["project_id", "dataset_id"]
        if set(required_parameters).issubset(input_data.keys()):
            dataset_id = input_data["dataset_id"]
            project_id = input_data["project_id"]
            try:
                dataset = Dataset.objects.get(pk=dataset_id)
            except Dataset.DoesNotExist:
                return Response("No dataset found for id: {}".format(dataset_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, dataset):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            try:
                project = Project.objects.get(pk=project_id)
            except Project.DoesNotExist:
                return Response("No dataset found for id: {}".format(project_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, project):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            de = DataExploration(dataset_id=dataset_id, project_id=project_id)
            data = de.get_kerneldensity()
            response_status = status.HTTP_200_OK
            return Response(data, status=response_status)
        data = "Missing required parameters: {}".format(", ".join(required_parameters))
        response_status = status.HTTP_200_OK
        return Response(data, status=response_status)

    @swagger_auto_schema(manual_parameters=[project_id_p, dataset_id_p, linkage_p, dist_p])
    @action(detail=False, methods=["GET"], name="Get dendrogram data in the dataset")
    def get_dendrogram(self, request):
        """
        GET request for processing and retrieving the dendrogram data of the specified dataset.
        :param request: GET query containing project_id and the dataset_id.
        :return: 401 for unauthorized requests, 200 and error message for an invalid request, 200 with data if valid
        """
        input_data = self.request.query_params
        required_parameters = ["project_id", "dataset_id"]
        if set(required_parameters).issubset(input_data.keys()):
            dataset_id = input_data["dataset_id"]
            project_id = input_data["project_id"]
            try:
                dataset = Dataset.objects.get(pk=dataset_id)
            except Dataset.DoesNotExist:
                return Response("No dataset found for id: {}".format(dataset_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, dataset):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            try:
                project = Project.objects.get(pk=project_id)
            except Project.DoesNotExist:
                return Response("No dataset found for id: {}".format(project_id), status=status.HTTP_400_BAD_REQUEST)
            if not IsOwner().has_object_permission(request, self, project):
                return Response(status=status.HTTP_401_UNAUTHORIZED)
            dargs = {}
            if "linkage" in input_data.keys():
                dargs["linkage"] = input_data["linkage"]
            if "dist" in input_data.keys():
                dargs["dist"] = input_data["dist"]
            de = DataExploration(dataset_id=dataset_id, project_id=project_id)
            data = de.get_dendrogram(**dargs)
            response_status = status.HTTP_200_OK
            return Response(data, status=response_status)
        data = "Missing required parameters: {}".format(", ".join(required_parameters))
        response_status = status.HTTP_200_OK
        return Response(data, status=response_status)

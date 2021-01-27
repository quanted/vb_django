from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from vb_django.models import Dataset
from vb_django.serializers import DatasetSerializer
from vb_django.permissions import IsOwner
from vb_django.app.metadata import Metadata
from vb_django.app.statistics import DatasetStatistics
from io import StringIO
import pandas as pd
from vb_django.utilities import load_dataset, load_request


class DatasetView(viewsets.ViewSet):
    """
    The Dataset API endpoint viewset for managing user datasets in the database.
    """
    serializer_class = DatasetSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsOwner]

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
                response = meta["response"]
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

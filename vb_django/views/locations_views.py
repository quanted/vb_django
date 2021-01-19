from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from vb_django.models import Location, Project
from vb_django.serializers import LocationSerializer
from vb_django.permissions import IsOwnerOfProject
from vb_django.app.metadata import Metadata


class LocationView(viewsets.ViewSet):
    """
    The Location API endpoint viewset for managing user locations in the database.
    """
    serializer_class = LocationSerializer
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsOwnerOfProject]

    def list(self, request):
        """
        GET request that lists all the locations owned by the user.
        :param request: GET request
        :return: List of locations
        """
        locations = Location.objects.filter(owner=request.user)
        # TODO: Add ACL access objects
        serializer = self.serializer_class(locations, many=True)
        response_data = serializer.data
        for l in response_data:
            loc = Location.objects.get(pk=int(l["id"]))
            m = Metadata(loc, None)
            l["metadata"] = m.get_metadata("LocationMetadata")
        return Response(response_data, status=status.HTTP_200_OK)

    def create(self, request):
        """
        POST request that creates a new location.
        :param request: POST request
        :return: New location object
        """
        dataset_inputs = request.data.dict()
        serializer = self.serializer_class(data=dataset_inputs, context={'request': request})
        # TODO: Add project existence and ownership check
        if serializer.is_valid():
            location = serializer.save()
            location_data = serializer.data
            if "metadata" not in dataset_inputs.keys():
                dataset_inputs["metadata"] = None
            m = Metadata(location, dataset_inputs["metadata"])
            meta = m.set_metadata("LocationMetadata")
            if meta:
                location_data["metadata"] = meta
            if location:
                return Response(location_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        """
        PUT request to update an existing location.
        :param request: PUT request
        :param pk: Location ID
        :return:
        """
        dataset_inputs = request.data.dict()
        serializer = self.serializer_class(data=dataset_inputs, context={'request': request})
        if serializer.is_valid() and pk is not None:
            try:
                original_location = Location.objects.get(id=int(pk))
            except Location.DoesNotExist:
                return Response("No location found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if original_location.owner == request.user:
                location = serializer.update(original_location, serializer.validated_data)
                if location:
                    l = serializer.data
                    m = Metadata(location, dataset_inputs["metadata"])
                    meta = m.set_metadata("LocationMetadata")
                    if meta:
                        l["metadata"] = meta
                    request_status = status.HTTP_201_CREATED
                    if int(pk) == location.id:
                        request_status = status.HTTP_200_OK
                    return Response(l, status=request_status)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        """
        DEL request to delete an existing location.
        :param request: DEL request
        :param pk: Location ID
        :return:
        """
        if pk is not None:
            try:
                location = Location.objects.get(id=int(pk))
            except Location.DoesNotExist:
                return Response("No location found for id: {}".format(pk), status=status.HTTP_400_BAD_REQUEST)
            if location.owner == request.user:
                location.delete()
                return Response(status=status.HTTP_200_OK)
            else:
                return Response(status=status.HTTP_401_UNAUTHORIZED)
        return Response("No location 'id' in request.", status=status.HTTP_400_BAD_REQUEST)

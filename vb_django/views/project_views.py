from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
# from rest_framework.decorators import action
from vb_django.models import Project
from vb_django.serializers import ProjectSerializer
from vb_django.permissions import IsOwner
from vb_django.app.metadata import Metadata

# from vb_django.app.preprocessing import PPGraph
# from vb_django.task_controller import DaskTasks
# from vb_django.app.metadata import Metadata
# from django.core.exceptions import ObjectDoesNotExist
# from io import StringIO
# import pandas as pd
# import json


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

    def create(self, request):
        """
        POST request that creates a new project.
        :param request: POST request
        :return: New project object
        """
        dataset_inputs = request.data.dict()
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

    def update(self, request, pk=None):
        """
        PUT request for updating a project.
        :param request: PUT request
        :return: The updated/200
        """
        dataset_inputs = request.data.dict()
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

    # @action(detail=True, methods=["get"], name="Execute Pre-processing")
    # def run_preprocessing(self, request, pk=None):
    #     """
    #     Execute a preprocessing configuration on a specified dataset.
    #     :param request: GET request containing two parameters dataset_id and preprocessing_id
    #     :param pk: id of the workflow
    #     :return: complete pre-processing transformation of the dataset
    #     """
    #     if "dataset_id" in self.request.query_params.keys():
    #         try:
    #             dataset = Dataset.objects.get(id=int(self.request.query_params.get('dataset_id')))
    #         except Dataset.DoesNotExist:
    #             return Response("No dataset found for id: {}".format(int(self.request.query_params.get('dataset_id'))),
    #                             status=status.HTTP_400_BAD_REQUEST)
    #         if "preprocessing_id" in self.request.query_params.keys():
    #             try:
    #                 preprocess_config = PreProcessingConfig.objects.get(
    #                     id=int(self.request.query_params.get('preprocessing_id'))
    #                 )
    #             except PreProcessingConfig.DoesNotExist:
    #                 return Response(
    #                     "No preprocessing configuration found for id: {}".format(
    #                         int(self.request.query_params.get('preprocessing_id'))),
    #                     status=status.HTTP_400_BAD_REQUEST
    #                 )
    #             raw_data = pd.read_csv(StringIO(dataset.data.decode()))
    #             pp_configuration = json.loads(preprocess_config.config)
    #             result_string = StringIO()
    #             result = PPGraph(raw_data, pp_configuration).data
    #             result_columns = set.difference(set(result.columns), set(raw_data.columns))
    #             result = result[result_columns]
    #             result.to_csv(result_string)
    #             response_result = {"processed_data": result}
    #             return Response(response_result, status=status.HTTP_200_OK)
    #         else:
    #             return Response("No preprocessing 'preprocessing_id' in request.", status=status.HTTP_400_BAD_REQUEST)
    #     else:
    #         return Response("No dataset 'dataset_id' in request.", status=status.HTTP_400_BAD_REQUEST)
    #
    # @action(detail=False, methods=["post"], name="Execute technique for specified dataset, analytical model and preprocessing")
    # def execute(self, request):
    #     input_data = request.data.dict()
    #     required_parameters = ["workflow_id", "dataset_id", "model_id"]
    #     if set(required_parameters).issubset(input_data.keys()):
    #         try:
    #             workflow = Workflow.objects.get(id=int(input_data["workflow_id"]))
    #         except ObjectDoesNotExist:
    #             workflow = None
    #         try:
    #             amodel = AnalyticalModel.objects.get(id=int(input_data["model_id"]))
    #         except ObjectDoesNotExist:
    #             amodel = None
    #         try:
    #             dataset = Dataset.objects.get(id=int(input_data["dataset_id"]))
    #         except ObjectDoesNotExist:
    #             dataset = None
    #         if workflow is None or dataset is None or amodel is None:
    #             message = []
    #             if workflow is None:
    #                 message.append("No workflow found for id: {}".format(input_data["workflow_id"]))
    #             if dataset is None:
    #                 message.append("No dataset found for id: {}".format(input_data["dataset_id"]))
    #             if amodel is None:
    #                 message.append("No analytical model found for id: {}".format(input_data["model_id"]))
    #             return Response(", ".join(message), status=status.HTTP_400_BAD_REQUEST)
    #         elif IsOwnerOfLocationChild().has_object_permission(request, self, workflow):
    #             try:
    #                 DaskTasks.setup_task(dataset_id=dataset.id, amodel_id=amodel.id)
    #                 response = "Successfully executed analytical model"
    #             except Exception as ex:
    #                 response = "Error occured attempting to execute analytical model. Message: {}".format(ex)
    #             return Response(response, status=status.HTTP_200_OK)
    #         else:
    #             return Response(status=status.HTTP_401_UNAUTHORIZED)
    #     else:
    #         return Response(
    #             "Missing required parameters in POST request. Required parameters: {}".format(", ".join(required_parameters)),
    #             status.HTTP_400_BAD_REQUEST
    #         )
    #
    # @action(detail=False, methods=["POST"], name="Get the status/results of an executed task.")
    # def data(self, request):
    #     inputs = request.data.dict()
    #     required_parameters = ["workflow_id", "model_id"]
    #     if set(required_parameters).issubset(inputs.keys()):
    #         try:
    #             workflow = Workflow.objects.get(id=int(inputs["workflow_id"]))
    #         except ObjectDoesNotExist:
    #             workflow = None
    #         try:
    #             amodel = AnalyticalModel.objects.get(id=int(inputs["model_id"]))
    #         except ObjectDoesNotExist:
    #             amodel = None
    #         if workflow is None or amodel is None:
    #             message = []
    #             if workflow is None:
    #                 message.append("No workflow found for id: {}".format(inputs["workflow_id"]))
    #             if amodel is None:
    #                 message.append("No analytical model found for id: {}".format(inputs["model_id"]))
    #             return Response(", ".join(message), status=status.HTTP_400_BAD_REQUEST)
    #         elif IsOwnerOfLocationChild().has_object_permission(request, self, workflow):
    #             response = {}
    #             meta = Metadata(parent=amodel)
    #             metadata = meta.get_metadata("ModelMetadata", ['status', 'stage', 'message'])
    #             response["metadata"] = metadata
    #             completed = False
    #             if "stage" in metadata.keys():
    #                 i = metadata["stage"].split("/")
    #                 if int(i[0]) == int(i[1]):
    #                     completed = True
    #             if completed:
    #                 if amodel.model:
    #                     data = None
    #                     if "data" in inputs.keys():
    #                         data = pd.read_csv(StringIO(inputs["data"]))
    #                     response["data"] = DaskTasks.make_prediction(amodel.id, data)
    #                     response["dataset_id"] = amodel.dataset
    #             response["analytical_model_id"] = amodel.id
    #             response["workflow_id"] = workflow.id
    #             return Response(response, status=status.HTTP_200_OK)
    #     data = "Missing required parameters: {}".format(", ".join(required_parameters))
    #     response_status = status.HTTP_200_OK
    #     return Response(data, status=response_status)
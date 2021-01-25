from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from vb_django.models import Pipeline, Project, Dataset, Model
from vb_django.app.metadata import Metadata
from vb_django.serializers import PipelineSerializer
from vb_django.permissions import IsOwnerOfPipeline, IsOwnerOfProject, IsOwnerOfDataset, IsOwnerOfModel
from vb_django.task_controller import DaskTasks


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

    @action(detail=False, methods=["post"], name="Execute a pipeline")
    def execute(self, request):
        input_data = request.data.dict()
        required_parameters = ["project_id", "dataset_id", "pipeline_id"]
        if set(required_parameters).issubset(input_data.keys()):
            permissions = []
            try:
                project = Project.objects.get(id=int(input_data["project_id"]))
                if not IsOwnerOfProject().has_object_permission(request, self, project):
                    permissions.append("Unauthorized to access project.")
            except Project.DoesNotExist:
                project = None
            try:
                pipeline = Pipeline.objects.get(id=int(input_data["pipeline_id"]))
                if not IsOwnerOfPipeline().has_object_permission(request, self, pipeline):
                    permissions.append("Unauthorized to access pipeline")
            except Pipeline.DoesNotExist:
                pipeline = None
            try:
                dataset = Dataset.objects.get(id=int(input_data["dataset_id"]))
                if not IsOwnerOfDataset().has_object_permission(request, self, dataset):
                    permissions.append("Unauthorized to access dataset")
            except Dataset.DoesNotExist:
                dataset = None
            if len(permissions) > 0:
                return Response(permissions, status=status.HTTP_401_UNAUTHORIZED)
            if project is None or dataset is None or pipeline is None:
                message = []
                if project is None:
                    message.append("No project found for id: {}".format(input_data["project_id"]))
                if dataset is None:
                    message.append("No dataset found for id: {}".format(input_data["dataset_id"]))
                if pipeline is None:
                    message.append("No pipeline found for id: {}".format(input_data["pipeline_id"]))
                return Response(", ".join(message), status=status.HTTP_400_BAD_REQUEST)
            try:
                DaskTasks.setup_task(project_id=project.id, dataset_id=dataset.id, pipeline_id=pipeline.id)
                response = "Successfully executed pipeline"
            except Exception as ex:
                response = "Error occured attempting to execute pipeline. Message: {}".format(ex)
            return Response(response, status=status.HTTP_200_OK)

        else:
            return Response(
                "Missing required parameters in POST request. Required parameters: {}".format(", ".join(required_parameters)),
                status.HTTP_400_BAD_REQUEST
            )

    @action(detail=False, methods=["POST"], name="Get the status of an executed pipeline.")
    def status(self, request):
        input_data = request.data.dict()
        required_parameters = ["project_id", "pipeline_id"]
        if set(required_parameters).issubset(input_data.keys()):
            permissions = []
            try:
                project = Project.objects.get(id=int(input_data["project_id"]))
                if not IsOwnerOfProject().has_object_permission(request, self, project):
                    permissions.append("Unauthorized to access project.")
            except Project.DoesNotExist:
                project = None
            try:
                pipeline = Pipeline.objects.get(id=int(input_data["pipeline_id"]))
                if not IsOwnerOfPipeline().has_object_permission(request, self, pipeline):
                    permissions.append("Unauthorized to access pipeline")
            except Pipeline.DoesNotExist:
                pipeline = None
            if len(permissions) > 0:
                return Response(permissions, status=status.HTTP_401_UNAUTHORIZED)
            if pipeline is None or project is None:
                message = []
                if project is None:
                    message.append("No project found for id: {}".format(input_data["project_id"]))
                if pipeline is None:
                    message.append("No pipeline found for id: {}".format(input_data["pipeline_id"]))
                return Response(", ".join(message), status=status.HTTP_400_BAD_REQUEST)
            response = {}
            meta = Metadata(parent=pipeline)
            metadata = meta.get_metadata("PipelineMetadata", ['status', 'stage', 'message'])
            response["metadata"] = metadata
            completed = False
            if "stage" in metadata.keys():
                i = metadata["stage"].split("/")
                if int(i[0]) == int(i[1]):
                    completed = True
                if completed:
                    #TODO: Add additional pipeline completion results
                    models = Model.objects.filter(pipeline=pipeline)
                    model_details = []
                    for m in models:
                        model_details.append({
                            "id": m.id,
                            "name": m.name,
                            "description": m.description
                        })
                    response["models"] = model_details
                response["project_id"] = project.id
                response["pipeline_id"] = pipeline.id
            return Response(response, status=status.HTTP_200_OK)
        data = "Missing required parameters: {}".format(", ".join(required_parameters))
        response_status = status.HTTP_200_OK
        return Response(data, status=response_status)

    @action(detail=False, methods=["POST"], name="Make a prediction with a completed pipeline's model.")
    def predict(self, request):
        input_data = request.data.dict()
        required_parameters = ["project_id", "model_id"]
        if set(required_parameters).issubset(input_data.keys()):
            permissions = []
            try:
                project = Project.objects.get(id=int(input_data["project_id"]))
                if not IsOwnerOfProject().has_object_permission(request, self, project):
                    permissions.append("Unauthorized to access project.")
            except Project.DoesNotExist:
                project = None
            try:
                model = Model.objects.get(id=int(input_data["model_id"]))
                if not IsOwnerOfModel().has_object_permission(request, self, model):
                    permissions.append("Unauthorized to access pipeline")
            except Model.DoesNotExist:
                model = None
            if len(permissions) > 0:
                return Response(permissions, status=status.HTTP_401_UNAUTHORIZED)
            if model is None or project is None:
                message = []
                if project is None:
                    message.append("No project found for id: {}".format(input_data["project_id"]))
                if model is None:
                    message.append("No model found for id: {}".format(input_data["model_id"]))
                return Response(", ".join(message), status=status.HTTP_400_BAD_REQUEST)
            response = {}
            if "data" in input_data.keys():
                data = str(input_data["data"])
            else:
                data = None
            results = DaskTasks.make_prediction(project.id, model.id, data)
            response["project_id"] = project.id
            response["pipeline_id"] = model.pipeline.id
            response["model_id"] = model.id
            response["results"] = results
            return Response(response, status=status.HTTP_200_OK)
        data = "Missing required parameters: {}".format(", ".join(required_parameters))
        response_status = status.HTTP_200_OK
        return Response(data, status=response_status)

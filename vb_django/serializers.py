from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.validators import UniqueValidator
# import vb_django.vb_django.models as vb_models
# from vb_django.validation import Validator
from vb_django.models import Project, ProjectMetadata, Dataset, DatasetMetadata, AnalyticalModel, Location, \
    LocationMetadata, PreProcessingConfig, ModelResults, ModelMetadata, AccessControlList


class UserSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(
            required=True,
            validators=[UniqueValidator(queryset=User.objects.all())]
            )
    username = serializers.CharField(
            validators=[UniqueValidator(queryset=User.objects.all())]
            )
    password = serializers.CharField(min_length=8, write_only=True)

    def create(self, validated_data):
        user = User.objects.create_user(validated_data['username'], validated_data['email'], validated_data['password'])
        return user

    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'password']


class LocationSerializer(serializers.ModelSerializer):
    project_id = serializers.PrimaryKeyRelatedField(read_only=True)

    # def validate_points(self, validated_data):
    #     validated = True
    #     if not Validator.validate_point(validated_data["start_latitude"], validated_data["start_longitude"]):
    #         self.errors["start_point"] = "Start point coordinates, start_latitude/start_longitude, are invalid"
    #         validated = False
    #     if not Validator.validate_point(validated_data["end_latitude"], validated_data["end_longitude"]):
    #         self.errors["end_point"] = "End point coordinates, end_latitude/end_longitude, are invalid"
    #         validated = False
    #     if not Validator.validate_point(validated_data["o_latitude"], validated_data["o_longitude"]):
    #         self.errors["o_point"] = "Orientation point coordinates, o_latitude/o_longitude, are invalid"
    #         validated = False
    #     return validated
    #
    def check_integrity(self, location):
        can_update = True
        # a_models = AnalyticalModel.objects.filter(location_id=location.id)
        a_models = []
        for m in a_models:
            if m.model is not None:
                can_update = False
        return can_update

    def create(self, validated_data):
        location = None
        # validated = self.validate_points(validated_data)
        validated = True
        if validated:
            location = Location(**validated_data)
            location.save()
        return location

    def update(self, instance, validated_data):
        location = instance
        # validated = self.validate_points(validated_data)
        validated = True
        if validated:
            can_update = self.check_integrity(instance)
            if can_update:
                location = Location(**validated_data)
                location.id = instance.id
                location.save()
            else:
                location = Location(**validated_data)
                location.save()
        return location

    class Meta:
        model = Location
        fields = [
            "id", "project_id", "name", "description", "type"
        ]


class LocationMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = LocationMetadata
        fields = [
            "id", "location_id", "name", "value"
        ]


class ProjectSerializer(serializers.ModelSerializer):
    owner_id = serializers.PrimaryKeyRelatedField(read_only=True, default=serializers.CurrentUserDefault())

    def create(self, validated_data):
        project = Project(**validated_data)
        project.save()
        return project

    def check_integrity(self, workflow):
        can_update = True
        # a_models = AnalyticalModel.objects.filter(workflow__id=workflow.id)
        a_models = []
        for m in a_models:
            if m.model is not None:
                can_update = False
        return can_update

    def update(self, instance, validated_data):
        can_update = self.check_integrity(instance)
        project = Project(**validated_data)
        if can_update:
            project.id = instance.id
        project.save()
        return project

    class Meta:
        model = Project
        fields = [
            "id", "owner_id", "name", "description"
        ]


class ProjectMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectMetadata
        fields = [
            "id", "project_id", "name", "value"
        ]


class DatasetSerializer(serializers.ModelSerializer):
    owner_id = serializers.PrimaryKeyRelatedField(read_only=True, default=serializers.CurrentUserDefault())
    data = serializers.CharField()

    def check_integrity(self, project):
        can_update = True
        # a_models = AnalyticalModel.objects.filter(workflow_id=workflow.id)
        a_models = []
        for m in a_models:
            if m.model is not None:
                can_update = False
        return can_update

    def create(self, validated_data):
        if "data" in validated_data.keys():
            validated_data["data"] = str(validated_data["data"]).encode()
        dataset = Dataset(**validated_data)
        dataset.save()
        return dataset

    def update(self, instance, validated_data):
        if "data" in validated_data.keys():
            validated_data["data"] = str(validated_data["data"]).encode()
        dataset = Dataset(**validated_data)
        if self.check_integrity(dataset.owner_id):
            dataset.id = instance.id
        dataset.owner_id = instance.owner_id
        dataset.save()
        return dataset

    class Meta:
        model = Dataset
        fields = [
            "id", "owner_id", "location_id", "name", "description", "data"
        ]


class DatasetMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetMetadata
        fields = [
            "id", "dataset_id", "name", "value"
        ]


class AnalyticalModelSerializer(serializers.ModelSerializer):
    project_id = serializers.PrimaryKeyRelatedField(queryset=Project.objects.all())

    def create(self, validated_data):
        amodel = AnalyticalModel(**validated_data)
        amodel.save()
        return amodel

    def update(self, instance, validated_data):
        amodel = AnalyticalModel(**validated_data)
        if instance.model is None:
            amodel.id = instance.id
        amodel.project_id = instance.project_id
        amodel.save()
        return amodel

    class Meta:
        model = AnalyticalModel
        fields = [
            "id", "project_id", "name", "description", "variables", "dataset_id"
        ]


class PreProcessingConfigSerializer(serializers.ModelSerializer):

    class Meta:
        model = PreProcessingConfig
        fields = ["analytical_model_id", "id", "name", "config"]


class ModelMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetadata
        fields = [
            "id", "analytical_model_id", "name", "value"
        ]


class ModelResultsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelResults
        fields = [
            "id", "model_id", "dataset_id", "timestamp", "result"
        ]


class AccessControlListSerializer(serializers.ModelSerializer):
    class Meta:
        model = AccessControlList
        fields = [
            "owner_id", "target_user_id", "object_id", "object_type", "expiration", "access_type"
        ]

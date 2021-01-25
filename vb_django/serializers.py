from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.validators import UniqueValidator
from vb_django.models import Project, ProjectMetadata, Dataset, DatasetMetadata, Pipeline, Location, \
    LocationMetadata, PipelineMetadata, Model, ModelMetadata, AccessControlList, PipelineInstance, \
    PipelineInstanceParameters, PipelineInstanceMetadata
from vb_django.utilities import save_dataset, load_dataset


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
    owner = serializers.PrimaryKeyRelatedField(read_only=True, default=serializers.CurrentUserDefault())

    def create(self, validated_data):
        location = None
        validated = True
        if validated:
            location = Location(**validated_data)
            location.owner = self.context["request"].user
            location.save()
        return location

    def update(self, instance, validated_data):
        location = instance
        validated = True
        if validated:
            location = Location(**validated_data)
            location.id = instance.id
            location.owner = self.context["request"].user
            location.save()
        return location

    class Meta:
        model = Location
        fields = [
            "id", "owner", "name", "description", "type"
        ]


class LocationMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = LocationMetadata
        fields = [
            "id", "parent", "name", "value"
        ]


class ProjectSerializer(serializers.ModelSerializer):
    owner = serializers.PrimaryKeyRelatedField(read_only=True, default=serializers.CurrentUserDefault())

    def create(self, validated_data):
        project = Project(**validated_data)
        project.owner = self.context["request"].user
        project.save()
        return project

    def update(self, instance, validated_data):
        project = Project(**validated_data)
        project.id = instance.id
        project.owner = instance.owner
        project.save()
        return project

    class Meta:
        model = Project
        fields = [
            "id", "owner", "dataset", "location", "name", "description"
        ]


class ProjectMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectMetadata
        fields = [
            "id", "parent", "name", "value"
        ]


class DatasetSerializer(serializers.ModelSerializer):
    owner = serializers.PrimaryKeyRelatedField(read_only=True, default=serializers.CurrentUserDefault())
    data = serializers.CharField()

    def create(self, validated_data):
        if "data" in validated_data.keys():
            validated_data["data"] = save_dataset(str(validated_data["data"]))
        dataset = Dataset(**validated_data)
        dataset.owner = self.context["request"].user
        dataset.save()
        return dataset

    def update(self, instance, validated_data):
        if "data" in validated_data.keys():
            validated_data["data"] = save_dataset(str(validated_data["data"]))
        dataset = Dataset(**validated_data)
        dataset.id = instance.id
        dataset.owner = instance.owner
        dataset.save()
        return dataset

    class Meta:
        model = Dataset
        fields = [
            "id", "owner", "name", "description", "data"
        ]


class DatasetMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetMetadata
        fields = [
            "id", "parent", "name", "value"
        ]


class PipelineSerializer(serializers.ModelSerializer):
    project = serializers.PrimaryKeyRelatedField(queryset=Project.objects.all())

    def create(self, validated_data):
        pipeline = Pipeline(**validated_data)
        pipeline.save()
        return pipeline

    def update(self, instance, validated_data):
        pipeline = Pipeline(**validated_data)
        try:
            models = Model.objects.get(pipeline=pipeline)
        except Model.DoesNotExist:
            pipeline.id = instance.id
        pipeline.project = instance.project
        pipeline.save()
        return pipeline

    class Meta:
        model = Pipeline
        fields = [
            "id", "project", "name", "type", "description"
        ]


class PipelineMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = PipelineMetadata
        fields = [
            "id", "parent", "name", "value"
        ]


class ModelSerializer(serializers.ModelSerializer):
    experiment = serializers.PrimaryKeyRelatedField(read_only=True)

    def create(self, validated_data):
        model = Model(**validated_data)
        model.save()
        return model

    def update(self, instance, validated_data):
        model = Model(**validated_data)
        if instance.model is None:
            model.id = instance.id
        model.save()
        return model

    class Meta:
        model = Model
        fields = [
            "id", "project", "name", "description"
        ]


class ModelMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetadata
        fields = [
            "id", "parent", "name", "value"
        ]


class PipelineInstanceSerializer(serializers.ModelSerializer):
    def create(self, validated_data):
        pipeline = Pipeline(**validated_data)
        pipeline.save()
        return pipeline

    def update(self, instance, validated_data):
        pipeline = Pipeline(**validated_data)
        if instance.model is None:
            pipeline.id = instance.id
        pipeline.save()
        return pipeline

    class Meta:
        model = Model
        fields = [
            "id", "name", "ptype", "active", "description"
        ]


class PipelineInstanceParameterSerializer(serializers.ModelSerializer):
    pipeline = serializers.PrimaryKeyRelatedField(read_only=True)

    def create(self, validated_data):
        pipelinep = PipelineInstanceParameters(**validated_data)
        pipelinep.save()
        return pipelinep

    def update(self, instance, validated_data):
        pipelinep = PipelineInstanceParameters(**validated_data)
        if instance.model is None:
            pipelinep.id = instance.id
        pipelinep.save()
        return pipelinep

    class Meta:
        model = Model
        fields = [
            "id", "name", "vtype", "type", "value", "options"
        ]


class PipelineInstanceMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetadata
        fields = [
            "id", "parent", "name", "value"
        ]


class AccessControlListSerializer(serializers.ModelSerializer):
    class Meta:
        model = AccessControlList
        fields = [
            "owner", "target_user", "object", "object_type", "expiration", "access_type"
        ]

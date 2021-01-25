from rest_framework import permissions
from vb_django.models import Project, Model, Dataset


class IsOwner(permissions.BasePermission):
    """
    Checks if the user ia the owner of the object.
    """
    def has_object_permission(self, request, view, obj):
        return obj.owner == request.user


class IsOwnerOfProject(permissions.BasePermission):
    """
    Checks if the user is the owner of some specified project
    """
    def has_object_permission(self, request, view, obj):
        try:
            project = Project.objects.filter(id=obj.id)
        except Project.DoesNotExist:
            return False
        return project[0].owner == request.user


class IsOwnerOfDataset(permissions.BasePermission):
    """
    Checks if the user is the owner of some specified dataset
    """
    def has_object_permission(self, request, view, obj):
        try:
            dataset = Dataset.objects.filter(id=obj.id)
        except Project.DoesNotExist:
            return False
        return dataset[0].owner == request.user


class IsOwnerOfPipeline(permissions.BasePermission):
    """
    Checks if the user is the owner of the pipeline's project
    """
    def has_object_permission(self, request, view, obj):
        return obj.project.owner == request.user


class IsOwnerOfModel(permissions.BasePermission):
    """
    Checks if the user is the owner of the project for the model of this experiment
    """
    def has_object_permission(self, request, view, obj):
        return obj.model.experiement.project.owner == request.user


class HasModelIntegrity(permissions.BasePermission):
    """
    Checks if the object has been used for creating a model, where an update would degrade the integrity of the workflow
    """
    def has_object_permission(self, request, view, obj):
        existing_model = False
        models = Model.objects.filter(experiment=obj)
        for m in models:
            if m.model is not None:
                existing_model = True
        return existing_model

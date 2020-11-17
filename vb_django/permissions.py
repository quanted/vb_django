from rest_framework import permissions
from vb_django.models import AnalyticalModel, Location


class IsOwner(permissions.BasePermission):
    """
    Checks if the user ia the owner of the object.
    """
    def has_object_permission(self, request, view, obj):
        return obj.owner_id == request.user


class IsOwnerOfProjectChild(permissions.BasePermission):
    """
    Checks if the user is the owner of the parent project
    """
    def has_object_permission(self, request, view, obj):
        return obj.project.owner_id == request.user


class IsOwnerOfAnalyticalModelChild(permissions.BasePermission):
    """
    Checks if the user is the owner of the analytical model's parent project.
    """
    def has_object_permission(self, request, view, obj):
        return obj.analytical_model.project.owner_id == request.user


class HasModelIntegrity(permissions.BasePermission):
    """
    Checks if the object has been used for creating a model, where an update would degrade the integrity of the workflow
    """
    def has_object_permission(self, request, view, obj):
        existing_model = False
        models = AnalyticalModel.objects.filter(workflow__location__id=obj.id)
        for m in models:
            if m.model is not None:
                existing_model = True
        return existing_model

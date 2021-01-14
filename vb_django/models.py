from django.db import models
from django.contrib.auth.models import User


class Project(models.Model):
    owner_id = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)


class ProjectMetadata(models.Model):
    parent_id = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class Dataset(models.Model):
    owner_id = models.ForeignKey(User, on_delete=models.CASCADE)
    location_id = models.CharField(max_length=64, null=True)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)
    data = models.BinaryField()


class DatasetMetadata(models.Model):
    parent_id = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class Location(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)
    type = models.CharField(max_length=64)


class LocationMetadata(models.Model):
    parent_id = models.ForeignKey(Location, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class AnalyticalModel(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    type = models.CharField(max_length=16)
    description = models.CharField(max_length=512)
    variables = models.CharField(max_length=512, null=True, blank=True)        # serializable JSON
    model = models.BinaryField(null=True, blank=True)


class PreProcessingConfig(models.Model):
    analytical_model = models.ForeignKey(AnalyticalModel, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    config = models.CharField(max_length=512)


class ModelMetadata(models.Model):
    parent_id = models.ForeignKey(AnalyticalModel, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class ModelResults(models.Model):
    model_id = models.ForeignKey(AnalyticalModel, on_delete=models.CASCADE)
    dataset_id = models.CharField(max_length=64)
    timestamp = models.DateTimeField()
    result = models.FloatField()


class AccessControlList(models.Model):
    types = (
        ('Project', 'Project'),
        ('Location', 'Location'),
        ('AnalyticalModel', 'AnalyticalModel'),
        ('Dataset', 'Dataset')
    )
    a_types = (
        ('Read', 'Read'),
        ('Write', 'Write')
    )
    owner_id = models.ForeignKey(User, on_delete=models.CASCADE)
    target_user_id = models.CharField(max_length=32)
    object_id = models.CharField(max_length=32)
    object_type = models.CharField(max_length=16, choices=types)
    expiration = models.DateTimeField()
    access_type = models.CharField(max_length=8, choices=a_types)

from django.db import models
from django.contrib.auth.models import User


class Project(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    location = models.IntegerField(null=True)
    dataset = models.IntegerField(null=True)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1024)


class ProjectMetadata(models.Model):
    parent = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=1024)


class Dataset(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1024)
    data = models.BinaryField()


class DatasetMetadata(models.Model):
    parent = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=1024)


class Location(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1024)
    type = models.CharField(max_length=64)


class LocationMetadata(models.Model):
    parent = models.ForeignKey(Location, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=1024)


class Pipeline(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    type = models.CharField(max_length=16)
    description = models.CharField(max_length=1024)


class PipelineMetadata(models.Model):
    parent = models.ForeignKey(Pipeline, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=1024)


class Model(models.Model):
    pipeline = models.ForeignKey(Pipeline, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1024)
    model = models.BinaryField()


class ModelMetadata(models.Model):
    parent = models.ForeignKey(Model, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=1024)


class PipelineInstance(models.Model):
    # id = models.AutoField(max_length=16, primary_key=True)
    name = models.CharField(max_length=128)
    ptype = models.CharField(max_length=32)
    description = models.CharField(max_length=1024)
    active = models.BooleanField()


class PipelineInstanceParameters(models.Model):
    pipeline = models.ForeignKey(PipelineInstance, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    vtype = models.CharField(max_length=64)
    value = models.CharField(max_length=128)
    options = models.CharField(max_length=1024)


class PipelineInstanceMetadata(models.Model):
    parent = models.ForeignKey(PipelineInstance, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=1024)


class AccessControlList(models.Model):
    types = (
        ('Project', 'Project'),
        ('Location', 'Location'),
        ('Experiment', 'Experiment'),
        ('Dataset', 'Dataset'),
        ('Model', 'Model')
    )
    a_types = (
        ('Read', 'Read'),
        ('Write', 'Write')
    )
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    target_user = models.CharField(max_length=32)
    object = models.CharField(max_length=32)
    object_type = models.CharField(max_length=16, choices=types)
    expiration = models.DateTimeField()
    access_type = models.CharField(max_length=8, choices=a_types)

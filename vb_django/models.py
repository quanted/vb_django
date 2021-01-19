from django.db import models
from django.contrib.auth.models import User


class Project(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    location = models.IntegerField(null=True)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)


class ProjectMetadata(models.Model):
    parent = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class Dataset(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)
    data = models.BinaryField()


class DatasetMetadata(models.Model):
    parent = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class Location(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)
    type = models.CharField(max_length=64)


class LocationMetadata(models.Model):
    parent = models.ForeignKey(Location, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class Experiment(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    type = models.CharField(max_length=16)
    description = models.CharField(max_length=512)


class ExperimentMetadata(models.Model):
    parent = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class Model(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)
    model = models.BinaryField()


class ModelMetadata(models.Model):
    parent = models.ForeignKey(Model, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


class Pipeline(models.Model):
    id = models.CharField(max_length=16, primary_key=True)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=512)
    active = models.BooleanField()


class PipelineParameters(models.Model):
    pipeline = models.ForeignKey(Pipeline, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    vtype = models.CharField(max_length=64)
    value = models.CharField(max_length=128)
    options = models.CharField(max_length=512)


class PipelineMetadata(models.Model):
    parent = models.ForeignKey(Pipeline, on_delete=models.CASCADE)
    name = models.CharField(max_length=128)
    value = models.CharField(max_length=512)


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

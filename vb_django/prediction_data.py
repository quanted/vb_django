from vb_django.models import Project, Dataset, Pipeline, Model


class PredictionData:

    def __init__(self, project_id, model_id):
        self.project_id = project_id
        self.model_id = model_id
        self.project = Project.objects.get(pk=project_id)
        self.model = Model.objects.get(pk=model_id)

    def get_cv_predictions(self):
        data = self.model.predictandSave()
        return data



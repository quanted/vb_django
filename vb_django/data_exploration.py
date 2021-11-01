from vb_django.app.vb_helper import VBHelper
from vb_django.app.vb_summary import VBSummary
from vb_django.utilities import load_dataset
from vb_django.models import Project, Dataset
from vb_django.app.metadata import Metadata
import json
import pandas as pd


class DataExploration:

    def __init__(self, dataset_id, project_id):
        # TODO: replace the need for the project_id with providing the target variable
        self.dataset_id = dataset_id
        self.project_id = project_id
        self.dataset = Dataset.objects.get(pk=dataset_id)
        self.project = Project.objects.get(pk=project_id)

        self.df = load_dataset(dataset_id, self.dataset)
        self.project_metadata = Metadata(parent=Project.objects.get(id=project_id)).get_metadata("ProjectMetadata")
        self.target_label = "response" if "target" not in self.project_metadata.keys() else self.project_metadata["target"]
        self.features_label = None if "features" not in self.project_metadata.keys() else self.project_metadata["features"]

        self.y_df = self.df[self.target_label].to_frame()
        if self.features_label:
            self.features_list = json.loads(self.features_label.replace("\'", "\""))
            self.X_df = self.df[self.features_list]
        else:
            self.X_df = self.df.drop(self.target_label, axis=1)

        self.vbhelper = VBHelper(pipeline_id=-1)
        self.vbhelper.setData(X_df=self.X_df, y_df=self.y_df)

    def get_missing_vals(self):
        data = VBHelper.saveFullFloatXy(X_df=self.X_df, y_df=self.y_df, X_df_s=self.vbhelper.X_df_start_order, y_df_s=self.vbhelper.y_df_start_order)
        vbs = VBSummary()
        vbs.setData(data)
        return vbs.missingVals()

    def get_components(self, num_cols, keep_cats=False):
        try:
            if "," in num_cols:
                _num_cols = num_cols.split(",")
                num_cols = []
                for n in _num_cols:
                    num_cols.append(int(n))
            else:
                num_cols = [int(num_cols)]
        except Exception:
            num_cols = [1]
        data = VBHelper.saveFullFloatXy(X_df=self.X_df, y_df=self.y_df, X_df_s=self.vbhelper.X_df_start_order, y_df_s=self.vbhelper.y_df_start_order)
        vbs = VBSummary()
        vbs.setData(data)
        return vbs.viewComponents(num_cols=num_cols, keep_cats=keep_cats)

    def get_kerneldensity(self):
        data = VBHelper.saveFullFloatXy(X_df=self.X_df, y_df=self.y_df, X_df_s=self.vbhelper.X_df_start_order, y_df_s=self.vbhelper.y_df_start_order)
        vbs = VBSummary()
        vbs.setData(data)
        return vbs.kernelDensityPie()

    def get_dendrogram(self, linkage='ward', dist='spearmanr'):
        data = VBHelper.saveFullFloatXy(X_df=self.X_df, y_df=self.y_df, X_df_s=self.vbhelper.X_df_start_order, y_df_s=self.vbhelper.y_df_start_order)
        vbs = VBSummary()
        vbs.setData(data)
        return vbs.hierarchicalDendrogram(linkage=linkage, dist=dist)







from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from vb_django.app.vb_transformers import None_T, FeatureNameExtractor
import pandas as pd
import numpy as np
import logging


class MissingValHandler(BaseEstimator, TransformerMixin):
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#use-columntransformer-by-selecting-column-by-names
    def __init__(self, strategy='drop_row', transformer=None, cat_idx=None, ):
        self.strategy = strategy
        self.transformer = transformer
        self.cat_idx = cat_idx

    def fit(self, X, y):
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X)
        if self.cat_idx is None:
            self.X_dtypes_ = dict(X.dtypes)
            self.obj_idx_ = [i for i, (var, dtype) in enumerate(self.X_dtypes_.items()) if dtype == 'object']
        else:
            self.obj_idx_ = self.cat_idx
        self.float_idx_ = [i for i in range(X.shape[1]) if i not in self.obj_idx_]

        self.cat_list_ = [X.iloc[:, idx].unique() for idx in self.obj_idx_]
        x_nan_count = X.isnull().sum().sum()  # sums by column and then across columns
        try:
            y_nan_count = y.isnull().sum().sum()
        except:
            try:
                y_nan_count = np.isnan(y).sum()
            except:
                if not y is None:
                    y_nan_count = 'error'
                else:
                    pass

        return self

    def get_feature_names(self, input_features=None):
        return FeatureNameExtractor(self.T, input_features=input_features)

    def transform(self, X, y=None):
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X)

        cat_encoder = OneHotEncoder(categories=self.cat_list_, sparse=False, )  # drop='first'
        xvars = list(X.columns)
        if type(self.strategy) is str:
            if self.strategy == 'drop':
                assert False, 'develop drop columns with >X% missing vals then drop rows with missing vals'

            if self.strategy == 'pass-through':
                numeric_T = ('no_transform', None_T(), self.float_idx_)
                categorical_T = ('cat_drop_hot', cat_encoder, self.obj_idx_)
            if self.strategy == 'drop_row':
                X = X.dropna(axis=0)  # overwrite it

                numeric_T = ('no_transform', None_T(), self.float_idx_)
                categorical_T = ('cat_drop_hot', cat_encoder, self.obj_idx_)

            if self.strategy == 'impute_middle':
                numeric_T = ('num_imputer', SimpleImputer(strategy='mean'), self.float_idx_)
                cat_imputer = make_pipeline(SimpleImputer(strategy='most_frequent'), cat_encoder)
                categorical_T = ('cat_imputer', cat_imputer, self.obj_idx_)
            if self.strategy[:10] == 'impute_knn':
                if len(self.strategy) == 10:
                    k = 5
                else:
                    k = int(''.join([char for char in self.strategy[10:] if char.isdigit()]))  # extract k from the end
                numeric_T = ('num_imputer', KNNImputer(n_neighbors=k), self.float_idx_)
                cat_imputer = make_pipeline(SimpleImputer(strategy='most_frequent'), cat_encoder)
                categorical_T = ('cat_imputer', cat_imputer, self.obj_idx_)

        Tlist = [numeric_T, categorical_T]
        self.T = ColumnTransformer(transformers=Tlist)
        self.T.fit(X, y)
        X = self.T.transform(X)

        x_nan_count = np.isnan(X).sum()  # sums by column and then across columns
        return X

import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import re


class VBSummary:

    def __init__(self):
        self.full_X_float_df = None
        self.full_y_df = None
        self.X_nan_bool_df = None
        self.spear_xy = None
        self.r_list = None

    def setData(self, df_dict):
        self.full_X_float_df = pd.read_json(df_dict['full_float_X'])
        self.full_y_df = pd.read_json(df_dict['full_y'])
        self.X_nan_bool_df = pd.read_json(df_dict['X_nan_bool'])

    def viewComponents(self, num_cols, keep_cats=False):
        n = self.full_X_float_df.shape[0]
        k = self.full_X_float_df.shape[1]
        g = len(num_cols)
        data = {
            "n": n,
            "k": k,
            "g": g
        }
        X = self.full_X_float_df
        # data["X"] = X
        data["components"] = []

        for g_idx, col_count in enumerate(num_cols):
            keep_cols = self.getTopNCols(col_count, keep_cats=keep_cats)
            X_scaled_expanded = StandardScaler().fit_transform(X.loc[(slice(None), keep_cols)])
            X_orthog = PCA(n_components=3).fit_transform(X_scaled_expanded)
            data["components"].append({
                "g_idx": g_idx,
                "col_count": col_count,
                "keep_cols": keep_cols,
                "X_scaled_expanded": X_scaled_expanded,
                "X_orthog": X_orthog
            })
        return data

    def getTopNCols(self, n_cols, keep_cats=True):

        try:
            self.spear_xy, self.r_list
        except:
            self.spear_xy = []
            self.r_list = []

            for col in self.full_X_float_df.columns:
                r = spearmanr(self.full_y_df, self.full_X_float_df[col]).correlation
                self.spear_xy.append((r, col))
                self.r_list.append(r)
        if keep_cats:
            r_arr = np.array(self.r_list)
        else:
            r_arr = np.array([r for r, col in self.spear_xy if not re.search('__', col)])
        r_min = np.sort(np.abs(r_arr))[-n_cols]
        keep_cols = []
        for r, col in self.spear_xy:
            if np.abs(r) >= r_min:
                keep_cols.append(col)
        return keep_cols

    def kernelDensityPie(self):
        try:
            self.spear_xy, self.r_list
        except:
            _ = self.getTopNCols(1)
        spear_xy_indexed = [
            (np.abs(tup[0]), tup[1], i)
            for i, tup in enumerate(self.spear_xy)]
        abs_r_sort, col_sort, idx_sort = zip(
            *sorted(spear_xy_indexed, reverse=True))
        r_sort = [self.r_list[i] for i in idx_sort]

        all_vars = col_sort
        float_vars, float_idx = zip(*[(name, i) for i, name in enumerate(all_vars) if not re.search('__', name)])
        if len(float_vars) < len(all_vars):
            cat_vars, cat_idx_list = zip(*[(name, i) for i, name in enumerate(all_vars) if not name in float_vars])

            cat_var_dict = self.mergeCatVars(cat_vars)
            cat_group_names = list(cat_var_dict.keys())
        else:
            cat_vars = []
            cat_idx_list = []
            cat_var_dict = {}
            cat_group_names = []
        float_var_count = len(float_vars)
        total_var_count = float_var_count + len(cat_var_dict) + 1  # dep var too

        plot_cols = int(total_var_count ** 0.5)
        plot_rows = -(-total_var_count // plot_cols)  # ceiling divide
        data = {
            "float_vars": float_vars, "cat_vars": cat_vars, "dk": [], "pies": []
        }
        for name, idx in enumerate(float_vars):
            density = gaussian_kde(self.full_X_float_df.loc[:, [name]])
            r = round(r_sort[float_idx[idx]], 2)
            data["dk"].append({"name": name, "r": r, "density": density})
        for name, idx, in enumerate(cat_var_dict.keys()):
            cat_flavors, var_names = zip(*cat_var_dict[name])
            cum_r = np.sum(np.abs(np.array([r_sort[cat_idx_list[cat_vars.index(cat)]] for cat in var_names])))
            cat_df = self.full_X_float_df.loc[:, var_names]
            cat_df.columns = cat_flavors
            cat_shares = cat_df.sum()
            r = round(cum_r, 2)
            data["pies"].append({"name": name, "r": r, "data": cat_shares})
        return data

    def mergeCatVars(self, var_names):
        var_dict = {}
        for var in var_names:
            parts = re.split('__', var)
            if len(parts) > 2:
                parts = ['_'.join(parts[:-1]), parts[-1]]
            assert len(parts) == 2, f'problem with parts of {var}'
            if not parts[0] in var_dict:
                var_dict[parts[0]] = []
            var_dict[parts[0]].append((parts[1], var))
        return var_dict

    def missingVals(self):
        n = self.X_nan_bool_df.shape[0]
        if np.sum(self.X_nan_bool_df.to_numpy().ravel()) == 0:
            print(f'no missing values found')
            return
        data = {"n": n}
        nan_01 = self.X_nan_bool_df.to_numpy().astype(np.int16)
        data["nan_01"] = nan_01
        feature_names = self.X_nan_bool_df.columns.to_list()
        data["feature_names"] = feature_names
        feature_idx = np.arange(len(feature_names))
        data["feature_idx"] = feature_idx

        feat_miss_count_ser = self.X_nan_bool_df.astype(np.int16).sum(axis=0)
        data["feat_miss_count_ser"] = feat_miss_count_ser

        pct_missing_list = [f'{round(pct)}%' for pct in (100 * feat_miss_count_ser / n).tolist()]
        data["pct_missing_list"] = pct_missing_list

        row_miss_count_ser = feat_miss_count_ser = self.X_nan_bool_df.astype(np.int16).sum(axis=1)
        data["row_miss_count_ser"] = row_miss_count_ser

        nan_01_sum = nan_01.sum(axis=0)
        has_nan_features = nan_01_sum > 0
        nan_01_hasnan = nan_01[:, has_nan_features]
        hasnan_features = [name for i, name in enumerate(feature_names) if has_nan_features[i]]
        data["hasnan_features"] = hasnan_features
        nan_corr = self.pearsonCorrelationMatrix(nan_01_hasnan)
        data["nan_corr"] = nan_corr
        nan_corr_df = pd.DataFrame(data=nan_corr, columns=hasnan_features)
        self.nan_corr = nan_corr
        self.nan_corr_df = nan_corr_df
        corr_linkage = hierarchy.ward(nan_corr)
        dendro = hierarchy.dendrogram(  # just used for ordering the features by the grouping
            corr_linkage, labels=hasnan_features, ax=None, no_plot=True, leaf_rotation=90)
        data["dendro"] = dendro

        colors = [plt.get_cmap('plasma')(value) for value in [255]]
        data["colors"] = colors
        labels = ['missing data']
        patches = [Patch(color=colors[i], label=labels[i]) for i in [0]]
        data["patches"] = patches

        hasnan_feature_idx = np.arange(len(hasnan_features))
        data["hasnan_feature_idx"] = hasnan_feature_idx

        return data

    def hierarchicalDendrogram(self, linkage='ward', dist='spearmanr'):
        X = self.full_X_float_df  # .to_numpy()
        data = {}
        if dist.lower() == 'spearmanr':
            corr = spearmanr(X, nan_policy='omit').correlation
        elif dist.lower() == 'pearsonr':
            corr = self.pearsonCorrelationMatrix(X)
        else:
            assert False, 'distance not developed'
        if linkage.lower() == 'ward':
            corr_linkage = hierarchy.ward(corr)
        else:
            assert False, 'linkage not developed'
        dendro = hierarchy.dendrogram(
            corr_linkage, labels=X.columns.tolist(), ax=None, leaf_rotation=90, no_plot=True
        )
        dendro_idx = np.arange(0, len(dendro['ivl']))
        data["dendro"] = dendro
        data["dendro_idx"] = dendro_idx
        data["labels"] = X.columns.tolist()
        return data

    def pearsonCorrelationMatrix(self, Xdf):
        if type(Xdf) is pd.DataFrame:
            X = Xdf.to_numpy()
        else:
            X = Xdf
        cols = X.shape[1]
        corr_mat = np.empty((cols, cols))
        for c0 in range(cols):
            corr_mat[c0, c0] = 1
            for c1 in range(cols):
                if c0 < c1:
                    corr = pearsonr(X[:, c0], X[:, c1])[0]
                    if np.isnan(corr):
                        print(f'nan for {X[:, c0]} and {X[:, c1]}')
                    corr_mat[c0, c1] = corr
                    corr_mat[c1, c0] = corr
        return corr_mat
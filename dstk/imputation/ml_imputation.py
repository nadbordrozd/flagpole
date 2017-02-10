from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from .encoders import IdentityEncoder
from utils import missing_mask


class MLImputer(object):
    """generic class for imputing missing data using ML models.
    It uses a given regressor and a given classifier to impute numeric
    and categorical fields - respectively.
    For every column in the input dataframe a regressor (classifier) is
    trained on all the rows where this column is not missing. It is
    trained on all the remaining columns (if those other columns have
    missing values themselves, they are imputed using provided base_imputer.

    !!! it is assumed that missing values are encoded as NaN for float-y
    columns and as -1 for categorical (and boolean) columns. If they are not, a
    feature_encoder must be specified that will perform that encoding.
    """

    def __init__(self,
                 base_classifier=None,
                 base_regressor=None,
                 base_imputer=IdentityEncoder,
                 feature_encoder=IdentityEncoder()):
        """
        :param base_classifier: the sklearn-like classifier used to impute categorical columns
        :param base_regressor: the sklearn-like regressor used to impute continous columns
        :param base_imputer: the imputer used for the first crude run at imputation. It's only
            necessary when base_classifier or base_regressor can't handle missing values itself
            - which is the case for RandomForest. MasterExploder is a good default base_imputer
            in such cases. XGBoost regressors and classifiers don't need any base_imputer.
        :param feature_encoder: transformer (with .fit, .transform, .inverse_transform methods)
            that transforms a dataframe to a format where all categorical columns are integers
            (with missing values denoted by -1) and all continous columns are floats (with missing
            values denoted by np.NaN)
        """

        self.base_imputer = base_imputer
        self.feature_encoder = feature_encoder
        self.base_classifier = base_classifier
        self.base_regressor = base_regressor
        self.col2imputer = {}
        self.column_set = {}
        self.col2feats = {}
        self.col2type = {}

    def __str__(self):
        return "MLImputer(%s, %s, %s, %s)" % (
            self.base_classifier, self.base_regressor, self.base_imputer, self.feature_encoder)

    def fit(self, df, y=False):
        df = self.feature_encoder.fit(df).transform(df)
        self.column_set = set(df.columns)

        for col in self.column_set:
            other_cols = sorted(self.column_set.difference({col}))
            self.col2feats[col] = other_cols

        for col in self.column_set:
            column = df.get(col)
            if np.issubdtype(column.dtype, np.floating):
                self.col2type[col] = 'numeric'
            elif all(x in {-1, 0, 1, False, True} for x in column):
                self.col2type[col] = 'boolean'
            else:
                self.col2type[col] = 'integer'

            feats = self.col2feats[col]
            if np.issubdtype(column.dtype, np.floating):
                model = self.base_regressor()
            else:
                model = self.base_classifier()

            X = df.get(feats)[~missing_mask(column)].reset_index(drop=True)
            y = column[~missing_mask(column)].values
            if len(y) == 0:
                raise ValueError(
                    'need at least 1 nonmissing value to train imputer')

            imputer = Pipeline(
                [('encoder', self.base_imputer()), ('imputer', model)])
            self.col2imputer[col] = imputer.fit(X, y)
        return self

    def transform(self, df, proba=False):
        df = self.feature_encoder.transform(df)
        result_dict = {}
        for col, feats in self.col2feats.items():
            column = df[col]
            column_copy = np.copy(column)

            missing = missing_mask(column)
            if missing.any():
                if proba and self.col2type[col] == 'boolean':
                    predictions = self.col2imputer[col] \
                                      .predict_proba(
                        df.get(feats)[missing]
                        .reset_index(drop=True))[:, 1]
                    column_copy = column_copy + 0.0
                else:
                    predictions = self.col2imputer[col] \
                        .predict(
                        df.get(feats)[missing]
                            .reset_index(drop=True))

                column_copy[np.where(missing_mask(column))] = predictions

            result_dict[col] = column_copy

        return self.feature_encoder.inverse_transform(
            pd.DataFrame(result_dict))

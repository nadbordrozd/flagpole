"""this module implements basic encoders - sklearn-like transformers
that take a dataframe and transform it to a format compatible with sklearn
regressors and classifiers while handling missing values.
The encoder you will most likely want to use is MasterExploder.
For usage example see test_master_exploder at the bottom.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder
from pandas.util.testing import assert_frame_equal


class MissingNumericEncoder(object):
    """transformer that takes a numeric dataframe column (with missing values)
    and encodes it as 2 columns - first numeric with missing values replaced by
    the median; second boolean to indicate where there was a missing value.
    >>> df = pd.DataFrame({'a': np.array([np.nan, 1., 7., 4.5, 2., 9., np.nan]), 'b': [1]*7})
    >>> encoder = MissingNumericEncoder('a')
    >>> encoded = encoder.fit(df).transform(df)
    >>> type(encoded) == pd.DataFrame
    True
    >>> list(encoded.columns)
    ['a_filled_in', 'a_missing']
    >>> list(encoded.a_filled_in)
    [4.5, 1.0, 7.0, 4.5, 2.0, 9.0, 4.5]
    >>> list(encoded.a_missing)
    [True, False, False, False, False, False, True]
    """

    def __init__(self, column_name, strategy='median'):
        self.column_name = column_name
        self.imputer = Imputer(strategy=strategy)

    def fit(self, df, y=None):
        self.imputer.fit(df.get([self.column_name]))
        return self

    def transform(self, df):
        filled_in = self.imputer.transform(df.get([self.column_name]))
        missing = np.isnan(df.get([self.column_name]))
        return pd.DataFrame({
            (self.column_name + '_filled_in'): filled_in.ravel(),
            (self.column_name + '_missing'): missing.get(self.column_name)
        })


class MissingCategoricalEncoder(object):
    """transformer that encodes a single non-numeric column as integer
    This encoder only transforms one column of the input dataframe - specified
    in __init__, ignoring all the other columns. It returns a dataframe with
    a single column containing the encoded version of the original.
    None is treated like every other value.
    >>> df = pd.DataFrame({'a': np.array([None, 7, 7, 9, 7, 9, None]), 'b': [1]*7})
    >>> encoder = MissingCategoricalEncoder('a')
    >>> encoded = encoder.fit(df).transform(df)
    >>> type(encoded) == pd.DataFrame
    True
    >>> list(encoded.columns)
    ['a']
    >>> list(encoded.a)
    [0, 1, 1, 2, 1, 2, 0]
    """

    def __init__(self, column_name):
        self.column_name = column_name
        self.label_encoder = LabelEncoder()

    def fit(self, df, y=None):
        vals = list(df.get(self.column_name)) + [None]
        self.label_encoder.fit(vals)
        return self

    def transform(self, df):
        return pd.DataFrame({
            self.column_name: self.label_encoder.transform(df.get(self.column_name))
        })


class IdentityEncoder(object):
    """same interface as MissingCategoricalEncoder but doesn't do anything"""

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df.get(self.column_name)


class MasterExploder(object):
    """sklearn-like transformer that encodes all columns in a dataframe as either
    integer or numeric for use by sklearn algorithms. Handles missing values (crudely)
    Uses MissingNumericEncoder to encode numeric columns and MissingCategoricalEncoder
    for everything else. Since MissingNumericEncoder encodes a single column as 2, the
    output of MasterExploder may also have more columns than the input.
    Assumes that missing values are denoted by np.nan in numeric columns and by None in
    all other types."""

    def __init__(self, numeric_strategy='median', encode_categorical=False):
        self.num_strategy = numeric_strategy
        self.cat_encoder = MissingCategoricalEncoder if encode_categorical else IdentityEncoder
        self.columns = None
        self.column2encoder = None

    def fit(self, df, y=None):
        self.columns = df.columns
        self.column2encoder = {
            column_name:
                MissingNumericEncoder(column_name, strategy=self.num_strategy).fit(df)
                if df.get(column_name).dtype == np.dtype('float64')
                else self.cat_encoder(column_name).fit(df)
            for column_name in self.columns
            }
        return self

    def transform(self, df):
        return pd.concat([self.column2encoder[col].transform(df)
                          for col in self.columns], axis=1)


def test_master_exploder_encodes_ints_bools_floats_strings():
    T = True
    F = False
    N = None
    NaN = np.NaN
    S = 's'

    data = pd.DataFrame(dict(
        a=[T, N, T, T, F, F, F, N],
        b=[N, F, T, F, N, F, T, F],
        c=[T, T, F, N, N, T, F, F],
        d=np.array([NaN, NaN, 1.0, NaN, NaN, 2.5, 0.0, NaN]),
        e=[S, N, N, S, N, S, S, N]
    ))

    actual = MasterExploder(encode_categorical=True).fit(data).transform(data)
    expected = pd.DataFrame(dict(
        a=[2, 0, 2, 2, 1, 1, 1, 0],
        b=[0, 1, 2, 1, 0, 1, 2, 1],
        c=[2, 2, 1, 0, 0, 2, 1, 1],
        d_filled_in=[1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 0.0, 1.0],
        d_missing=[True, True, False, True, True, False, False, True],
        e=[1, 0, 0, 1, 0, 1, 1, 0]
    ))

    assert_frame_equal(actual, expected)


def test_master_exploder_ignores_categorical_columns_when_told():
    NaN = np.NaN

    data = pd.DataFrame(dict(
        a=[1, -1, 1, 1, 0, 0, 0, 1],
        b=[-1, 0, 1, 0, -1, 0, 1, 0],
        c=[1, 1, 0, -1, -1, 1, 0, 0],
        d=np.array([NaN, NaN, 1.0, NaN, NaN, 2.5, 0.0, NaN]),
        e=[0, -1, -1, 0, -1, 0, -1, 0],
        f=[1, 2, 3, 0, 0, 2, 2, 1]
    ))

    actual = MasterExploder(encode_categorical=False).fit(data).transform(data)
    expected = pd.DataFrame(dict(
        a=[1, -1, 1, 1, 0, 0, 0, 1],
        b=[-1, 0, 1, 0, -1, 0, 1, 0],
        c=[1, 1, 0, -1, -1, 1, 0, 0],
        d_filled_in=[1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 0.0, 1.0],
        d_missing=[True, True, False, True, True, False, False, True],
        e=[0, -1, -1, 0, -1, 0, -1, 0],
        f=[1, 2, 3, 0, 0, 2, 2, 1]
    ))

    assert_frame_equal(actual, expected)
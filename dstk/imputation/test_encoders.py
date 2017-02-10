import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np

from dstk.imputation.encoders import MasterExploder, StringFeatureEncoder


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


def test_StringFeatureEncoder_encodes_string_as_int():
    encoder = StringFeatureEncoder()
    df = pd.DataFrame({
        'a': ['a', 'b', 'b'],
        'b': [1, 2, 3],
        'c': [0.4, 0.1, 0.5],
        'd': ['x', 'y', 'z']
    })

    expected = pd.DataFrame({
        'a': [1, 2, 2],
        'b': [1, 2, 3],
        'c': [0.4, 0.1, 0.5],
        'd': [1, 2, 3]
    })
    actual = encoder.fit(df).transform(df)
    assert actual.equals(expected)


def test_StringFeatureEncoder_encodes_missing_val_as_minus_one():
    encoder = StringFeatureEncoder(missing_marker='IM_MISSING')
    df = pd.DataFrame({
        'a': ['a', 'b', 'IM_MISSING'],
        'b': [1, 2, 3],
        'c': [0.4, 0.1, 0.5],
        'd': ['IM_MISSING', 'IM_MISSING', 'z']
    })

    expected = pd.DataFrame({
        'a': [1, 2, -1],
        'b': [1, 2, 3],
        'c': [0.4, 0.1, 0.5],
        'd': [-1, -1, 1]
    })
    actual = encoder.fit(df).transform(df)
    assert actual.equals(expected)


def test_StringFeatureEncoder_decodes_encoded():
    encoder = StringFeatureEncoder(missing_marker='IM_MISSING')
    df = pd.DataFrame({
        'a': ['a', 'b', 'IM_MISSING'],
        'b': [1, 2, 3],
        'c': [0.4, 0.1, 0.5],
        'd': ['IM_MISSING', 'IM_MISSING', 'z']
    })

    encoded = encoder.fit(df).transform(df)
    assert df.equals(encoder.inverse_transform(encoded))

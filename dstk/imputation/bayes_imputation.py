"""
This is a template for missing value imputation with bayesian networks. By their nature, bayesian
nets require to be constructed separately for every new usecase - so this module can't provide a net
that would be usable right away for any future application. What it does provide is:
 - a template classBayesImputer that conforms to the same interface as other imputers in this
   package
 - a few utility functions specific to missing data (in addition to the stuff in dstk.pymc_utils)
 - example usage - see tests at the bottom
"""

import pandas as pd
import numpy as np


def predict(sampler, data):
    predicted = {}
    for node in sampler.observed_stochastics:
        feature_name = str(node)
        values = data[feature_name]
        missing_inds = np.where(pd.isnull(values))
        trace = sampler.trace(feature_name)[:]

        mean_posterior = values[:] + 0.0
        for samples, ind in zip(trace.T, missing_inds[0]):
            mean_posterior[ind] = samples.mean()

        predicted[feature_name] = mean_posterior

    return pd.DataFrame(predicted)


class BayesNetImputer(object):
    """superclass of bayes-net based models that for transforming data"""

    def __init__(self):
        self.sampler = None

    def fit(self, df, y=None):
        """fit your bayes net here"""
        raise NotImplementedError()

    def transform(self, df):
        return predict(self.sampler, df)

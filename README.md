# ds-tools
buncha utils for ml and ds

Installation:
```
sudo pip install git+git://github.com/nadbor-aimia/ds-tools.git
```

Usage:
```python
from dstk.imputation import wet_dataset

data = wet_dataset()
print data
```
prints 

```
   rain  some_numeric  sprinkler  wet_sidewalk
0     0           1.1          0             0
1     0           NaN          1             1
2     1           0.2          1             1
3     1          -0.4          0             1
4     1           0.1          1             1
5    -1           0.2          0             1
6     0           0.0          1            -1
7    -1           3.9         -1             0
```

In this example 'sprinkler' and 'rain' variables are meant to be independent random variables, while 'wet_sidewalk' is true iff 'rain' OR 'sprinkler' is true. 'some_numeric' is just a nonsense numeric column thrown in there for completeness.

By convention we use `-1` to denote a missing boolean or categorical value and `np.NaN` for missing numeric values.

### ML-based imputation
Now let's do some imputation:
```python
from dstk.imputation import DefaultImputer

imputer = DefaultImputer()
filled_in = imputer.fit(data).transform(data)

print filled_in
```
prints

```   
   rain  some_numeric  sprinkler  wet_sidewalk
0     0        1.1000          0             0
1     0        0.1475          1             1
2     1        0.2000          1             1
3     1       -0.4000          0             1
4     1        0.1000          1             1
5     1        0.2000          0             1
6     0        0.0000          1             1
7     0        3.9000          0             0
```

This default imputer uses XGBoost regressors and classifiers under the hood that are trained to fill in one column at a time. It should be sufficient in most applications. It is possible to replace Random Forests with any other sklearn-compatible algorithm - for example SVM - like this:

```python
from dstk.imputation import MLImputer
from sklearn.svm import SVC, SVR

svm_imputer = MLImputer(base_classifier=SVR, base_regressor=SVC)
```

### Bayes-Net imputation
To do imputation the bayesian way one needs to create an imputer that inherits from `dstk.imputation.BayesNetImputer` and override the `construct_net` method.

```python
import pymc

from dstk.imputation import mask_missing, BayesNetImputer
from dstk.pymc_utils import make_bernoulli, cartesian_child


class RSWImputer(BayesNetImputer):

    def construct_net(self, df):
        # pymc requires that missing data is represented by masked numpy arrays        
        # mask_missing converts a list of values with -1 or np.NaN denoting missing
        # into a masked array
        rain_data = mask_missing(df.rain)
        sprinkler_data = mask_missing(df.sprinkler)
        sidewalk_data = mask_missing(df.wet_sidewalk)

        # define the network
        # make_bernoulli creates a binary (observed in this case) random variable 'rain'
        # and also implicitly creates a parent for it 'p(rain)' - which is an unobserved
        # random variable with a uniform prior on the interval [0, 1]
        rain = make_bernoulli('rain', value=rain_data)
        # ditto for 'sprinkler'
        sprinkler = make_bernoulli('sprinkler', value=sprinkler_data)
        # cartesian child creates a child variable of 'sprinkler' and 'rain' with 
        # a conditional probability distribution given its parents. The child 'wet_sidewalk'
        # has a different probability for every combination of values of parents 
        # (which there are 4 of in this case). For every combination of parent values an
        # independent, unobserved random variable is created:
        # p(wet_sidewalk=1 | rain=0, sprinkler=0)
        # p(wet_sidewalk=1 | rain=0, sprinkler=1)
        # p(wet_sidewalk=1 | rain=1, sprinkler=0)
        # p(wet_sidewalk=1 | rain=1, sprinkler=1)
        sidewalk = cartesian_child('wet_sidewalk', parents=[rain, sprinkler],
                                   value=sidewalk_data)
    
        # pymc boilerplate. sorry :(
        model = pymc.Model([rain, sprinkler, sidewalk])
        sampler = pymc.MCMC(model)
        return sampler
```

and then use it the same way as the ML imputer:

```python
imputer = RSWImputer(iter=500, burn=300, thin=2)
filled_data = imputer.fit(data).transform(data)

print filled_data
```
prints
```
    rain  some_numeric  sprinkler  wet_sidewalk
0     0           1.1          0             0
1     0           NaN          1             1
2     1           0.2          1             1
3     1          -0.4          0             1
4     1           0.1          1             1
5     1           0.2          0             1
6     0           0.0          1             0
7     0           3.9          0             0
```
The fields that are not part of the network are simply passed unchanged - hence the NaN in 'some_numeric'. 
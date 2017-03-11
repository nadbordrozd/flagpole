# ds-tools
buncha utils for ml and ds

Installation:
```
sudo pip install git+git://github.com/nadbordrozd/ds-tools.git
```

Usage:
```python
from dstk.imputation import wet_dataset

data = wet_dataset()
print data
```
prints 

```
   rain  some_numeric some_string  sprinkler  wet_sidewalk
0     0           1.1           B          0             0
1     0           NaN           A          1             1
2     1           0.2           A          1             1
3     1          -0.4           A          0             1
4     1           0.1           A          1             1
5    -1           0.2           A          0             1
6     0           0.0           A          1            -1
7    -1           3.9     UNKNOWN         -1             0
```

In this example 'sprinkler' and 'rain' variables are meant to be independent random variables, while 'wet_sidewalk' is true iff 'rain' OR 'sprinkler' is true. 'some_numeric' and 'some_string' are just nonsense numeric and string columns thrown in there for completeness.

All the imputers expect data in a the form of a pandas DataFrame, where:

- numeric columns are treated as continuous variables and can have missing values denoted `np.NaN`
- iteger columns are treated as categorical variables and must have values from `-1` to `n - 1`, where `n` is the number of classes. `-1` is used to denote a missing value.
- string  columns are treated as categorical. Missing values can be denoted by any string specified by user.

### ML-based imputation
Now let's do some imputation:
```python
from dstk.imputation import DefaultImputer
 
imputer = DefaultImputer(missing_string_marker='UNKNOWN')  # treat 'UNKNOWN' as missing value
filled_in = imputer.fit(data).transform(data)

print filled_in
```
prints

```   
   rain  some_numeric some_string  sprinkler  wet_sidewalk
0     0       1.10000           B          0             0
1     0       0.00611           A          1             1
2     1       0.20000           A          1             1
3     1      -0.40000           A          0             1
4     1       0.10000           A          1             1
5     0       0.20000           A          0             1
6     0       0.00000           A          1             1
7     0       3.90000           A          1             0
```

This default imputer uses XGBoost regressors and classifiers under the hood that are trained to fill in one column at a time. It should be sufficient in most applications. It is possible to replace XGBoost with any other sklearn-compatible algorithm - for example SVM - like this:

```python
from dstk.imputation import MLImputer, MasterExploder, StringFeatureEncoder
from sklearn.svm import SVC, SVR

svm_imputer = MLImputer(
    base_classifier=SVR, 
    base_regressor=SVC, 
    base_imputer=MasterExploder, 
    feature_encoder=StringFeatureEncoder(missing_marker='UNKNOWN'))
```

Here `MasterExploder` is a simple imputer that imputes missing values with median. It is necessary
as a preprocessing step because SVM can't handle missing features. The default imputer doesn't need
this intermediate step because XGBoost directly deals with missing features.

### Bayes-Net imputation
To do imputation the bayesian way one needs to create an imputer that inherits from `dstk.imputation.BayesNetImputer` and override the `construct_net` method.

```python
import pymc

from dstk.imputation import mask_missing, BayesNetImputer
from dstk.pymc_utils import make_bernoulli, cartesian_child


class RSWImputer(BayesNetImputer):

    def construct_net(self, df):
        # define the network
        # make_bernoulli creates a binary (observed in this case) random variable 'rain'
        # and also implicitly creates a parent for it 'p(rain)' - which is an unobserved
        # random variable with a uniform prior on the interval [0, 1]
        rain = make_bernoulli('rain', value=df.rain)
        # ditto for 'sprinkler'
        sprinkler = make_bernoulli('sprinkler', value=df.sprinkler)
        # cartesian child creates a child variable of 'sprinkler' and 'rain' with 
        # a conditional probability distribution given its parents. The child 'wet_sidewalk'
        # has a different probability for every combination of values of parents 
        # (which there are 4 of in this case). For every combination of parent values an
        # independent, unobserved random variable is created:
        # p(wet_sidewalk=1 | rain=0 sprinkler=0)
        # p(wet_sidewalk=1 | rain=0 sprinkler=1)
        # p(wet_sidewalk=1 | rain=1 sprinkler=0)
        # p(wet_sidewalk=1 | rain=1 sprinkler=1)
        sidewalk = cartesian_child('wet_sidewalk', parents=[rain, sprinkler],
                                   value=df.wet_sidewalk)
    
        return pymc.Model([rain, sprinkler, sidewalk])
```

and then use it the same way as the ML imputer:

```python
imputer = RSWImputer(iter=500, burn=300, thin=2)
filled_data = imputer.fit(data).transform(data)

print filled_data
```
prints
```
    rain  some_numeric some_string  sprinkler  wet_sidewalk
0     0           1.1           B          0             0
1     0           NaN           A          1             1
2     1           0.2           A          1             1
3     1          -0.4           A          0             1
4     1           0.1           A          1             1
5     1           0.2           A          0             1
6     0           0.0           A          1             1
7     1           3.9     UNKNOWN          1             0
```
The fields that are not part of the network are simply passed unchanged - hence the NaN in 'some_numeric'. 

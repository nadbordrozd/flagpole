# ds-tools
buncha utils for ml and ds

Installation:
```
sudo pip install git+git://github.com/nadbor-aimia/ds-tools.git
```

Usage:
```
from dstk.imputation import sample_dataset

data = sample_dataset()
print data
```
prints 

```
   a  b  c     d  e
0  1 -1  1   NaN  3
1  1  0  0   NaN -1
2  1  1  0  1.00 -1
3  1  0 -1   NaN  3
4  0 -1 -1   NaN -1
5  0  0  1  2.14  3
6  0  1  0  0.00  3
7  1  0  0   NaN -1
```

By convention we use `-1` to denote a missing boolean or categorical value and `np.NaN` for missing numeric values.

Now let's do some imputation:
```
from dstk.imputation import DefaultImputer

imputer = DefaultImputer()
filled_in = imputer.fit(data).transform(data)

print filled_in
```
prints

```
   a  b  c      d  e
0  1  0  1  1.698  3
1  1  0  0  1.456  3
2  1  1  0  1.000  3
3  1  0  1  1.256  3
4  0  0  0  1.156  3
5  0  0  1  2.140  3
6  0  1  0  0.000  3
7  1  0  0  1.456  3
```

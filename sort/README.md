```python
from sklearn.utils.linear_assignment_ import linear_assignment
```
in [original code](https://github.com/nwojke/deep_sort/blob/master/deep_sort/linear_assignment.py) is deprecated and will be removed, so we change it to 
```python
from scipy.optimize import linear_sum_assignment
```

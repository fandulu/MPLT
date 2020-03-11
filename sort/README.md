```python
from sklearn.utils.linear_assignment_ import linear_assignment
indices = linear_assignment(cost_matrix)
for row, col in indices:
  ...
```
in [original code](https://github.com/nwojke/deep_sort/blob/master/deep_sort/linear_assignment.py) is deprecated and will be removed, so we change it to 
```python
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost_matrix)
for row, col in zip(row_ind,col_ind):
   ...
```

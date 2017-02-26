"""Softmax.

softmax is a fuction for calculating linear regression outputs:
for vector Y = [y_1, y_2, ... y_n]

softmax y_n = e^y_n / sum(e^y_i)
"""


scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if len(x) == 0:
	return None
    x = np.array(x)
    if x.ndim == 1:
       return softmax1d(x)
    # note: each COLUMN represents a sample, so we take one column as single input
    return np.apply_along_axis(softmax1d, 0, x)

def softmax1d(x): 
    # print 'input 1d', x
    n = len(x)
    e = np.exp(x)
    s = np.sum(e)
    ret = np.zeros(n)
    for i, val in enumerate(e):
      ret[i] = val / s
    # print 'return values', ret
    return ret


def test():
  print 'softmax test'

  scores = [1.0, 2.0, 3.0]
  print 'scores:', scores
  print 'expected: [ 0.09003057  0.24472847  0.66524096]'
  print 'actual:  ', softmax(scores)
  print 

  scores = np.array([[1, 2, 3, 6],
            [2, 4, 5, 6],
            [3, 8, 7, 6]])
  expected = np.array([[ 0.09003057,  0.00242826,  0.01587624,  0.33333333],
              [ 0.24472847,  0.01794253,  0.11731043,  0.33333333],
              [ 0.66524096,  0.97962921,  0.86681333 , 0.33333333]])
  print 'scores:', scores
  print 'expected: ', expected
  print 'actual:   ', softmax(scores)


#test()
print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
# x returns range [-2.0 - 6.0] with step 0.1
x = np.arange(-2.0, 6.0, 0.1)
# np.ones_linke returns array of ones, with same size as x
# vstack arranges the arrays in sequence vertically (row wise == one below another)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# diagram shows the values of softmax, for tuple (x e [-2, 6], 0.2, 1).
# Note how values of 1 and 0.2 decrease their share, as x is growing.
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()




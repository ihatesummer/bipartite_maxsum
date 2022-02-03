from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 1.5, 2, 3])
bins = np.linspace(0, 8, 41)
print(bins)
plt.hist(a, bins, rwidth = 0.8)
plt.show()
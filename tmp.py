from scipy.optimize import linear_sum_assignment
import numpy as np

w = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0.5]])

print(w)

print(linear_sum_assignment(w, maximize=True))